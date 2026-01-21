r"""Triton forward kernels for streaming Semi-CRF.

This module contains the Triton kernels for the forward pass (log and max semiring)
and the launcher function that allocates buffers and dispatches the kernels.

The kernels implement streaming edge computation where edge potentials are
computed on-the-fly from cumulative scores via prefix-sum decomposition:

.. math::
    \text{edge}[c_{\text{dst}}, c_{\text{src}}] =
    (\text{cum\_scores}[t+k, c_{\text{dst}}] - \text{cum\_scores}[t, c_{\text{dst}}])
    + \text{duration\_bias}[k, c_{\text{dst}}]
    + \text{transition}[c_{\text{src}}, c_{\text{dst}}]

Functions:
    launch_streaming_triton_kernel: Main entry point for launching Triton forward kernels.
"""

import torch

from .constants import NEG_INF
from .pytorch_reference import _compute_checkpoint_interval

# Triton is optional
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def _next_power_of_2(n: int) -> int:
    r"""_next_power_of_2(n) -> int

    Return the smallest power of 2 greater than or equal to n.

    This is used for padding tensor dimensions to powers of 2, which is
    required for efficient Triton kernel execution.

    Args:
        n (int): Input value.

    Returns:
        int: Smallest power of 2 >= n.

    Examples::

        >>> _next_power_of_2(5)
        8
        >>> _next_power_of_2(8)
        8
        >>> _next_power_of_2(24)
        32
    """
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    p = 1
    while p < n:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def semi_crf_streaming_scan_kernel(
        # Inputs
        cum_scores_ptr,  # (batch, T+1, C) - cumulative projected scores
        transition_ptr,  # (C, C) or (K, C, C) - transition matrix
        duration_bias_ptr,  # (K, C) - duration-specific bias
        lengths_ptr,  # (batch,) - sequence lengths
        # Boundary projections (optional, may be null if HAS_BOUNDARIES=False)
        proj_start_ptr,  # (batch, T, C) - start boundary scores
        proj_end_ptr,  # (batch, T, C) - end boundary scores
        # Outputs
        out_ptr,  # (batch,) - partition function
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer (read/write)
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints for backward
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length (T, not T+1)
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,  # interval for saving ring buffer
        NUM_CKPTS: tl.constexpr,  # number of checkpoints
        HAS_BOUNDARIES: tl.constexpr,  # whether boundary projections are provided
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C) or (K, C, C)
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
        # Strides for proj_start/proj_end (batch, T, C) - only used if HAS_BOUNDARIES
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        # Strides for ring buffer (batch, K, C_PAD)
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        # Strides for ring checkpoints (batch, num_ckpts, K, C_PAD)
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
    ):
        r"""Streaming Semi-CRF forward scan with on-the-fly edge computation (log semiring).

        This Triton kernel computes the forward pass of the Semi-CRF using logsumexp
        reductions. Edge potentials are computed on-the-fly from cumulative scores:

        .. code-block:: text

            edge[c_dst, c_src] = (cum_scores[t+k, c_dst] - cum_scores[t, c_dst])
                               + duration_bias[k, c_dst]
                               + transition[c_src, c_dst]

        Memory: Uses a ring buffer for alpha values with :math:`O(KC)` memory.
        Saves ring buffer checkpoints at regular intervals for backward pass.

        One program is launched per batch element (grid size = batch_size).
        """
        NEG_INF: tl.constexpr = -1e9

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        # 2D indices for (C_dst, C_src) operations
        c_dst_idx = tl.arange(0, C_PAD)[:, None]  # (C_PAD, 1)
        c_src_idx = tl.arange(0, C_PAD)[None, :]  # (1, C_PAD)
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Load transition matrix into registers: (C_PAD, C_PAD)
        # For static transitions (C, C): load once here
        # For duration-dependent (K, C, C): load inside k-loop
        # transition[c_src, c_dst] -> we need transition.T for edge computation
        # So we load transition_ptr[c_dst, c_src] effectively
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )  # (C_PAD, C_PAD) - this is transition.T

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        # Track final alpha for each batch element
        final_alpha = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        # Main forward loop: t = 1, 2, ..., T
        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            active = t <= seq_len

            # Accumulate alpha[t] = logsumexp over (k, c_src)
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Loop over valid segment durations k = 1, 2, ..., min(K-1, t)
            # tl.maximum ensures K=1 processes at least one duration
            for k in tl.range(1, tl.maximum(K, 2)):
                k_valid = (k <= t) & (k <= K - 1)
                start_pos = t - k

                # Ring index for alpha[start_pos]
                ring_k_idx = start_pos % K

                # Load alpha_prev from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )  # (C_PAD,) - alpha[start_pos, c_src]

                # === Compute edge block on-the-fly (prefix-sum) ===

                # Load cum_scores[t, :] and cum_scores[start_pos, :]
                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                # Content score = cum_scores[t, c_dst] - cum_scores[start, c_dst]
                content_score = cum_end - cum_start  # (C_PAD,)

                # Load duration bias
                dur_bias = tl.load(
                    duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                # Segment score = content_score + duration_bias
                segment_score = content_score + dur_bias  # (C_PAD,)

                # Add boundary scores if provided
                # Segment starts at start_pos, ends at t-1 (inclusive)
                if HAS_BOUNDARIES:
                    # proj_start[start_pos, :] - start boundary score
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    # proj_end[t-1, :] - end boundary score (t-1 is last position in segment)
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                # Edge block: edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst]
                # segment_score is (C_PAD,), expand to (C_PAD, 1) for c_dst
                # transition_block is (C_PAD, C_PAD) as transition.T

                # For duration-dependent transitions, load transition[k] inside the loop
                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + k * stride_tr_k
                        + c_dst_idx * stride_tr_dst
                        + c_src_idx * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )  # (C_PAD, C_PAD) - transition[k].T

                edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)

                # === Compute scores and reduction ===
                # scores[c_dst, c_src] = alpha_prev[c_src] + edge[c_dst, c_src]
                scores = alpha_prev[None, :] + edge_block  # (C_PAD, C_PAD)

                # Mask out invalid entries
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Logsumexp over c_src (axis=1) -> (C_PAD,)
                max_scores = tl.max(scores, axis=1)
                score_for_k = max_scores + tl.log(
                    tl.sum(tl.exp(scores - max_scores[:, None]), axis=1)
                )

                # Mask invalid durations and labels
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Accumulate into alpha_t via logsumexp
                max_alpha = tl.maximum(alpha_t, score_for_k)
                alpha_t = max_alpha + tl.log(
                    tl.exp(alpha_t - max_alpha) + tl.exp(score_for_k - max_alpha)
                )

            # Mask inactive sequences
            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            # Checkpoint i stores the ring buffer state at position i * CHECKPOINT_INTERVAL
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                # Save entire ring buffer to checkpoint
                for k_save in tl.range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    # Only save if checkpoint index is valid
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_save * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            is_final = t == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Final reduction: logsumexp over labels
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        max_val = tl.max(final_alpha_masked, axis=0)
        exp_fa = tl.where(c_mask, tl.exp(final_alpha - max_val), 0.0)
        sum_exp = tl.sum(exp_fa, axis=0)
        partition = max_val + tl.log(sum_exp)

        # Store result
        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_scan_kernel_max(
        # Same signature as log kernel
        cum_scores_ptr,
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,
        lengths_ptr,
        # Boundary projections (optional)
        proj_start_ptr,
        proj_end_ptr,
        out_ptr,
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer
        ring_ckpt_ptr,
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        HAS_BOUNDARIES: tl.constexpr,
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
    ):
        r"""Streaming Semi-CRF forward scan with max semiring (Viterbi decoding).

        Same structure as :func:`semi_crf_streaming_scan_kernel` but uses max
        instead of logsumexp for reductions. This implements the Viterbi algorithm
        for finding the most likely segmentation.

        See :func:`semi_crf_streaming_scan_kernel` for detailed documentation of
        parameters and the edge computation formula.
        """
        NEG_INF: tl.constexpr = -1e9

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]
        c_src_idx = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        seq_len = tl.load(lengths_ptr + batch_idx)

        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Load static transitions once (duration-dependent loaded inside k-loop)
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        final_alpha = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            active = t <= seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # tl.maximum ensures K=1 processes at least one duration
            for k in tl.range(1, tl.maximum(K, 2)):
                k_valid = (k <= t) & (k <= K - 1)
                start_pos = t - k
                ring_k_idx = start_pos % K

                # Load from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )

                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                content_score = cum_end - cum_start
                dur_bias = tl.load(
                    duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )
                segment_score = content_score + dur_bias

                # Add boundary scores if provided
                # Segment starts at start_pos, ends at t-1 (inclusive)
                if HAS_BOUNDARIES:
                    # proj_start[start_pos, :] - start boundary score
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    # proj_end[t-1, :] - end boundary score (t-1 is last position in segment)
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                # For duration-dependent transitions, load transition[k] inside the loop
                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + k * stride_tr_k
                        + c_dst_idx * stride_tr_dst
                        + c_src_idx * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )  # (C_PAD, C_PAD) - transition[k].T

                edge_block = segment_score[:, None] + transition_block

                scores = alpha_prev[None, :] + edge_block
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max semiring: max over c_src
                score_for_k = tl.max(scores, axis=1)
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Max semiring: max over k
                alpha_t = tl.maximum(alpha_t, score_for_k)

            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                for k_save in tl.range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_save * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            is_final = t == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Max semiring: max over labels
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)

    def launch_streaming_triton_kernel(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""launch_streaming_triton_kernel(cum_scores, transition, duration_bias, lengths, K, semiring="log", checkpoint_interval=None, proj_start=None, proj_end=None) -> tuple[Tensor, Tensor, int]

        Launch the streaming Triton kernel with proper buffer allocation.

        This function allocates the required buffers (ring buffer, checkpoints)
        and dispatches the appropriate Triton kernel based on the semiring.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            K (int): Maximum segment duration.
            semiring (str, optional): ``"log"`` or ``"max"``. Default: ``"log"``
            checkpoint_interval (int, optional): Interval for saving ring buffer.
                If ``None``, uses :math:`\sqrt{T \times K}`. Default: ``None``
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``

        Returns:
            tuple[Tensor, Tensor, int]: Tuple of:
                - **partition** (Tensor): Partition function values of shape
                  :math:`(\text{batch},)`.
                - **ring_checkpoints** (Tensor): Saved ring buffer states of shape
                  :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
                - **checkpoint_interval** (int): Actual interval used.
        """
        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        device = cum_scores.device
        dtype = cum_scores.dtype

        # Compute checkpoint interval if not provided
        if checkpoint_interval is None:
            checkpoint_interval = _compute_checkpoint_interval(T, K)
        else:
            checkpoint_interval = max(checkpoint_interval, K)

        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Determine if boundaries are provided
        has_boundaries = proj_start is not None and proj_end is not None

        # Determine if duration-dependent transitions (Phase 4A)
        has_duration_transitions = transition.ndim == 3

        # Ensure inputs are contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
        else:
            # Create dummy tensor for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]  # Reuse cum_scores memory, won't be accessed
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0  # Strides don't matter when not used

        # Allocate outputs
        partition = torch.empty(batch, device=device, dtype=dtype)

        # Live ring buffer (will be L1/L2 cached for small K*C)
        # Initialize to NEG_INF, then set k=0 to 0.0 (initial alpha state)
        ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
        ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

        # Checkpoint storage for backward pass
        # Initialize to NEG_INF, then set checkpoint 0, k=0 to 0.0
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )
        ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        # Handle transition strides for both (C, C) and (K, C, C)
        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0  # Not used for static transitions
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()

        # Launch kernel with device context for multi-GPU support
        grid = (batch,)
        kernel = (
            semi_crf_streaming_scan_kernel
            if semiring == "log"
            else semi_crf_streaming_scan_kernel_max
        )
        with torch.cuda.device(device):
            kernel[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                proj_start,
                proj_end,
                partition,
                ring_buffer,
                ring_checkpoints,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                has_boundaries,  # HAS_BOUNDARIES constexpr
                has_duration_transitions,  # HAS_DURATION_TRANSITIONS constexpr
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
                stride_tr_k,
                stride_tr_src,
                stride_tr_dst,
                stride_db_k,
                stride_db_c,
                stride_ps_b,
                stride_ps_t,
                stride_ps_c,
                stride_ring_b,
                stride_ring_k,
                stride_ring_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
            )

        # Trim padding from checkpoints for return
        ring_checkpoints = ring_checkpoints[:, :, :, :C]

        return partition, ring_checkpoints, checkpoint_interval
