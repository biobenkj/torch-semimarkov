r"""Triton backward kernel for streaming Semi-CRF.

This module contains the Triton kernel for the backward pass and the launcher
function that allocates buffers and dispatches the kernel.

The backward pass uses the forward-backward algorithm with checkpointing:

1. **Phase 1**: Recompute alpha values from saved ring buffer checkpoints
2. **Phase 2**: Compute beta backward while accumulating gradients

Gradients are computed via marginal probabilities:

.. math::
    P(\text{segment}[t, k, c_{\text{dst}}, c_{\text{src}}]) =
    \frac{\exp(\alpha[t, c_{\text{src}}] + \text{edge} + \beta[t+k, c_{\text{dst}}])}
    {\exp(\log Z)}

Functions:
    launch_streaming_triton_backward: Main entry point for launching backward kernel.
"""

import torch

from .constants import NEG_INF
from .triton_forward import _next_power_of_2

# Triton is optional
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:

    @triton.jit
    def semi_crf_streaming_backward_kernel(
        # Inputs (from forward)
        cum_scores_ptr,  # (batch, T+1, C)
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,  # (K, C)
        lengths_ptr,  # (batch,)
        log_Z_ptr,  # (batch,) - partition function values
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints from forward
        grad_output_ptr,  # (batch,) - upstream gradient
        # Boundary projections (optional, may be null if HAS_BOUNDARIES=False)
        proj_start_ptr,  # (batch, T, C) - start boundary scores
        proj_end_ptr,  # (batch, T, C) - end boundary scores
        # Working memory
        alpha_buffer_ptr,  # (batch, SEGMENT_SIZE, C_PAD) - recomputed alpha
        beta_ring_ptr,  # (batch, K, C_PAD) - beta ring buffer
        # Outputs (gradients)
        grad_cum_scores_ptr,  # (batch, T+1, C)
        grad_tr_workspace_ptr,  # (batch, C, C) or (batch, K, C, C) - per-batch accumulator
        grad_db_workspace_ptr,  # (batch, K, C) - per-batch accumulator, no atomic needed
        grad_proj_start_ptr,  # (batch, T, C) - gradient for proj_start (only if HAS_BOUNDARIES)
        grad_proj_end_ptr,  # (batch, T, C) - gradient for proj_end (only if HAS_BOUNDARIES)
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        SEGMENT_SIZE: tl.constexpr,  # = CHECKPOINT_INTERVAL + K
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
        # Strides for ring checkpoints (batch, num_ckpts, K, C_PAD)
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        # Strides for alpha buffer (batch, SEGMENT_SIZE, C_PAD)
        stride_ab_b,
        stride_ab_t,
        stride_ab_c,
        # Strides for beta ring (batch, K, C_PAD)
        stride_br_b,
        stride_br_k,
        stride_br_c,
        # Strides for grad_cum_scores (batch, T+1, C)
        stride_gcs_b,
        stride_gcs_t,
        stride_gcs_c,
        # Strides for grad_tr_workspace (batch, C, C) or (batch, K, C, C)
        stride_gtw_b,
        stride_gtw_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_gtw_src,
        stride_gtw_dst,
        # Strides for grad_db_workspace (batch, K, C)
        stride_gdbw_b,
        stride_gdbw_k,
        stride_gdbw_c,
    ):
        r"""Streaming Semi-CRF backward kernel with gradient computation.

        Computes gradients via the forward-backward algorithm:

        1. Recompute alpha from checkpoints (segment by segment)
        2. Compute beta backward while accumulating gradients

        Marginal probability:

        .. math::
            P(\text{segment}) = \exp(\alpha + \text{edge} + \beta - \log Z)

        Gradient accumulation uses atomic operations for shared parameters.

        .. important::
            **Gradient Scaling Semantics**

            There's a subtle difference in how gradients are scaled for per-batch
            vs shared parameters:

            - **Per-batch parameters** (``cum_scores``): Each batch element's gradient
              contribution is scaled by its corresponding ``grad_output[batch_idx]``.
              This happens INSIDE the kernel.

            - **Shared parameters** (``transition``, ``duration_bias``): These are
              accumulated across all batch elements WITHOUT per-element scaling. The
              scaling by ``grad_output`` happens AFTER the kernel via einsum.

            This matches PyTorch's backward semantics:

            .. code-block:: text

                grad_transition = einsum("bij, b -> ij", marginals, grad_output)

        One program is launched per batch element (grid size = batch_size).
        """
        NEG_INF: tl.constexpr = -1e9

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # Label indices
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]  # (C_PAD, 1)
        c_src_idx = tl.arange(0, C_PAD)[None, :]  # (1, C_PAD)
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        # Load batch-specific values
        seq_len = tl.load(lengths_ptr + batch_idx)
        log_Z = tl.load(log_Z_ptr + batch_idx)
        grad_out = tl.load(grad_output_ptr + batch_idx)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b
        alpha_buf_base = alpha_buffer_ptr + batch_idx * stride_ab_b
        beta_ring_base = beta_ring_ptr + batch_idx * stride_br_b
        grad_cs_base = grad_cum_scores_ptr + batch_idx * stride_gcs_b
        grad_tr_ws_base = grad_tr_workspace_ptr + batch_idx * stride_gtw_b
        grad_db_ws_base = grad_db_workspace_ptr + batch_idx * stride_gdbw_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b
            grad_ps_base = grad_proj_start_ptr + batch_idx * stride_ps_b
            grad_pe_base = grad_proj_end_ptr + batch_idx * stride_ps_b

        # Load transition matrix into registers (only for static transitions)
        # Duration-dependent transitions are loaded inside the k-loops
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )  # (C_PAD, C_PAD) - this is transition.T

        # Initialize beta ring buffer at final positions
        final_pos = seq_len
        final_ring_idx = final_pos % K
        # Note: Use tl.range (not static_range) to avoid compile-time explosion for large K
        for k_init in tl.range(0, K):
            is_final = k_init == final_ring_idx
            init_val = tl.where(is_final & c_mask, 0.0, NEG_INF)
            tl.store(
                beta_ring_base + k_init * stride_br_k + c_idx * stride_br_c,
                init_val,
                mask=c_mask,
            )

        # Process segments in reverse order
        # Note: Use tl.range to avoid compile-time unrolling for large NUM_CKPTS
        for ckpt_idx_loop in tl.range(0, NUM_CKPTS):
            ckpt_idx = NUM_CKPTS - 1 - ckpt_idx_loop
            seg_start = ckpt_idx * CHECKPOINT_INTERVAL
            seg_end = (ckpt_idx + 1) * CHECKPOINT_INTERVAL
            if seg_end > T:
                seg_end = T

            # Only process segments within sequence length
            if seg_start < seq_len:
                # === Phase 1: Recompute alpha for this segment ===
                # Load ring buffer state from checkpoint
                # Then recompute forward through the segment

                # Initialize alpha from checkpoint (stores ring buffer state at seg_start)
                for k_slot in tl.range(0, K):
                    alpha_val = tl.load(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_slot * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    # Store alpha[seg_start + k_slot - (seg_start % K)] if valid
                    # For simplicity, store at position 0 for initial ring state
                    if k_slot == seg_start % K:
                        tl.store(
                            alpha_buf_base + 0 * stride_ab_t + c_idx * stride_ab_c,
                            alpha_val,
                            mask=c_mask,
                        )

                # Recompute alpha values from seg_start+1 to seg_end
                # Note: Use tl.range to avoid compile-time unrolling for large SEGMENT_SIZE
                for local_t in tl.range(1, SEGMENT_SIZE):
                    t = seg_start + local_t
                    # Only process if within segment and sequence bounds
                    if t < seg_end and t < seq_len:
                        alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

                        # Loop over valid durations (tl.maximum ensures K=1 works)
                        for k in tl.range(1, tl.maximum(K, 2)):
                            start_pos = t - k
                            # Only process valid start positions
                            if start_pos >= 0:
                                # Get alpha_prev - either from buffer or checkpoint
                                local_start = start_pos - seg_start
                                if local_start >= 0 and local_start < SEGMENT_SIZE:
                                    alpha_prev = tl.load(
                                        alpha_buf_base
                                        + local_start * stride_ab_t
                                        + c_idx * stride_ab_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )
                                else:
                                    # Position is before seg_start, get from current checkpoint
                                    # The checkpoint at ckpt_idx contains alpha[seg_start-K+1..seg_start]
                                    # at ring indices (seg_start-K+1) % K .. seg_start % K
                                    prev_ring_idx = start_pos % K
                                    alpha_prev = tl.load(
                                        ring_ckpt_base
                                        + ckpt_idx * stride_ckpt_n
                                        + prev_ring_idx * stride_ckpt_k
                                        + c_idx * stride_ckpt_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )

                                # Compute edge on-the-fly
                                cum_end = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start = tl.load(
                                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score = cum_end - cum_start

                                dur_bias = tl.load(
                                    duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                segment_score = content_score + dur_bias

                                # Add boundary scores if provided
                                # Segment starts at start_pos, ends at t-1 (inclusive)
                                if HAS_BOUNDARIES:
                                    start_score = tl.load(
                                        proj_start_base
                                        + start_pos * stride_ps_t
                                        + c_idx * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = t - 1
                                    end_score = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score = segment_score + start_score + end_score

                                # Load k-indexed transition for duration-dependent case
                                if HAS_DURATION_TRANSITIONS:
                                    transition_block = tl.load(
                                        transition_ptr
                                        + k * stride_tr_k
                                        + c_dst_idx * stride_tr_dst
                                        + c_src_idx * stride_tr_src,
                                        mask=c_mask_2d,
                                        other=0.0,
                                    )

                                edge_block = segment_score[:, None] + transition_block

                                scores = alpha_prev[None, :] + edge_block
                                scores = tl.where(c_mask_2d, scores, NEG_INF)

                                # Logsumexp over c_src
                                max_scores = tl.max(scores, axis=1)
                                score_for_k = max_scores + tl.log(
                                    tl.sum(tl.exp(scores - max_scores[:, None]), axis=1) + 1e-10
                                )
                                score_for_k = tl.where(c_mask, score_for_k, NEG_INF)

                                # Accumulate via logsumexp
                                max_alpha = tl.maximum(alpha_t, score_for_k)
                                alpha_t = max_alpha + tl.log(
                                    tl.exp(alpha_t - max_alpha)
                                    + tl.exp(score_for_k - max_alpha)
                                    + 1e-10
                                )

                        # Store recomputed alpha
                        alpha_t = tl.where(c_mask, alpha_t, NEG_INF)
                        tl.store(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            alpha_t,
                            mask=c_mask,
                        )

                # === Phase 2: Compute beta backward and gradients ===
                # Note: Use tl.range to avoid compile-time unrolling for large CHECKPOINT_INTERVAL
                for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
                    t = seg_end - 1 - t_offset
                    # Only process valid positions
                    if t >= seg_start and t < seq_len and t >= 0:
                        # Get alpha[t] from buffer
                        local_t = t - seg_start
                        alpha_t = tl.load(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )

                        # Compute beta[t] and gradients
                        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

                        # tl.maximum ensures K=1 processes at least one duration
                        for k in tl.range(1, tl.maximum(K, 2)):
                            end_pos = t + k
                            # Only process valid end positions
                            if end_pos <= seq_len and end_pos <= T:
                                # Get beta[end_pos] from ring buffer
                                end_ring_idx = end_pos % K
                                beta_next = tl.load(
                                    beta_ring_base
                                    + end_ring_idx * stride_br_k
                                    + c_idx * stride_br_c,
                                    mask=c_mask,
                                    other=NEG_INF,
                                )

                                # Compute edge on-the-fly
                                cum_end = tl.load(
                                    cum_scores_base + end_pos * stride_cs_t + c_idx * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score = cum_end - cum_start

                                dur_bias = tl.load(
                                    duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                segment_score = content_score + dur_bias

                                # Add boundary scores if provided
                                # In Phase 2: segment starts at t, ends at end_pos-1
                                if HAS_BOUNDARIES:
                                    start_score = tl.load(
                                        proj_start_base + t * stride_ps_t + c_idx * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = end_pos - 1
                                    end_score = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score = segment_score + start_score + end_score

                                # Load k-indexed transition for duration-dependent case
                                if HAS_DURATION_TRANSITIONS:
                                    transition_block = tl.load(
                                        transition_ptr
                                        + k * stride_tr_k
                                        + c_dst_idx * stride_tr_dst
                                        + c_src_idx * stride_tr_src,
                                        mask=c_mask_2d,
                                        other=0.0,
                                    )

                                edge_block = (
                                    segment_score[:, None] + transition_block
                                )  # (C_PAD, C_PAD)

                                # === Compute marginal ===
                                # log_marginal[c_dst, c_src] = alpha[t, c_src] + edge[c_dst, c_src] + beta[end, c_dst] - log_Z
                                log_marginal = (
                                    alpha_t[None, :]  # (1, C_PAD) for c_src
                                    + edge_block  # (C_PAD, C_PAD)
                                    + beta_next[:, None]  # (C_PAD, 1) for c_dst
                                    - log_Z
                                )
                                marginal = tl.exp(log_marginal)  # (C_PAD, C_PAD)
                                marginal = tl.where(c_mask_2d, marginal, 0.0)

                                # === Accumulate gradients ===
                                # Note: For shared parameters (transition, duration_bias), we accumulate
                                # unscaled marginals. The scaling by grad_output.sum() is done after the
                                # kernel to match PyTorch's backward semantics.
                                # For per-batch parameters (cum_scores), we scale by grad_out here.

                                # grad_cum_scores: positive at end_pos, negative at t
                                # Scale by upstream gradient for per-batch tensor
                                marginal_sum_src = tl.sum(
                                    marginal, axis=1
                                )  # sum over c_src -> (C_PAD,)
                                marginal_sum_src = tl.where(c_mask, marginal_sum_src, 0.0)
                                marginal_sum_src_scaled = marginal_sum_src * grad_out

                                tl.atomic_add(
                                    grad_cs_base + end_pos * stride_gcs_t + c_idx * stride_gcs_c,
                                    marginal_sum_src_scaled,
                                    mask=c_mask,
                                )
                                tl.atomic_add(
                                    grad_cs_base + t * stride_gcs_t + c_idx * stride_gcs_c,
                                    -marginal_sum_src_scaled,
                                    mask=c_mask,
                                )

                                # grad_transition: write to per-batch workspace (unscaled marginal)
                                # marginal is (C_dst, C_src), grad_transition[c_src, c_dst] += marginal[c_dst, c_src]
                                marginal_T = tl.trans(marginal)  # (C_src, C_dst) = (C_PAD, C_PAD)
                                # Compute 2D offsets for workspace
                                # For duration-dependent: grad_tr_workspace[batch, k, src, dst]
                                # For static: grad_tr_workspace[batch, src, dst]
                                # c_dst_idx is (C_PAD, 1) for src, c_src_idx is (1, C_PAD) for dst
                                if HAS_DURATION_TRANSITIONS:
                                    tr_offsets_ws = (
                                        k * stride_gtw_k
                                        + c_dst_idx * stride_gtw_src
                                        + c_src_idx * stride_gtw_dst
                                    )
                                else:
                                    tr_offsets_ws = (
                                        c_dst_idx * stride_gtw_src + c_src_idx * stride_gtw_dst
                                    )
                                # atomic_add still needed within batch (multiple t iterations add to same location)
                                # but no inter-batch contention since each batch has its own workspace slice
                                tl.atomic_add(
                                    grad_tr_ws_base + tr_offsets_ws, marginal_T, mask=c_mask_2d
                                )

                                # grad_duration_bias: write to per-batch workspace (unscaled)
                                # grad_db_workspace[batch, k, c_dst] += sum over c_src
                                tl.atomic_add(
                                    grad_db_ws_base + k * stride_gdbw_k + c_idx * stride_gdbw_c,
                                    marginal_sum_src,
                                    mask=c_mask,
                                )

                                # grad_proj_start and grad_proj_end (per-batch, scaled)
                                # Segment starts at t, ends at end_pos-1
                                if HAS_BOUNDARIES:
                                    # grad_proj_start[t, c_dst] += marginal_sum_src * grad_out
                                    tl.atomic_add(
                                        grad_ps_base + t * stride_ps_t + c_idx * stride_ps_c,
                                        marginal_sum_src_scaled,
                                        mask=c_mask,
                                    )
                                    # grad_proj_end[end_pos-1, c_dst] += marginal_sum_src * grad_out
                                    tl.atomic_add(
                                        grad_pe_base
                                        + (end_pos - 1) * stride_ps_t
                                        + c_idx * stride_ps_c,
                                        marginal_sum_src_scaled,
                                        mask=c_mask,
                                    )

                                # === Update beta contribution ===
                                # beta[t, c_src] = logsumexp over (k, c_dst) of edge[c_dst, c_src] + beta[end, c_dst]
                                scores_for_beta = edge_block + beta_next[:, None]  # (C_dst, C_src)
                                scores_for_beta = tl.where(c_mask_2d, scores_for_beta, NEG_INF)

                                # Logsumexp over c_dst (axis 0)
                                max_beta_k = tl.max(scores_for_beta, axis=0)
                                beta_k = max_beta_k + tl.log(
                                    tl.sum(tl.exp(scores_for_beta - max_beta_k[None, :]), axis=0)
                                    + 1e-10
                                )
                                beta_k = tl.where(c_mask, beta_k, NEG_INF)

                                # Accumulate into new_beta via logsumexp over k
                                max_new = tl.maximum(new_beta, beta_k)
                                new_beta = max_new + tl.log(
                                    tl.exp(new_beta - max_new) + tl.exp(beta_k - max_new) + 1e-10
                                )

                        # Store beta[t] to ring buffer
                        t_ring_idx = t % K
                        tl.store(
                            beta_ring_base + t_ring_idx * stride_br_k + c_idx * stride_br_c,
                            new_beta,
                            mask=c_mask,
                        )

    def launch_streaming_triton_backward(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        log_Z: torch.Tensor,
        ring_checkpoints: torch.Tensor,
        checkpoint_interval: int,
        grad_output: torch.Tensor,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""launch_streaming_triton_backward(cum_scores, transition, duration_bias, lengths, log_Z, ring_checkpoints, checkpoint_interval, grad_output, proj_start=None, proj_end=None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

        Launch the Triton backward kernel with proper buffer allocation.

        This function allocates working memory (alpha buffer, beta ring buffer)
        and dispatches the backward kernel. Gradients for shared parameters
        are accumulated per-batch then reduced via einsum.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            log_Z (Tensor): Partition values from forward of shape :math:`(\text{batch},)`.
            ring_checkpoints (Tensor): Saved ring buffer states of shape
                :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
            checkpoint_interval (int): Interval used during forward pass.
            grad_output (Tensor): Upstream gradient of shape :math:`(\text{batch},)`.
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of gradients:
                - **grad_cum_scores** (Tensor): Shape :math:`(\text{batch}, T+1, C)`.
                - **grad_transition** (Tensor): Shape :math:`(C, C)` or :math:`(K, C, C)`.
                - **grad_duration_bias** (Tensor): Shape :math:`(K, C)`.
                - **grad_proj_start** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
                  if boundaries provided.
                - **grad_proj_end** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
                  if boundaries provided.
        """
        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        K = duration_bias.shape[0]
        device = cum_scores.device
        dtype = cum_scores.dtype

        num_checkpoints = ring_checkpoints.shape[1]
        C_PAD = _next_power_of_2(C)

        # Compute segment size for alpha buffer
        segment_size = checkpoint_interval + K

        # Determine if boundaries are provided
        has_boundaries = proj_start is not None and proj_end is not None

        # Determine if duration-dependent transitions (Phase 4A)
        has_duration_transitions = transition.ndim == 3

        # Ensure contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()
        log_Z = log_Z.contiguous()
        grad_output = grad_output.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
            # Allocate gradient outputs for boundaries
            grad_proj_start = torch.zeros(batch, T, C, device=device, dtype=dtype)
            grad_proj_end = torch.zeros(batch, T, C, device=device, dtype=dtype)
        else:
            # Create dummy tensors for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0
            grad_proj_start = None
            grad_proj_end = None
            # Kernel uses proj_start/proj_end directly when HAS_BOUNDARY_PROJ=False

        # Pad checkpoints to C_PAD
        if ring_checkpoints.shape[-1] < C_PAD:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints
        else:
            ring_ckpts_padded = ring_checkpoints.contiguous()

        # Allocate working memory
        alpha_buffer = torch.full((batch, segment_size, C_PAD), NEG_INF, device=device, dtype=dtype)
        beta_ring = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)

        # Allocate gradient outputs
        grad_cum_scores = torch.zeros(batch, T_plus_1, C, device=device, dtype=dtype)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=dtype)

        # Allocate per-batch workspace buffers to avoid atomic add contention
        # Each batch element accumulates to its own slice, then we sum after kernel
        if has_duration_transitions:
            # Duration-dependent: (batch, K, C, C)
            grad_tr_workspace = torch.zeros(batch, K, C, C, device=device, dtype=dtype)
        else:
            # Static: (batch, C, C)
            grad_tr_workspace = torch.zeros(batch, C, C, device=device, dtype=dtype)
        grad_db_workspace = torch.zeros(batch, K, C, device=device, dtype=dtype)

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        # Handle transition strides for both (C, C) and (K, C, C)
        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0  # Not used for static transitions
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_ckpts_padded.stride()
        stride_ab_b, stride_ab_t, stride_ab_c = alpha_buffer.stride()
        stride_br_b, stride_br_k, stride_br_c = beta_ring.stride()
        stride_gcs_b, stride_gcs_t, stride_gcs_c = grad_cum_scores.stride()

        # Handle grad_tr_workspace strides for both shapes
        if has_duration_transitions:
            stride_gtw_b, stride_gtw_k, stride_gtw_src, stride_gtw_dst = grad_tr_workspace.stride()
        else:
            stride_gtw_k = 0  # Not used for static transitions
            stride_gtw_b, stride_gtw_src, stride_gtw_dst = grad_tr_workspace.stride()

        stride_gdbw_b, stride_gdbw_k, stride_gdbw_c = grad_db_workspace.stride()

        # Use actual gradients or dummies for kernel call
        grad_ps_for_kernel = grad_proj_start if has_boundaries else grad_cum_scores
        grad_pe_for_kernel = grad_proj_end if has_boundaries else grad_cum_scores

        # Launch kernel with device context for multi-GPU support
        grid = (batch,)
        with torch.cuda.device(device):
            semi_crf_streaming_backward_kernel[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                log_Z,
                ring_ckpts_padded,
                grad_output,
                proj_start,
                proj_end,
                alpha_buffer,
                beta_ring,
                grad_cum_scores,
                grad_tr_workspace,
                grad_db_workspace,
                grad_ps_for_kernel,
                grad_pe_for_kernel,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                segment_size,
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
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                stride_ab_b,
                stride_ab_t,
                stride_ab_c,
                stride_br_b,
                stride_br_k,
                stride_br_c,
                stride_gcs_b,
                stride_gcs_t,
                stride_gcs_c,
                stride_gtw_b,
                stride_gtw_k,
                stride_gtw_src,
                stride_gtw_dst,
                stride_gdbw_b,
                stride_gdbw_k,
                stride_gdbw_c,
            )

        # Compute weighted sum of per-batch gradients for shared parameters.
        #
        # Correct gradient semantics for shared parameters:
        #   grad_θ = Σ_b[grad_output[b] × Σ_{t,k}(marginal[b,t,k])]
        #
        # NOT the buggy formula:
        #   grad_θ = Σ_{b,t,k}(marginal[b,t,k]) × Σ_b(grad_output[b])  # WRONG!
        #
        # The difference matters when grad_output varies across batch elements
        # (e.g., masked sequences, weighted losses). With uniform grad_output=[1,1,...],
        # both formulas happen to give the same result, which is why tests using
        # .sum().backward() didn't catch the bug.
        #
        # We use einsum for memory efficiency: it fuses the multiply + reduce
        # without creating a large intermediate tensor. For K=1024, C=64, batch=16,
        # the naive broadcast approach would allocate ~268MB just to sum immediately.
        #
        # Notation: b=batch, k=duration, i=src_state, j=dst_state, c=state
        if has_duration_transitions:
            grad_transition = torch.einsum("bkij, b -> kij", grad_tr_workspace, grad_output)
        else:
            grad_transition = torch.einsum("bij, b -> ij", grad_tr_workspace, grad_output)

        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace, grad_output)

        return grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end
