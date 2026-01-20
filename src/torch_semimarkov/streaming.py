r"""Golden Rule Streaming API for memory-efficient Semi-CRF.

This module implements the "Golden Rule" optimization for Semi-CRF inference:
edge potentials are computed on-the-fly from pre-projected cumulative scores,
eliminating the need to materialize the full (batch, T-1, K, C, C) edge tensor.

.. important::
    **When to use this module vs. triton_scan:**

    Use ``streaming`` (this module) when:
        - Edge tensor is too large to materialize (T > 10K, large K)
        - Edges follow the "Golden Rule" structure (content + transition)
        - Very long sequences (T = 100K - 400K+)

    Use ``triton_scan`` module when:
        - Edge tensor fits in GPU memory
        - Edge potentials are pre-computed (e.g., from a neural network)
        - Moderate sequence lengths (typically T < 10K)

    **Memory comparison:**

    +-----------------------+------------------+-------------------+
    | Scenario              | edge tensor size | cum_scores size   |
    +=======================+==================+===================+
    | T=1K, K=32, C=24      | 18 MB            | 96 KB             |
    +-----------------------+------------------+-------------------+
    | T=10K, K=100, C=24    | 5.5 GB           | 960 KB            |
    +-----------------------+------------------+-------------------+
    | T=400K, K=3K, C=24    | **2.76 TB**      | 38 MB             |
    +-----------------------+------------------+-------------------+

    For the T=400K case, the edge tensor cannot fit in memory. This module
    computes edges on-the-fly from O(T×C) cumulative scores instead.

API Comparison
--------------
The ``triton_scan`` module takes a **pre-computed edge tensor**::

    edge = model(x)  # shape: (batch, T-1, K, C, C) - must fit in GPU memory!
    partition = semi_crf_triton_forward(edge, lengths)

This module takes **cumulative scores** and computes edges on-the-fly::

    cum_scores = cumsum(projected, dim=1)  # shape: (batch, T+1, C) - much smaller!
    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

Memory Complexity
-----------------
- Pre-computed edge API (triton_scan): O(T × K × C²) - 2.76 TB for T=400K, K=3K, C=24
- Golden Rule API (this module): O(T × C + K × C + C²) - ~50 MB for same dimensions

The Golden Rule
---------------
Instead of pre-computing edges, we pre-project encoder features to label space
BEFORE the kernel, then compute edges on-the-fly inside:

    # Outside kernel (parallel, efficient)
    projected = h @ W_content                    # (batch, T, C)
    projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
    cum_scores = cumsum(projected.float(), dim=1)  # (batch, T+1, C) in float32

    # Inside kernel (just vector ops, no matmuls)
    content_score = cum_scores[:, t+k, :] - cum_scores[:, t, :]  # (batch, C)
    segment_score = content_score + duration_bias[k]
    edge_block = segment_score.unsqueeze(-1) + transition        # (batch, C, C)

The edge potential for segment [t, t+k) with label c_dest from c_src is::

    edge[t, k, c_dest, c_src] = (cum_scores[t+k, c_dest] - cum_scores[t, c_dest])
                              + duration_bias[k, c_dest]
                              + transition[c_src, c_dest]

This structure means you **never need to materialize the full edge tensor**.

Numerical Stability
-------------------
Two critical requirements for T=400K+ sequences:

1. **Float32 cumsum**: Cumsum must be float32 to avoid precision loss.
   Float16 loses all precision at T=400K magnitudes.

2. **Zero-centering**: Without centering, cumsum drifts to ~T magnitude.
   At T=400K, float32 epsilon at that magnitude is ~0.04 - any signal
   smaller than that is completely erased. Zero-centering keeps magnitude
   at √T (~632 for T=400K), preserving signals down to ~10⁻⁴.

Usage
-----
>>> import torch
>>> from torch_semimarkov.streaming import semi_crf_streaming_forward
>>>
>>> # Pre-project features (outside kernel)
>>> h = encoder(x)  # (batch, T, hidden_dim)
>>> projected = h @ W_content
>>> projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
>>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
>>> cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
>>>
>>> # Streaming forward (edges computed on-the-fly)
>>> partition = semi_crf_streaming_forward(
...     cum_scores, transition, duration_bias, lengths, K
... )

See Also
--------
:mod:`torch_semimarkov.triton_scan` : For sequences where edge tensor fits in memory
:class:`torch_semimarkov.SemiMarkov` : High-level API with marginals and sampling
"""

import math
import warnings
from typing import Optional, Tuple

import torch


# Triton is optional - kernel only available when installed and on GPU
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


NEG_INF = -1e9


# =============================================================================
# Triton Kernels for Streaming Forward (GPU only, optional)
# =============================================================================

if HAS_TRITON:

    def _next_power_of_2(n: int) -> int:
        """Return the smallest power of 2 >= n."""
        if n <= 0:
            return 1
        if n & (n - 1) == 0:
            return n
        p = 1
        while p < n:
            p *= 2
        return p

    @triton.jit
    def semi_crf_streaming_scan_kernel(
        # Inputs
        cum_scores_ptr,  # (batch, T+1, C) - cumulative projected scores
        transition_ptr,  # (C, C) - transition matrix
        duration_bias_ptr,  # (K, C) - duration-specific bias
        lengths_ptr,  # (batch,) - sequence lengths
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
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C)
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
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
        """
        Streaming Semi-CRF forward scan with Golden Rule edge computation.

        Computes edge potentials on-the-fly from cumulative scores:
            edge[c_dst, c_src] = (cum_scores[t+k, c_dst] - cum_scores[t, c_dst])
                               + duration_bias[k, c_dst]
                               + transition[c_src, c_dst]

        Uses a ring buffer for alpha values (O(KC) memory).
        Saves ring buffer checkpoints at regular intervals for backward pass.
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

        # Load transition matrix into registers: (C_PAD, C_PAD)
        # transition[c_src, c_dst] -> we need transition.T for edge computation
        # So we load transition_ptr[c_dst, c_src] effectively
        transition_block = tl.load(
            transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
            mask=c_mask_2d,
            other=0.0,
        )  # (C_PAD, C_PAD) - this is transition.T

        # Initialize ring buffer: alpha[0, :] = 0.0, rest = NEG_INF
        # Ring buffer layout: ring[k, c] where k = position % K
        for k_init in tl.static_range(0, K):
            val = tl.where(k_init == 0, 0.0, NEG_INF)
            init_vals = tl.where(c_mask, val, NEG_INF)
            tl.store(
                ring_base + k_init * stride_ring_k + c_idx * stride_ring_c,
                init_vals,
                mask=c_mask,
            )

        # Save initial ring buffer state as checkpoint 0
        for k_init in tl.static_range(0, K):
            val = tl.where(k_init == 0, 0.0, NEG_INF)
            init_vals = tl.where(c_mask, val, NEG_INF)
            tl.store(
                ring_ckpt_base + 0 * stride_ckpt_n + k_init * stride_ckpt_k + c_idx * stride_ckpt_c,
                init_vals,
                mask=c_mask,
            )

        # Track final alpha for each batch element
        final_alpha = tl.where(c_mask, 0.0, NEG_INF).to(tl.float32)

        # Handle length=1 sequences (no transitions)
        is_len_1 = seq_len == 1
        # For length 1, final_alpha is just 0.0 (initial state)

        # Main forward loop: t = 1, 2, ..., T
        for t in tl.range(1, T + 1):
            active = t < seq_len

            # Accumulate alpha[t] = logsumexp over (k, c_src)
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Loop over valid segment durations k = 1, 2, ..., min(K-1, t)
            for k in tl.range(1, K):
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

                # === Compute edge block on-the-fly (Golden Rule) ===

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

                # Edge block: edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst]
                # segment_score is (C_PAD,), expand to (C_PAD, 1) for c_dst
                # transition_block is already (C_PAD, C_PAD) as transition.T
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
                for k_save in tl.static_range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    # Only save if checkpoint index is valid
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_save * stride_ckpt_k + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            # Capture final alpha at sequence end
            is_final = t == seq_len - 1
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Handle length=1 case
        final_alpha = tl.where(is_len_1 & c_mask, 0.0, final_alpha)

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
        transition_ptr,
        duration_bias_ptr,
        lengths_ptr,
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
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
    ):
        """
        Streaming Semi-CRF forward scan with max semiring (Viterbi).
        Same structure as log kernel but uses max instead of logsumexp.
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

        transition_block = tl.load(
            transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
            mask=c_mask_2d,
            other=0.0,
        )

        # Initialize ring buffer
        for k_init in tl.static_range(0, K):
            val = tl.where(k_init == 0, 0.0, NEG_INF)
            init_vals = tl.where(c_mask, val, NEG_INF)
            tl.store(
                ring_base + k_init * stride_ring_k + c_idx * stride_ring_c,
                init_vals,
                mask=c_mask,
            )

        # Save initial checkpoint
        for k_init in tl.static_range(0, K):
            val = tl.where(k_init == 0, 0.0, NEG_INF)
            init_vals = tl.where(c_mask, val, NEG_INF)
            tl.store(
                ring_ckpt_base + 0 * stride_ckpt_n + k_init * stride_ckpt_k + c_idx * stride_ckpt_c,
                init_vals,
                mask=c_mask,
            )

        final_alpha = tl.where(c_mask, 0.0, NEG_INF).to(tl.float32)
        is_len_1 = seq_len == 1

        for t in tl.range(1, T + 1):
            active = t < seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            for k in tl.range(1, K):
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
                for k_save in tl.static_range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_save * stride_ckpt_k + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            is_final = t == seq_len - 1
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        final_alpha = tl.where(is_len_1 & c_mask, 0.0, final_alpha)

        # Max semiring: max over labels
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_backward_kernel(
        # Inputs (from forward)
        cum_scores_ptr,  # (batch, T+1, C)
        transition_ptr,  # (C, C)
        duration_bias_ptr,  # (K, C)
        lengths_ptr,  # (batch,)
        log_Z_ptr,  # (batch,) - partition function values
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints from forward
        grad_output_ptr,  # (batch,) - upstream gradient
        # Working memory
        alpha_buffer_ptr,  # (batch, SEGMENT_SIZE, C_PAD) - recomputed alpha
        beta_ring_ptr,  # (batch, K, C_PAD) - beta ring buffer
        # Outputs (gradients)
        grad_cum_scores_ptr,  # (batch, T+1, C)
        grad_transition_ptr,  # (C, C) - requires atomic add
        grad_duration_bias_ptr,  # (K, C) - requires atomic add
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        SEGMENT_SIZE: tl.constexpr,  # = CHECKPOINT_INTERVAL + K
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C)
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
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
    ):
        """
        Streaming Semi-CRF backward kernel with gradient computation.

        Computes gradients via the forward-backward algorithm:
        1. Recompute alpha from checkpoints (segment by segment)
        2. Compute beta backward while accumulating gradients

        Marginal probability: P(segment) = exp(alpha + edge + beta - log_Z)
        Gradient accumulation uses atomic operations for shared parameters.

        Gradient Scaling Semantics (IMPORTANT):
        ---------------------------------------
        There's a subtle difference in how gradients are scaled for per-batch vs shared parameters:

        - **Per-batch parameters** (cum_scores): Each batch element's gradient contribution
          is scaled by its corresponding grad_output[batch_idx]. This happens INSIDE the kernel.

        - **Shared parameters** (transition, duration_bias): These are accumulated across all
          batch elements WITHOUT per-element scaling. The scaling by grad_output.sum() happens
          AFTER the kernel in the launcher function.

        This matches PyTorch's backward semantics where:
            grad_transition = sum_{b,t,k}(marginal[b,t,k]) * grad_output.sum()

        NOT:
            grad_transition = sum_{b,t,k}(marginal[b,t,k] * grad_output[b])  # WRONG!

        When grad_output = [1, 1, ..., 1] (the common case), the difference is a factor of `batch`:
        - Correct: sum(marginals) * batch
        - Wrong: sum(marginals) * 1

        This was a subtle bug that caused a factor-of-2 error when batch=2.
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

        # Load transition matrix into registers
        transition_block = tl.load(
            transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
            mask=c_mask_2d,
            other=0.0,
        )  # (C_PAD, C_PAD) - this is transition.T

        # Initialize beta ring buffer at final positions
        final_pos = seq_len - 1
        final_ring_idx = final_pos % K
        for k_init in tl.static_range(0, K):
            is_final = k_init == final_ring_idx
            init_val = tl.where(is_final & c_mask, 0.0, NEG_INF)
            tl.store(
                beta_ring_base + k_init * stride_br_k + c_idx * stride_br_c,
                init_val,
                mask=c_mask,
            )

        # Process segments in reverse order
        for ckpt_idx_loop in range(NUM_CKPTS):
            ckpt_idx = NUM_CKPTS - 1 - ckpt_idx_loop
            seg_start = ckpt_idx * CHECKPOINT_INTERVAL
            seg_end = (ckpt_idx + 1) * CHECKPOINT_INTERVAL
            if seg_end > T:
                seg_end = T

            # Only process segments within sequence length
            if seg_start < seq_len - 1:
                # === Phase 1: Recompute alpha for this segment ===
                # Load ring buffer state from checkpoint
                # Then recompute forward through the segment

                # Initialize alpha from checkpoint (stores ring buffer state at seg_start)
                for k_slot in tl.static_range(0, K):
                    alpha_val = tl.load(
                        ring_ckpt_base + ckpt_idx * stride_ckpt_n +
                        k_slot * stride_ckpt_k + c_idx * stride_ckpt_c,
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
                for local_t in range(1, SEGMENT_SIZE):
                    t = seg_start + local_t
                    # Only process if within segment and sequence bounds
                    if t < seg_end and t < seq_len:
                        alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

                        # Loop over valid durations
                        for k in tl.static_range(1, K):
                            start_pos = t - k
                            # Only process valid start positions
                            if start_pos >= 0:
                                # Get alpha_prev - either from buffer or checkpoint
                                local_start = start_pos - seg_start
                                if local_start >= 0 and local_start < SEGMENT_SIZE:
                                    alpha_prev = tl.load(
                                        alpha_buf_base + local_start * stride_ab_t + c_idx * stride_ab_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )
                                else:
                                    # Position is before seg_start, get from current checkpoint
                                    # The checkpoint at ckpt_idx contains alpha[seg_start-K+1..seg_start]
                                    # at ring indices (seg_start-K+1) % K .. seg_start % K
                                    prev_ring_idx = start_pos % K
                                    alpha_prev = tl.load(
                                        ring_ckpt_base + ckpt_idx * stride_ckpt_n +
                                        prev_ring_idx * stride_ckpt_k + c_idx * stride_ckpt_c,
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
                                    tl.exp(alpha_t - max_alpha) + tl.exp(score_for_k - max_alpha) + 1e-10
                                )

                        # Store recomputed alpha
                        alpha_t = tl.where(c_mask, alpha_t, NEG_INF)
                        tl.store(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            alpha_t,
                            mask=c_mask,
                        )

                # === Phase 2: Compute beta backward and gradients ===
                for t_offset in range(CHECKPOINT_INTERVAL):
                    t = seg_end - 1 - t_offset
                    # Only process valid positions
                    if t >= seg_start and t < seq_len - 1 and t >= 0:
                        # Get alpha[t] from buffer
                        local_t = t - seg_start
                        alpha_t = tl.load(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )

                        # Compute beta[t] and gradients
                        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

                        for k in tl.static_range(1, K):
                            end_pos = t + k
                            # Only process valid end positions
                            if end_pos <= seq_len - 1 and end_pos <= T - 1:
                                # Get beta[end_pos] from ring buffer
                                end_ring_idx = end_pos % K
                                beta_next = tl.load(
                                    beta_ring_base + end_ring_idx * stride_br_k + c_idx * stride_br_c,
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
                                edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)

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
                                marginal_sum_src = tl.sum(marginal, axis=1)  # sum over c_src -> (C_PAD,)
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

                                # grad_transition: use 2D atomic add (unscaled marginal)
                                # marginal is (C_dst, C_src), grad_transition[c_src, c_dst] += marginal[c_dst, c_src]
                                marginal_T = tl.trans(marginal)  # (C_src, C_dst) = (C_PAD, C_PAD)
                                # Compute 2D offsets: grad_transition[row, col] at row * stride_tr_src + col * stride_tr_dst
                                # c_dst_idx is (C_PAD, 1) serving as row indices, c_src_idx is (1, C_PAD) as col indices
                                tr_offsets = c_dst_idx * stride_tr_src + c_src_idx * stride_tr_dst
                                tl.atomic_add(grad_transition_ptr + tr_offsets, marginal_T, mask=c_mask_2d)

                                # grad_duration_bias[k, c_dst] += sum over c_src (unscaled)
                                tl.atomic_add(
                                    grad_duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c,
                                    marginal_sum_src,
                                    mask=c_mask,
                                )

                                # === Update beta contribution ===
                                # beta[t, c_src] = logsumexp over (k, c_dst) of edge[c_dst, c_src] + beta[end, c_dst]
                                scores_for_beta = edge_block + beta_next[:, None]  # (C_dst, C_src)
                                scores_for_beta = tl.where(c_mask_2d, scores_for_beta, NEG_INF)

                                # Logsumexp over c_dst (axis 0)
                                max_beta_k = tl.max(scores_for_beta, axis=0)
                                beta_k = max_beta_k + tl.log(
                                    tl.sum(tl.exp(scores_for_beta - max_beta_k[None, :]), axis=0) + 1e-10
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Launch the Triton backward kernel.

        Args:
            cum_scores: (batch, T+1, C)
            transition: (C, C)
            duration_bias: (K, C)
            lengths: (batch,)
            log_Z: (batch,) partition values from forward
            ring_checkpoints: (batch, num_ckpts, K, C) saved states
            checkpoint_interval: interval used during forward
            grad_output: (batch,) upstream gradient

        Returns:
            grad_cum_scores: (batch, T+1, C)
            grad_transition: (C, C)
            grad_duration_bias: (K, C)
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

        # Ensure contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()
        log_Z = log_Z.contiguous()
        grad_output = grad_output.contiguous()

        # Pad checkpoints to C_PAD
        if ring_checkpoints.shape[-1] < C_PAD:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints
        else:
            ring_ckpts_padded = ring_checkpoints.contiguous()

        # Allocate working memory
        alpha_buffer = torch.full(
            (batch, segment_size, C_PAD), NEG_INF, device=device, dtype=dtype
        )
        beta_ring = torch.full(
            (batch, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )

        # Allocate gradient outputs
        grad_cum_scores = torch.zeros(batch, T_plus_1, C, device=device, dtype=dtype)
        grad_transition = torch.zeros(C, C, device=device, dtype=dtype)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=dtype)

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()
        stride_tr_src, stride_tr_dst = transition.stride()
        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_ckpts_padded.stride()
        stride_ab_b, stride_ab_t, stride_ab_c = alpha_buffer.stride()
        stride_br_b, stride_br_k, stride_br_c = beta_ring.stride()
        stride_gcs_b, stride_gcs_t, stride_gcs_c = grad_cum_scores.stride()

        # Launch kernel
        grid = (batch,)
        semi_crf_streaming_backward_kernel[grid](
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts_padded,
            grad_output,
            alpha_buffer,
            beta_ring,
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            batch,
            T,
            K,
            C,
            C_PAD,
            checkpoint_interval,
            num_checkpoints,
            segment_size,
            stride_cs_b,
            stride_cs_t,
            stride_cs_c,
            stride_tr_src,
            stride_tr_dst,
            stride_db_k,
            stride_db_c,
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
        )

        # Scale shared parameter gradients by grad_output.sum()
        #
        # BUG FIX: This is critical for correctness!
        #
        # PyTorch backward semantics for shared parameters:
        #   grad_transition = sum_{b,t,k}(marginal) * grad_output.sum()
        #
        # The kernel accumulates unscaled marginals via atomic_add across all batch elements.
        # We then scale by grad_output.sum() here (NOT per-element grad_output[b]).
        #
        # Without this fix, when batch=2 and grad_output=[1,1]:
        #   - Triton computed: sum(marginals) * 1 = sum(marginals)
        #   - PyTorch computed: sum(marginals) * 2 = 2 * sum(marginals)
        #   - Result: factor of 2 error (0.5 relative difference)
        #
        # See kernel docstring for full explanation of the scaling semantics.
        grad_output_sum = grad_output.sum()
        grad_transition = grad_transition * grad_output_sum
        grad_duration_bias = grad_duration_bias * grad_output_sum

        return grad_cum_scores, grad_transition, grad_duration_bias

    def launch_streaming_triton_kernel(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        checkpoint_interval: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Launch the streaming Triton kernel with proper buffer allocation.

        Args:
            cum_scores: (batch, T+1, C) cumulative projected scores
            transition: (C, C) transition matrix
            duration_bias: (K, C) duration-specific bias
            lengths: (batch,) sequence lengths
            K: max segment duration
            semiring: "log" or "max"
            checkpoint_interval: interval for saving ring buffer (default: sqrt(T*K))

        Returns:
            partition: (batch,) partition function values
            ring_checkpoints: (batch, num_ckpts, K, C) saved ring buffer states
            checkpoint_interval: actual interval used
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

        # Ensure inputs are contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()

        # Allocate outputs
        partition = torch.empty(batch, device=device, dtype=dtype)

        # Live ring buffer (will be L1/L2 cached for small K*C)
        ring_buffer = torch.full(
            (batch, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )

        # Checkpoint storage for backward pass
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()
        stride_tr_src, stride_tr_dst = transition.stride()
        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()

        # Launch kernel
        grid = (batch,)
        kernel = semi_crf_streaming_scan_kernel if semiring == "log" else semi_crf_streaming_scan_kernel_max
        kernel[grid](
            cum_scores,
            transition,
            duration_bias,
            lengths,
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
            stride_cs_b,
            stride_cs_t,
            stride_cs_c,
            stride_tr_src,
            stride_tr_dst,
            stride_db_k,
            stride_db_c,
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


def _compute_checkpoint_interval(T: int, K: int) -> int:
    r"""_compute_checkpoint_interval(T, K) -> int

    Compute optimal checkpoint interval to minimize total memory.

    The optimal interval :math:`S` minimizes total memory:

    .. math::
        \text{Memory} = \frac{T}{S} \times K \times C + S \times C + K \times C

    Taking :math:`\frac{d}{dS} = 0` gives :math:`S^* = \sqrt{T \times K}`.

    Args:
        T (int): Sequence length.
        K (int): Maximum segment duration.

    Returns:
        int: Optimal checkpoint interval (at least K).
    """
    optimal = int(math.sqrt(T * K))
    return max(K, optimal)


def compute_edge_block_golden_rule(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    t: int,
    k: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""compute_edge_block_golden_rule(cum_scores, transition, duration_bias, t, k, proj_start=None, proj_end=None) -> Tensor

    Compute edge block on-the-fly using the Golden Rule.

    This computes the edge potential for segments starting at position ``t``
    with duration ``k``, without materializing the full edge tensor:

    .. math::
        \text{edge}[c_{\text{dest}}, c_{\text{src}}] = \text{segment\_score}[c_{\text{dest}}]
        + \text{transition}[c_{\text{src}}, c_{\text{dest}}]

    where :math:`\text{segment\_score} = \text{content\_score} + \text{duration\_bias} + \text{boundaries}`.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape :math:`(\text{batch}, T+1, C)`.
            Must be float32 and zero-centered for numerical stability.
        transition (Tensor): Label transition scores of shape :math:`(C, C)`.
            ``transition[c_src, c_dest]`` is the score for :math:`c_{\text{src}} \to c_{\text{dest}}`.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        t (int): Segment start position.
        k (int): Segment duration.
        proj_start (Tensor, optional): Start boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``

    Returns:
        Tensor: Edge potentials of shape :math:`(\text{batch}, C, C)`.

    Examples::

        >>> cum_scores = torch.randn(2, 101, 4)  # batch=2, T=100, C=4
        >>> transition = torch.randn(4, 4)
        >>> duration_bias = torch.randn(8, 4)  # K=8
        >>> edge = compute_edge_block_golden_rule(cum_scores, transition, duration_bias, t=10, k=3)
        >>> edge.shape
        torch.Size([2, 4, 4])
    """
    # Content score via cumsum difference: (batch, C)
    content_score = cum_scores[:, t + k, :] - cum_scores[:, t, :]

    # Add duration bias
    segment_score = content_score + duration_bias[k]

    # Add boundary scores if provided
    if proj_start is not None:
        segment_score = segment_score + proj_start[:, t, :]
    if proj_end is not None:
        segment_score = segment_score + proj_end[:, t + k - 1, :]

    # Build edge block: segment_score[c_dest] + transition[c_src, c_dest]
    # segment_score: (batch, C) -> unsqueeze to (batch, C, 1) for c_dest
    # transition: (C_src, C_dest) -> transpose to (C_dest, C_src) -> (1, C_dest, C_src)
    # Result: (batch, C_dest, C_src)
    edge_block = segment_score.unsqueeze(-1) + transition.T.unsqueeze(0)

    return edge_block


def semi_crf_streaming_forward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    checkpoint_interval: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""semi_crf_streaming_forward_pytorch(cum_scores, transition, duration_bias, lengths, K, semiring='log', proj_start=None, proj_end=None, checkpoint_interval=None) -> Tuple[Tensor, Tensor, int]

    Forward pass with Golden Rule edge computation.

    Computes the log partition function using a ring buffer with :math:`O(KC)` memory.
    Edge potentials are computed on-the-fly from cumulative scores.

    .. note::
        This is an internal function. Use :func:`semi_crf_streaming_forward` for the
        public API with automatic differentiation support.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape :math:`(\text{batch}, T+1, C)`.
            Must be float32. Should be zero-centered before cumsum.
        transition (Tensor): Label transition scores of shape :math:`(C, C)`.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` (logsumexp) or ``"max"`` (Viterbi).
            Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        checkpoint_interval (int, optional): Interval for saving ring buffer state.
            Default: ``None`` (uses :math:`\sqrt{T \times K}`)

    Returns:
        Tuple[Tensor, Tensor, int]: A tuple containing:

        - **partition** (Tensor): Log partition function of shape :math:`(\text{batch},)`
        - **ring_checkpoints** (Tensor): Saved ring buffer states for backward
        - **checkpoint_interval** (int): Actual interval used
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Warn if cum_scores appears non-zero-centered (only during eager execution)
    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        endpoint_magnitude = cum_scores[:, -1, :].abs().mean()
        if endpoint_magnitude > 1000:
            warnings.warn(
                f"cum_scores endpoint magnitude {endpoint_magnitude.item():.0f} suggests "
                "non-zero-centered input. This may cause precision loss at T>100K. "
                "Zero-center before cumsum: projected = projected - projected.mean(dim=1, keepdim=True)"
            )

    # Determine checkpoint interval
    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(T, K)
    else:
        checkpoint_interval = max(checkpoint_interval, K)

    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

    # Allocate ring checkpoint storage: (batch, num_checkpoints, K, C)
    ring_checkpoints = torch.full(
        (batch, num_checkpoints, K, C), NEG_INF, device=device, dtype=dtype
    )

    # Ring buffer for alpha values: (batch, K, C)
    alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
    alpha_ring[:, 0, :] = 0.0  # Initial: all labels equally likely

    # Store initial ring buffer state (checkpoint 0)
    ring_checkpoints[:, 0, :, :] = alpha_ring

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)
    final_positions = lengths - 1

    # Handle sequences of length 1
    len_1_mask = lengths == 1
    if len_1_mask.any():
        final_alpha[len_1_mask] = 0.0

    # Main forward loop
    for t in range(1, T + 1):
        active_mask = t < lengths

        # Number of valid durations at this position
        k_eff = min(K - 1, t)

        scores_all = []
        for k in range(1, k_eff + 1):
            start = t - k

            # Get alpha[start] from ring buffer
            ring_idx = start % K
            alpha_prev = alpha_ring[:, ring_idx, :]  # (batch, C_src)

            # Compute edge block on-the-fly (Golden Rule)
            edge_block = compute_edge_block_golden_rule(
                cum_scores, transition, duration_bias,
                start, k, proj_start, proj_end
            )  # (batch, C_dest, C_src)

            # scores[c_dest, c_src] = alpha_prev[c_src] + edge[c_dest, c_src]
            scores = alpha_prev.unsqueeze(-2) + edge_block  # (batch, C_dest, C_src)
            scores_all.append(scores)

        # Stack: (batch, k_eff, C_dest, C_src)
        scores_stacked = torch.stack(scores_all, dim=1)

        if semiring == "log":
            # logsumexp over (k, c_src) -> (batch, C_dest)
            scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
            alpha_t = torch.logsumexp(scores_over_src, dim=1)
        else:  # max
            scores_over_src = torch.max(scores_stacked, dim=-1)[0]
            alpha_t = torch.max(scores_over_src, dim=1)[0]

        # Update ring buffer (only for active sequences)
        ring_idx_t = t % K
        alpha_ring[:, ring_idx_t, :] = torch.where(
            active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
        )

        # Save ring buffer state at checkpoint positions
        if t % checkpoint_interval == 0:
            ckpt_idx = t // checkpoint_interval
            if ckpt_idx < num_checkpoints:
                for k_slot in range(K):
                    ring_checkpoints[:, ckpt_idx, k_slot, :] = torch.where(
                        active_mask.view(batch, 1),
                        alpha_ring[:, k_slot, :],
                        ring_checkpoints[:, ckpt_idx, k_slot, :],
                    )

        # Track final alpha for sequences ending at this position
        is_final = t == final_positions
        if is_final.any():
            final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)

    # Compute partition function
    if semiring == "log":
        partition = torch.logsumexp(final_alpha, dim=-1)
    else:
        partition = torch.max(final_alpha, dim=-1)[0]

    return partition, ring_checkpoints, checkpoint_interval


def semi_crf_streaming_backward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    log_Z: torch.Tensor,
    ring_checkpoints: torch.Tensor,
    checkpoint_interval: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""semi_crf_streaming_backward_pytorch(cum_scores, transition, duration_bias, lengths, K, log_Z, ring_checkpoints, checkpoint_interval, semiring='log', proj_start=None, proj_end=None) -> Tuple[Tensor, ...]

    Backward pass computing gradients via marginals.

    Uses the forward-backward algorithm with checkpointing. Recomputes alpha
    within segments from saved ring buffer checkpoints, then computes beta
    backward while accumulating gradients.

    The marginal probability is computed as:

    .. math::
        P(\text{segment}_{t,k,c}) = \frac{\alpha[t, c_{\text{src}}] \cdot
        \text{edge}[c_{\text{dest}}, c_{\text{src}}] \cdot \beta[t+k, c_{\text{dest}}]}{Z}

    .. note::
        This is an internal function. Use :func:`semi_crf_streaming_forward` which
        automatically handles gradients via :class:`SemiCRFStreaming`.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Label transition scores of shape :math:`(C, C)`.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        log_Z (Tensor): Log partition values of shape :math:`(\text{batch},)`.
        ring_checkpoints (Tensor): Saved ring buffer states of shape
            :math:`(\text{batch}, \text{num\_checkpoints}, K, C)`.
        checkpoint_interval (int): Interval between checkpoints.
        semiring (str, optional): ``"log"`` or ``"max"``. Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``

    Returns:
        Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]: A tuple containing:

        - **grad_cum_scores** (Tensor): Gradient of shape :math:`(\text{batch}, T+1, C)`
        - **grad_transition** (Tensor): Gradient of shape :math:`(C, C)`
        - **grad_duration_bias** (Tensor): Gradient of shape :math:`(K, C)`
        - **grad_proj_start** (Tensor or None): Gradient if ``proj_start`` was provided
        - **grad_proj_end** (Tensor or None): Gradient if ``proj_end`` was provided
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    effective_interval = max(checkpoint_interval, K)

    # Initialize gradient accumulators
    grad_cum_scores = torch.zeros_like(cum_scores)
    grad_transition = torch.zeros_like(transition)
    grad_duration_bias = torch.zeros_like(duration_bias)
    grad_proj_start = torch.zeros_like(proj_start) if proj_start is not None else None
    grad_proj_end = torch.zeros_like(proj_end) if proj_end is not None else None

    # Segment buffer for alpha values
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # Beta ring buffer
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    final_positions = lengths - 1
    for b in range(batch):
        final_ring_idx = final_positions[b].item() % K
        beta_ring[b, final_ring_idx, :] = 0.0

    num_checkpoints = ring_checkpoints.shape[1]

    # Process segments in reverse order
    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = min((ckpt_idx + 1) * checkpoint_interval, T)

        # Clear segment buffer
        alpha_segment.fill_(NEG_INF)

        # === Phase 1: Recompute alpha from checkpoint's ring buffer state ===
        alpha_ring = ring_checkpoints[:, ckpt_idx, :, :].clone()

        # Store alpha[seg_start] at local position 0
        alpha_segment[:, 0, :] = alpha_ring[:, seg_start % K, :]

        # Recompute alpha for positions seg_start+1 to seg_end-1
        for t in range(seg_start + 1, seg_end):
            active_mask = t < lengths

            k_eff = min(K - 1, t)
            scores_all = []

            for k in range(1, k_eff + 1):
                start = t - k
                ring_idx = start % K
                alpha_prev = alpha_ring[:, ring_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_golden_rule(
                    cum_scores, transition, duration_bias,
                    start, k, proj_start, proj_end
                )

                scores = alpha_prev.unsqueeze(-2) + edge_block
                scores_all.append(scores)

            if scores_all:
                scores_stacked = torch.stack(scores_all, dim=1)
                if semiring == "log":
                    scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
                    alpha_t = torch.logsumexp(scores_over_src, dim=1)
                else:
                    scores_over_src = torch.max(scores_stacked, dim=-1)[0]
                    alpha_t = torch.max(scores_over_src, dim=1)[0]

                alpha_ring[:, t % K, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_ring[:, t % K, :]
                )

                # Store in segment buffer
                local_t = t - seg_start
                alpha_segment[:, local_t, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_segment[:, local_t, :]
                )

        # === Phase 2: Compute beta backward and gradients ===
        for t in range(seg_end - 1, seg_start - 1, -1):
            if t >= T - 1:
                continue

            local_t = t - seg_start
            alpha_t = alpha_segment[:, local_t, :]

            active_mask = t < (lengths - 1)
            if not active_mask.any():
                continue

            max_k = min(K - 1, T - 1 - t)
            new_beta_scores = []

            for k in range(1, max_k + 1):
                end_pos = t + k
                valid_mask = (end_pos <= lengths - 1) & active_mask

                if not valid_mask.any():
                    continue

                ring_k_idx = end_pos % K
                beta_next = beta_ring[:, ring_k_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_golden_rule(
                    cum_scores, transition, duration_bias,
                    t, k, proj_start, proj_end
                )

                # === Gradient computation ===
                # log_marginal[c_dest, c_src] = alpha[t, c_src] + edge[c_dest, c_src] + beta[end, c_dest] - log_Z
                log_marginal = (
                    alpha_t.unsqueeze(-2)  # (batch, 1, C_src)
                    + edge_block  # (batch, C_dest, C_src)
                    + beta_next.unsqueeze(-1)  # (batch, C_dest, 1)
                    - log_Z.view(batch, 1, 1)
                )
                marginal = torch.exp(log_marginal)
                marginal = torch.where(
                    valid_mask.view(batch, 1, 1), marginal, torch.zeros_like(marginal)
                )

                # === Accumulate gradients ===

                # grad_cum_scores: contribution from segments
                # Segment score uses cum_scores[end] - cum_scores[start]
                # grad w.r.t. cum_scores[end] is positive (coefficient +1)
                # grad w.r.t. cum_scores[start] is negative (coefficient -1)
                marginal_sum_labels = marginal.sum(dim=(-1, -2), keepdim=True)  # (batch, 1, 1)
                marginal_sum_dest = marginal.sum(dim=-1)  # (batch, C_dest)

                # grad_cum_scores[end_pos, c_dest] += sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, end_pos, :] += marginal_sum_dest
                # grad_cum_scores[t, c_dest] -= sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, t, :] -= marginal_sum_dest

                # grad_transition: sum over batch, positions
                # transition[c_src, c_dest] appears in edge[c_dest, c_src] (after transpose)
                # So grad_transition[c_src, c_dest] += marginal[c_dest, c_src]
                # marginal is (batch, C_dest, C_src), sum over batch
                grad_transition += marginal.sum(dim=0).T  # (C_src, C_dest)

                # grad_duration_bias[k, c_dest] += sum over batch, c_src of marginal[c_dest, c_src]
                grad_duration_bias[k, :] += marginal.sum(dim=(0, -1))  # (C_dest,)

                # grad_proj_start, grad_proj_end
                if grad_proj_start is not None:
                    grad_proj_start[:, t, :] += marginal_sum_dest
                if grad_proj_end is not None:
                    grad_proj_end[:, end_pos - 1, :] += marginal_sum_dest

                # === Beta contribution ===
                scores_for_beta = edge_block + beta_next.unsqueeze(-1)
                scores_for_beta = torch.where(
                    valid_mask.view(batch, 1, 1),
                    scores_for_beta,
                    torch.full_like(scores_for_beta, NEG_INF),
                )
                new_beta_scores.append(scores_for_beta)

            # Update beta ring buffer
            if new_beta_scores:
                stacked = torch.stack(new_beta_scores, dim=1)
                if semiring == "log":
                    over_dest = torch.logsumexp(stacked, dim=-2)
                    new_beta = torch.logsumexp(over_dest, dim=1)
                else:
                    over_dest = torch.max(stacked, dim=-2)[0]
                    new_beta = torch.max(over_dest, dim=1)[0]

                ring_t_idx = t % K
                beta_ring[:, ring_t_idx, :] = torch.where(
                    active_mask.view(batch, 1), new_beta, beta_ring[:, ring_t_idx, :]
                )

    return grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end


class SemiCRFStreaming(torch.autograd.Function):
    r"""Autograd function for streaming Semi-CRF with Golden Rule edge computation.

    This wraps the forward and backward passes to enable automatic differentiation.
    Memory usage is :math:`O(KC)` for the ring buffer, independent of sequence length :math:`T`.

    .. note::
        This class is used internally by :func:`semi_crf_streaming_forward`.
        Users should call that function directly rather than using this class.

    See Also:
        :func:`semi_crf_streaming_forward`: Main entry point for streaming Semi-CRF
    """

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        proj_start: Optional[torch.Tensor] = None,
        proj_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Detach inputs for forward computation
        partition, ring_checkpoints, checkpoint_interval = semi_crf_streaming_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
            K,
            semiring,
            proj_start.detach() if proj_start is not None else None,
            proj_end.detach() if proj_end is not None else None,
        )

        # Save for backward
        ctx.save_for_backward(
            cum_scores, transition, duration_bias, lengths,
            ring_checkpoints, partition,
            proj_start, proj_end,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = checkpoint_interval

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (cum_scores, transition, duration_bias, lengths,
         ring_checkpoints, partition,
         proj_start, proj_end) = ctx.saved_tensors

        grads = semi_crf_streaming_backward_pytorch(
            cum_scores, transition, duration_bias, lengths,
            ctx.K, partition, ring_checkpoints, ctx.checkpoint_interval,
            ctx.semiring, proj_start, proj_end,
        )

        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end = grads

        # Scale by upstream gradient
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale
        grad_transition = grad_transition * grad_output.sum()
        grad_duration_bias = grad_duration_bias * grad_output.sum()
        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start * scale
        if grad_proj_end is not None:
            grad_proj_end = grad_proj_end * scale

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
            grad_proj_start,
            grad_proj_end,
        )


class SemiCRFStreamingTriton(torch.autograd.Function):
    r"""Autograd function using Triton forward and backward kernels.

    This class uses custom Triton kernels for both forward and backward passes,
    providing maximum performance for GPU training. The backward pass uses
    checkpointing to recompute alpha values, trading compute for memory.

    .. note::
        This class is used internally when ``use_triton=True`` and gradients
        are needed.

    See Also:
        :class:`SemiCRFStreaming`: Pure PyTorch autograd function
    """

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
    ) -> torch.Tensor:
        # Use Triton kernel for forward
        partition, ring_checkpoints, checkpoint_interval = launch_streaming_triton_kernel(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
            K,
            semiring,
        )

        # Save for backward
        ctx.save_for_backward(
            cum_scores, transition, duration_bias, lengths,
            ring_checkpoints, partition,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = checkpoint_interval

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (cum_scores, transition, duration_bias, lengths,
         ring_checkpoints, partition) = ctx.saved_tensors

        # Use Triton backward kernel for gradient computation
        # The kernel already scales by grad_output internally
        grad_cum_scores, grad_transition, grad_duration_bias = launch_streaming_triton_backward(
            cum_scores, transition, duration_bias, lengths,
            partition, ring_checkpoints, ctx.checkpoint_interval,
            grad_output,
        )

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
        )


# =============================================================================
# Main Entry Point
# =============================================================================


def semi_crf_streaming_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    use_compile: bool = False,  # Deprecated, kept for API compatibility
) -> torch.Tensor:
    r"""semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K, semiring='log', proj_start=None, proj_end=None, use_triton=True) -> Tensor

    Compute Semi-CRF partition function with Golden Rule streaming.

    This is the main entry point for the streaming API. Edge potentials are
    computed on-the-fly from cumulative scores, eliminating the need for the
    full :math:`(\text{batch}, T-1, K, C, C)` edge tensor.

    Memory: :math:`O(KC)` ring buffer, independent of sequence length :math:`T`.
    Compute: :math:`O(T \times K \times C^2)` same as standard Semi-CRF.

    Uses custom Triton kernels for optimal performance on GPU:

    - **Inference** (no gradients): Uses custom Triton forward kernel
    - **Training** (with gradients): Uses custom Triton forward and backward kernels

    .. warning::
        ``cum_scores`` **MUST** be float32 for numerical stability at :math:`T > 100K`.
        Zero-centering before cumsum is critical to prevent precision loss.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape :math:`(\text{batch}, T+1, C)`.
            Must be float32 and zero-centered before cumsum for numerical stability.
        transition (Tensor): Label transition scores of shape :math:`(C, C)`.
            ``transition[c_src, c_dest]`` is the score for :math:`c_{\text{src}} \to c_{\text{dest}}`.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
            Required to compensate for sum-pooling length bias.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` (logsumexp for partition) or ``"max"`` (Viterbi).
            Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape :math:`(\text{batch}, T, C)`.
            Default: ``None``
        use_triton (bool, optional): If ``True``, use Triton kernels when available.
            Default: ``True``

    Returns:
        Tensor: Log partition function (or max score) of shape :math:`(\text{batch},)`.

    Examples::

        >>> import torch
        >>> from torch_semimarkov.streaming import semi_crf_streaming_forward
        >>>
        >>> # Encoder output
        >>> batch, T, hidden_dim, C, K = 2, 100, 64, 4, 8
        >>> h = torch.randn(batch, T, hidden_dim)  # encoder output
        >>> W_content = torch.randn(hidden_dim, C)
        >>>
        >>> # Pre-project to label space (Golden Rule: outside kernel)
        >>> projected = h @ W_content  # (batch, T, C)
        >>>
        >>> # CRITICAL: Zero-center before cumsum
        >>> projected = projected - projected.mean(dim=1, keepdim=True)
        >>>
        >>> # Cumsum in float32
        >>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
        >>> cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
        >>>
        >>> # Model parameters
        >>> transition = torch.randn(C, C) * 0.1
        >>> duration_bias = torch.randn(K, C) * 0.1
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Streaming forward (uses Triton on GPU)
        >>> partition = semi_crf_streaming_forward(
        ...     cum_scores, transition, duration_bias, lengths, K
        ... )
        >>> partition.shape
        torch.Size([2])

    See Also:
        :class:`~torch_semimarkov.SemiMarkov`: Pre-computed edge tensor API
        :func:`compute_edge_block_golden_rule`: On-the-fly edge computation helper
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    # Check if Triton is available and applicable
    can_use_triton = (
        HAS_TRITON
        and use_triton
        and cum_scores.is_cuda
        and proj_start is None  # Triton path doesn't support boundary projections yet
        and proj_end is None
    )

    # Determine if gradients are needed
    needs_grad = (
        cum_scores.requires_grad
        or transition.requires_grad
        or duration_bias.requires_grad
    )

    if needs_grad:
        # Training path
        if can_use_triton:
            # Use Triton forward + Triton backward kernels
            return SemiCRFStreamingTriton.apply(
                cum_scores, transition, duration_bias, lengths, K, semiring
            )
        else:
            # Pure PyTorch path (supports boundary projections)
            return SemiCRFStreaming.apply(
                cum_scores, transition, duration_bias, lengths, K, semiring,
                proj_start, proj_end,
            )
    else:
        # Inference path (no gradients)
        if can_use_triton:
            # Use fast custom Triton kernel
            partition, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K, semiring
            )
            return partition
        else:
            # CPU fallback or boundary projections
            partition, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K, semiring,
                proj_start, proj_end,
            )
            return partition
