r"""Triton backward kernel for streaming Semi-CRF.

Uses forward-backward algorithm with checkpointing:
1. Recompute alpha values from saved ring buffer checkpoints
2. Compute beta backward while accumulating gradients

Gradients computed via marginal probabilities:

.. math::
    P(\text{segment}) = \frac{\exp(\alpha + \text{edge} + \beta)}{\exp(\log Z)}

**Configuration:**

- ``accum_dtype=torch.float64`` (default): Stable at batch >= 128, ~1.5x memory, ~2x slower
- ``accum_dtype=torch.float32``: Lower memory, baseline speed, stable only at batch <= 64

**Recommendations:**

- Memory-constrained: Use ``accum_dtype=torch.float32`` with batch size <= 64
- Throughput-focused: Use ``accum_dtype=torch.float64`` with batch size 256+
- The sweet spot balances GPU memory utilization against samples/second
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
    def semi_crf_streaming_backward_fused_kernel(
        # Inputs (from forward)
        cum_scores_ptr,  # (batch, T+1, C)
        transition_ptr,  # (C, C) - ONLY static transitions for fused kernel
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
        grad_tr_workspace_ptr,  # (batch, C_PAD, C_PAD) - per-batch accumulator
        grad_db_workspace_ptr,  # (batch, K, C_PAD) - per-batch accumulator
        grad_proj_start_ptr,  # (batch, T, C) - gradient for proj_start
        grad_proj_end_ptr,  # (batch, T, C) - gradient for proj_end
        boundary_marginals_ptr,  # (batch, T) - output for boundary marginals
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
        RETURN_BOUNDARY_MARGINALS: tl.constexpr,  # whether to accumulate boundary marginals
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C) - static only
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
        # Strides for proj_start/proj_end (batch, T, C)
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
        # Strides for grad_tr_workspace (batch, C_PAD, C_PAD)
        stride_gtw_b,
        stride_gtw_src,
        stride_gtw_dst,
        # Strides for grad_db_workspace (batch, K, C_PAD)
        stride_gdbw_b,
        stride_gdbw_k,
        stride_gdbw_c,
        # Strides for boundary_marginals (batch, T)
        stride_bm_b,
        stride_bm_t,
        # Tile size (must be at end for constexpr)
        TILE_C: tl.constexpr,
    ):
        """Fused backward kernel with local gradient accumulation.

        This kernel eliminates atomic operations for grad_transition and
        grad_duration_bias by accumulating in registers and writing once at
        the end. This reduces ~4.6M atomics to ~48K, achieving 10-50x speedup.

        Key differences from semi_crf_streaming_backward_kernel:
        - ONLY supports static transitions (C, C), not duration-dependent (K, C, C)
        - Uses local register accumulators for transition and duration_bias grads
        - Writes accumulated gradients once at kernel end

        Atomics are still used for:
        - grad_cum_scores (different t positions, unavoidable)
        - boundary gradients (different t positions)
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

        # Clamp indices for safe pointer calculation
        c_idx_safe = tl.minimum(c_idx, C - 1)
        c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
        c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b
        alpha_buf_base = alpha_buffer_ptr + batch_idx * stride_ab_b
        beta_ring_base = beta_ring_ptr + batch_idx * stride_br_b
        grad_cs_base = grad_cum_scores_ptr + batch_idx * stride_gcs_b
        grad_tr_ws_base = grad_tr_workspace_ptr + batch_idx * stride_gtw_b
        grad_db_ws_base = grad_db_workspace_ptr + batch_idx * stride_gdbw_b

        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b
            grad_ps_base = grad_proj_start_ptr + batch_idx * stride_ps_b
            grad_pe_base = grad_proj_end_ptr + batch_idx * stride_ps_b

        # Load transition matrix into registers (static transitions only)
        transition_block = tl.load(
            transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
            mask=c_mask_2d,
            other=0.0,
        )  # (C_PAD, C_PAD)

        # LOCAL GRADIENT ACCUMULATORS - key optimization
        # These accumulate across all (t, k) iterations, written once at end
        # Total: 64×64×8 + 16×64×8 = 32KB + 8KB = 40KB registers
        local_grad_tr = tl.zeros([C_PAD, C_PAD], dtype=tl.float64)
        local_grad_db = tl.zeros([K, C_PAD], dtype=tl.float64)

        # Initialize beta ring buffer at final positions
        final_pos = seq_len
        final_ring_idx = final_pos % K
        for k_init in tl.range(0, K):
            is_final = k_init == final_ring_idx
            init_val = tl.where(is_final & c_mask, 0.0, NEG_INF)
            tl.store(
                beta_ring_base + k_init * stride_br_k + c_idx * stride_br_c,
                init_val,
                mask=c_mask,
            )

        # Process segments in reverse order
        for ckpt_idx_loop in tl.range(0, NUM_CKPTS):
            ckpt_idx = NUM_CKPTS - 1 - ckpt_idx_loop
            seg_start = ckpt_idx * CHECKPOINT_INTERVAL
            seg_end = (ckpt_idx + 1) * CHECKPOINT_INTERVAL
            if seg_end > T:
                seg_end = T

            if seg_start < seq_len:
                # Recompute alpha from checkpoint
                for k_slot in tl.range(0, K):
                    alpha_val = tl.load(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_slot * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    if k_slot == seg_start % K:
                        tl.store(
                            alpha_buf_base + 0 * stride_ab_t + c_idx * stride_ab_c,
                            alpha_val,
                            mask=c_mask,
                        )

                # Recompute alpha values from seg_start+1 to seg_end
                for local_t in tl.range(1, SEGMENT_SIZE):
                    t = seg_start + local_t
                    if t < seg_end and t < seq_len:
                        alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

                        for k in tl.range(1, tl.maximum(K, 2)):
                            start_pos = t - k
                            if start_pos >= 0:
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
                                    prev_ring_idx = start_pos % K
                                    alpha_prev = tl.load(
                                        ring_ckpt_base
                                        + ckpt_idx * stride_ckpt_n
                                        + prev_ring_idx * stride_ckpt_k
                                        + c_idx * stride_ckpt_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )

                                cum_end = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start = tl.load(
                                    cum_scores_base
                                    + start_pos * stride_cs_t
                                    + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score = cum_end - cum_start

                                dur_idx = tl.minimum(k, K - 1)
                                dur_bias = tl.load(
                                    duration_bias_ptr
                                    + dur_idx * stride_db_k
                                    + c_idx_safe * stride_db_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                segment_score = content_score + dur_bias

                                if HAS_BOUNDARIES:
                                    start_score = tl.load(
                                        proj_start_base
                                        + start_pos * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = t - 1
                                    end_score = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score = segment_score + start_score + end_score

                                edge_block = segment_score[:, None] + transition_block
                                scores = alpha_prev[None, :] + edge_block
                                scores = tl.where(c_mask_2d, scores, NEG_INF)

                                max_scores = tl.max(scores, axis=1)
                                is_all_neginf = max_scores < (NEG_INF + 1.0)
                                max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                                log_sum_exp = tl.log(
                                    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1)
                                    + 1e-10
                                )
                                score_for_k = tl.where(
                                    is_all_neginf, NEG_INF, max_scores + log_sum_exp
                                )
                                score_for_k = tl.where(c_mask, score_for_k, NEG_INF)

                                max_alpha = tl.maximum(alpha_t, score_for_k)
                                is_both_neginf = (alpha_t < (NEG_INF + 1.0)) & (
                                    score_for_k < (NEG_INF + 1.0)
                                )
                                max_alpha_safe = tl.where(is_both_neginf, 0.0, max_alpha)
                                log_sum_exp_acc = tl.log(
                                    tl.exp(alpha_t - max_alpha_safe)
                                    + tl.exp(score_for_k - max_alpha_safe)
                                    + 1e-10
                                )
                                alpha_t = tl.where(
                                    is_both_neginf, NEG_INF, max_alpha + log_sum_exp_acc
                                )

                        alpha_t = tl.where(c_mask, alpha_t, NEG_INF)
                        tl.store(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            alpha_t,
                            mask=c_mask,
                        )

                # Compute beta backward and gradients
                for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
                    t = seg_end - 1 - t_offset
                    if t >= seg_start and t < seq_len and t >= 0:
                        local_t = t - seg_start
                        alpha_t = tl.load(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )

                        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

                        for k in tl.range(1, tl.maximum(K, 2)):
                            end_pos = t + k
                            if end_pos <= seq_len and end_pos <= T:
                                end_ring_idx = end_pos % K
                                dur_idx = tl.minimum(k, K - 1)

                                marginal_sum_all_k = 0.0
                                m_beta_k = tl.full([C_PAD], NEG_INF, dtype=tl.float64)
                                l_beta_k = tl.zeros([C_PAD], dtype=tl.float64)

                                alpha_t_clamped = tl.minimum(tl.maximum(alpha_t, -1e6), 1e6)

                                # Load full beta_next for this k
                                beta_next = tl.load(
                                    beta_ring_base
                                    + end_ring_idx * stride_br_k
                                    + c_idx_safe * stride_br_c,
                                    mask=c_mask,
                                    other=NEG_INF,
                                )
                                beta_next_clamped = tl.minimum(tl.maximum(beta_next, -1e6), 1e6)

                                # Compute segment scores (full, not tiled)
                                cum_end_full = tl.load(
                                    cum_scores_base
                                    + end_pos * stride_cs_t
                                    + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start_full = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score_full = cum_end_full - cum_start_full

                                dur_bias_full = tl.load(
                                    duration_bias_ptr
                                    + dur_idx * stride_db_k
                                    + c_idx_safe * stride_db_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                segment_score_full = content_score_full + dur_bias_full

                                if HAS_BOUNDARIES:
                                    start_score_full = tl.load(
                                        proj_start_base
                                        + t * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = end_pos - 1
                                    end_score_full = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score_full = (
                                        segment_score_full + start_score_full + end_score_full
                                    )

                                # Compute full (C_PAD, C_PAD) edge and marginal
                                # edge[c_dst, c_src] = segment_score[c_dst] + transition[c_dst, c_src]
                                edge_full = segment_score_full[:, None] + transition_block
                                edge_full_clamped = tl.minimum(tl.maximum(edge_full, -1e6), 1e6)

                                # log_marginal[c_dst, c_src] = alpha[c_src] + edge[c_dst, c_src] + beta[c_dst] - log_Z
                                log_marginal_full = (
                                    alpha_t_clamped[None, :]  # (1, C_PAD) for c_src
                                    + edge_full_clamped  # (C_PAD, C_PAD)
                                    + beta_next_clamped[:, None]  # (C_PAD, 1) for c_dst
                                    - log_Z
                                )
                                log_marginal_full = tl.minimum(
                                    tl.maximum(log_marginal_full, -700.0), 700.0
                                )
                                marginal_full = tl.exp(log_marginal_full)
                                marginal_full = tl.where(c_mask_2d, marginal_full, 0.0)

                                if RETURN_BOUNDARY_MARGINALS:
                                    marginal_sum_all_k += tl.sum(marginal_full)

                                # grad_cum_scores: sum over c_src -> (C_PAD,)
                                marginal_sum_src = tl.sum(marginal_full, axis=1)
                                marginal_sum_src = tl.where(c_mask, marginal_sum_src, 0.0)
                                marginal_sum_src_scaled = marginal_sum_src * grad_out

                                # grad_cum_scores[end_pos]: +marginal (must use atomic)
                                tl.atomic_add(
                                    grad_cs_base + end_pos * stride_gcs_t + c_idx * stride_gcs_c,
                                    marginal_sum_src_scaled,
                                    mask=c_mask,
                                )
                                # grad_cum_scores[t]: -marginal (must use atomic)
                                tl.atomic_add(
                                    grad_cs_base + t * stride_gcs_t + c_idx * stride_gcs_c,
                                    -marginal_sum_src_scaled,
                                    mask=c_mask,
                                )

                                # LOCAL ACCUMULATION - key optimization
                                # grad_transition: marginal_T = (C_PAD, C_PAD)
                                marginal_T = tl.trans(marginal_full)
                                local_grad_tr += marginal_T

                                # grad_duration_bias: sum over c_src (unscaled)
                                # local_grad_db[k, c] += marginal_sum_src[c] for k == dur_idx
                                # Use tl.where with row mask to update only the dur_idx row
                                row_mask = tl.arange(0, K)[:, None] == dur_idx  # (K, 1)
                                local_grad_db += tl.where(row_mask, marginal_sum_src[None, :], 0.0)

                                # Boundary gradients (still use atomics)
                                if HAS_BOUNDARIES:
                                    tl.atomic_add(
                                        grad_ps_base + t * stride_ps_t + c_idx * stride_ps_c,
                                        marginal_sum_src_scaled,
                                        mask=c_mask,
                                    )
                                    tl.atomic_add(
                                        grad_pe_base
                                        + (end_pos - 1) * stride_ps_t
                                        + c_idx * stride_ps_c,
                                        marginal_sum_src_scaled,
                                        mask=c_mask,
                                    )

                                # Online logsumexp for beta_k
                                scores_for_beta = edge_full + beta_next[:, None]
                                scores_for_beta = tl.where(c_mask_2d, scores_for_beta, NEG_INF)

                                max_scores_beta = tl.max(scores_for_beta, axis=0)
                                is_neginf = max_scores_beta < (NEG_INF + 1.0)
                                max_safe = tl.where(is_neginf, 0.0, max_scores_beta)

                                sum_exp_beta = tl.sum(
                                    tl.exp(scores_for_beta - max_safe[None, :]),
                                    axis=0,
                                )
                                sum_exp_beta = tl.where(is_neginf, 0.0, sum_exp_beta)

                                m_new = tl.maximum(m_beta_k, max_scores_beta)
                                is_m_neginf = m_beta_k < (NEG_INF + 1.0)
                                m_new_safe = tl.where(is_m_neginf & is_neginf, 0.0, m_new)

                                l_beta_k = tl.where(
                                    is_m_neginf,
                                    sum_exp_beta * tl.exp(max_scores_beta - m_new_safe),
                                    l_beta_k * tl.exp(m_beta_k - m_new_safe)
                                    + sum_exp_beta * tl.exp(max_scores_beta - m_new_safe),
                                )
                                m_beta_k = m_new

                                is_beta_k_neginf = m_beta_k < (NEG_INF + 1.0)
                                beta_k = tl.where(
                                    is_beta_k_neginf,
                                    NEG_INF,
                                    m_beta_k + tl.log(l_beta_k + 1e-10),
                                )
                                beta_k = tl.where(c_mask, beta_k, NEG_INF)

                                if RETURN_BOUNDARY_MARGINALS:
                                    tl.atomic_add(
                                        boundary_marginals_ptr
                                        + batch_idx * stride_bm_b
                                        + t * stride_bm_t,
                                        marginal_sum_all_k,
                                    )

                                # Accumulate beta_k into new_beta via logsumexp over k
                                max_new = tl.maximum(new_beta, beta_k)
                                is_both_neginf_beta = (new_beta < (NEG_INF + 1.0)) & (
                                    beta_k < (NEG_INF + 1.0)
                                )
                                max_new_safe = tl.where(is_both_neginf_beta, 0.0, max_new)
                                log_sum_exp_new = tl.log(
                                    tl.exp(new_beta - max_new_safe)
                                    + tl.exp(beta_k - max_new_safe)
                                    + 1e-10
                                )
                                new_beta = tl.where(
                                    is_both_neginf_beta, NEG_INF, max_new + log_sum_exp_new
                                )

                        # Store beta[t] to ring buffer
                        t_ring_idx = t % K
                        tl.store(
                            beta_ring_base + t_ring_idx * stride_br_k + c_idx * stride_br_c,
                            new_beta,
                            mask=c_mask,
                        )

        # WRITE LOCAL ACCUMULATORS TO WORKSPACE (single write, no atomics)
        # This is the key optimization - we've accumulated across all (t, k) iterations
        # and now write once instead of ~4.6M atomic operations

        # grad_transition: (C_PAD, C_PAD) -> workspace
        tr_offsets = c_idx[:, None] * stride_gtw_src + c_idx[None, :] * stride_gtw_dst
        tl.store(
            grad_tr_ws_base + tr_offsets,
            local_grad_tr,
            mask=c_mask_2d,
        )

        # grad_duration_bias: (K, C_PAD) -> workspace
        for k_write in tl.range(0, K):
            tl.store(
                grad_db_ws_base + k_write * stride_gdbw_k + c_idx * stride_gdbw_c,
                local_grad_db[k_write, :],
                mask=c_mask,
            )

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
        boundary_marginals_ptr,  # (batch, T) - output for boundary marginals (if RETURN_BOUNDARY_MARGINALS)
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
        RETURN_BOUNDARY_MARGINALS: tl.constexpr,  # whether to accumulate boundary marginals
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
        # Strides for boundary_marginals (batch, T) - only used if RETURN_BOUNDARY_MARGINALS
        stride_bm_b,
        stride_bm_t,
        # Autotuned parameter (must be at end for @triton.autotune)
        TILE_C: tl.constexpr,
    ):
        """Streaming Semi-CRF backward kernel with loop tiling.

        For each checkpoint segment (reverse order):
        1. Recompute alpha from checkpoint, then forward through segment
        2. Compute beta backward while accumulating gradients via marginals

        Numerical Stability:
        - Float64 accumulation for atomic_add precision
        - NEG_INF guards for all-invalid logsumexp inputs
        - Log-marginal clamping to [-700, 700] before exp()
        - Alpha/beta/edge clamping to [-1e6, 1e6]

        Loop Tiling:
        Process c_dst in TILE_C tiles to reduce register pressure from
        O(C_PAD²) to O(TILE_C × C_PAD). Uses online logsumexp for beta.

        Memory: O((S+K)×C) alpha buffer + O(KC) beta ring.
        One program per batch element.
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

        # Clamp indices to ensure address calculations stay within bounds
        # even for masked-out threads (indices C to C_PAD-1).
        # This prevents OOB pointer calculation which is undefined behavior.
        c_idx_safe = tl.minimum(c_idx, C - 1)
        c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
        c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

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
        # Use clamped indices to avoid OOB pointer calculation for masked-out threads
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
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
                # Recompute alpha from checkpoint
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
                        alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

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
                                # Use clamped indices to avoid OOB pointer calculation
                                cum_end = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start = tl.load(
                                    cum_scores_base
                                    + start_pos * stride_cs_t
                                    + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score = cum_end - cum_start

                                # Use min(k, K-1) to handle K=1 case: k=1 maps to index 0
                                dur_idx = tl.minimum(k, K - 1)
                                dur_bias = tl.load(
                                    duration_bias_ptr
                                    + dur_idx * stride_db_k
                                    + c_idx_safe * stride_db_c,
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
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = t - 1
                                    end_score = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score = segment_score + start_score + end_score

                                # Load k-indexed transition for duration-dependent case
                                # Use clamped indices to avoid OOB pointer calculation
                                if HAS_DURATION_TRANSITIONS:
                                    transition_block = tl.load(
                                        transition_ptr
                                        + k * stride_tr_k
                                        + c_dst_idx_safe * stride_tr_dst
                                        + c_src_idx_safe * stride_tr_src,
                                        mask=c_mask_2d,
                                        other=0.0,
                                    )

                                edge_block = segment_score[:, None] + transition_block

                                scores = alpha_prev[None, :] + edge_block
                                scores = tl.where(c_mask_2d, scores, NEG_INF)

                                # Logsumexp over c_src
                                # Guard against all-NEG_INF case to prevent undefined arithmetic
                                max_scores = tl.max(scores, axis=1)
                                is_all_neginf = max_scores < (NEG_INF + 1.0)
                                max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                                log_sum_exp = tl.log(
                                    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1)
                                    + 1e-10
                                )
                                score_for_k = tl.where(
                                    is_all_neginf, NEG_INF, max_scores + log_sum_exp
                                )
                                score_for_k = tl.where(c_mask, score_for_k, NEG_INF)

                                # Accumulate via logsumexp
                                # Guard against both inputs being NEG_INF
                                max_alpha = tl.maximum(alpha_t, score_for_k)
                                is_both_neginf = (alpha_t < (NEG_INF + 1.0)) & (
                                    score_for_k < (NEG_INF + 1.0)
                                )
                                max_alpha_safe = tl.where(is_both_neginf, 0.0, max_alpha)
                                log_sum_exp_acc = tl.log(
                                    tl.exp(alpha_t - max_alpha_safe)
                                    + tl.exp(score_for_k - max_alpha_safe)
                                    + 1e-10
                                )
                                alpha_t = tl.where(
                                    is_both_neginf, NEG_INF, max_alpha + log_sum_exp_acc
                                )

                        # Store recomputed alpha
                        alpha_t = tl.where(c_mask, alpha_t, NEG_INF)
                        tl.store(
                            alpha_buf_base + local_t * stride_ab_t + c_idx * stride_ab_c,
                            alpha_t,
                            mask=c_mask,
                        )

                # Compute beta backward and gradients
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
                        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

                        # Tiled computation over c_dst dimension
                        for k in tl.range(1, tl.maximum(K, 2)):
                            end_pos = t + k
                            # Only process valid end positions
                            if end_pos <= seq_len and end_pos <= T:
                                end_ring_idx = end_pos % K
                                dur_idx = tl.minimum(k, K - 1)

                                marginal_sum_all_k = 0.0
                                m_beta_k = tl.full([C_PAD], NEG_INF, dtype=tl.float64)
                                l_beta_k = tl.zeros([C_PAD], dtype=tl.float64)

                                # Clamp alpha_t once per k (reused across tiles)
                                alpha_t_clamped = tl.minimum(tl.maximum(alpha_t, -1e6), 1e6)

                                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                                    c_dst_tile = tl.arange(0, TILE_C)
                                    c_dst_idx_tile = c_dst_tile_start + c_dst_tile
                                    c_dst_mask_tile = c_dst_idx_tile < C
                                    c_dst_idx_tile_safe = tl.minimum(c_dst_idx_tile, C - 1)
                                    tile_mask_2d = c_dst_mask_tile[:, None] & c_mask[None, :]

                                    # Load beta_next tile (TILE_C,)
                                    beta_tile = tl.load(
                                        beta_ring_base
                                        + end_ring_idx * stride_br_k
                                        + c_dst_idx_tile_safe * stride_br_c,
                                        mask=c_dst_mask_tile,
                                        other=NEG_INF,
                                    )

                                    # Compute segment_score tile (TILE_C,)
                                    cum_end_tile = tl.load(
                                        cum_scores_base
                                        + end_pos * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    cum_start_tile = tl.load(
                                        cum_scores_base
                                        + t * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    content_score_tile = cum_end_tile - cum_start_tile

                                    dur_bias_tile = tl.load(
                                        duration_bias_ptr
                                        + dur_idx * stride_db_k
                                        + c_dst_idx_tile_safe * stride_db_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    segment_score_tile = content_score_tile + dur_bias_tile

                                    # Add boundary scores if provided
                                    if HAS_BOUNDARIES:
                                        start_score_tile = tl.load(
                                            proj_start_base
                                            + t * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        end_pos_boundary = end_pos - 1
                                        end_score_tile = tl.load(
                                            proj_end_base
                                            + end_pos_boundary * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        segment_score_tile = (
                                            segment_score_tile + start_score_tile + end_score_tile
                                        )

                                    # Load transition tile (TILE_C, C_PAD)
                                    # Rows = c_dst tile, Columns = all c_src
                                    if HAS_DURATION_TRANSITIONS:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + k * stride_tr_k
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )
                                    else:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )

                                    # edge_tile: (TILE_C, C_PAD)
                                    edge_tile = segment_score_tile[:, None] + transition_tile

                                    beta_tile_clamped = tl.minimum(tl.maximum(beta_tile, -1e6), 1e6)
                                    edge_tile_clamped = tl.minimum(tl.maximum(edge_tile, -1e6), 1e6)

                                    log_marginal_tile = (
                                        alpha_t_clamped[None, :]  # (1, C_PAD) for c_src
                                        + edge_tile_clamped  # (TILE_C, C_PAD)
                                        + beta_tile_clamped[:, None]  # (TILE_C, 1) for c_dst
                                        - log_Z
                                    )
                                    log_marginal_tile = tl.minimum(
                                        tl.maximum(log_marginal_tile, -700.0), 700.0
                                    )
                                    marginal_tile = tl.exp(log_marginal_tile)
                                    marginal_tile = tl.where(tile_mask_2d, marginal_tile, 0.0)

                                    if RETURN_BOUNDARY_MARGINALS:
                                        marginal_sum_all_k += tl.sum(marginal_tile)

                                    # grad_cum_scores: sum over c_src -> (TILE_C,)
                                    marginal_sum_src_tile = tl.sum(marginal_tile, axis=1)
                                    marginal_sum_src_tile = tl.where(
                                        c_dst_mask_tile, marginal_sum_src_tile, 0.0
                                    )
                                    marginal_sum_src_tile_scaled = marginal_sum_src_tile * grad_out

                                    # grad_cum_scores[end_pos]: +marginal (varies by k, must use atomic)
                                    tl.atomic_add(
                                        grad_cs_base
                                        + end_pos * stride_gcs_t
                                        + c_dst_idx_tile * stride_gcs_c,
                                        marginal_sum_src_tile_scaled,
                                        mask=c_dst_mask_tile,
                                    )
                                    # grad_cum_scores[t]: -marginal
                                    tl.atomic_add(
                                        grad_cs_base
                                        + t * stride_gcs_t
                                        + c_dst_idx_tile * stride_gcs_c,
                                        -marginal_sum_src_tile_scaled,
                                        mask=c_dst_mask_tile,
                                    )

                                    # grad_transition: marginal_T_tile = (C_PAD, TILE_C)
                                    marginal_T_tile = tl.trans(marginal_tile)
                                    if HAS_DURATION_TRANSITIONS:
                                        tr_offsets_tile = (
                                            k * stride_gtw_k
                                            + c_idx[:, None] * stride_gtw_src
                                            + c_dst_idx_tile[None, :] * stride_gtw_dst
                                        )
                                    else:
                                        tr_offsets_tile = (
                                            c_idx[:, None] * stride_gtw_src
                                            + c_dst_idx_tile[None, :] * stride_gtw_dst
                                        )
                                    tile_mask_T = c_mask[:, None] & c_dst_mask_tile[None, :]
                                    tl.atomic_add(
                                        grad_tr_ws_base + tr_offsets_tile,
                                        marginal_T_tile,
                                        mask=tile_mask_T,
                                    )

                                    # grad_duration_bias: (unscaled)
                                    tl.atomic_add(
                                        grad_db_ws_base
                                        + k * stride_gdbw_k
                                        + c_dst_idx_tile * stride_gdbw_c,
                                        marginal_sum_src_tile,
                                        mask=c_dst_mask_tile,
                                    )

                                    # grad_proj_start[t] and grad_proj_end[end_pos-1]
                                    if HAS_BOUNDARIES:
                                        # proj_start[t]: same position for all k, use atomic
                                        tl.atomic_add(
                                            grad_ps_base
                                            + t * stride_ps_t
                                            + c_dst_idx_tile * stride_ps_c,
                                            marginal_sum_src_tile_scaled,
                                            mask=c_dst_mask_tile,
                                        )
                                        # proj_end[end_pos-1]: varies by k, use atomic
                                        tl.atomic_add(
                                            grad_pe_base
                                            + (end_pos - 1) * stride_ps_t
                                            + c_dst_idx_tile * stride_ps_c,
                                            marginal_sum_src_tile_scaled,
                                            mask=c_dst_mask_tile,
                                        )

                                    # Online logsumexp for beta_k
                                    scores_for_beta_tile = edge_tile + beta_tile[:, None]
                                    scores_for_beta_tile = tl.where(
                                        tile_mask_2d, scores_for_beta_tile, NEG_INF
                                    )

                                    max_tile = tl.max(scores_for_beta_tile, axis=0)
                                    is_tile_neginf = max_tile < (NEG_INF + 1.0)
                                    max_tile_safe = tl.where(is_tile_neginf, 0.0, max_tile)

                                    sum_exp_tile = tl.sum(
                                        tl.exp(scores_for_beta_tile - max_tile_safe[None, :]),
                                        axis=0,
                                    )
                                    sum_exp_tile = tl.where(is_tile_neginf, 0.0, sum_exp_tile)

                                    m_new = tl.maximum(m_beta_k, max_tile)
                                    is_m_neginf = m_beta_k < (NEG_INF + 1.0)
                                    m_new_safe = tl.where(is_m_neginf & is_tile_neginf, 0.0, m_new)

                                    l_beta_k = tl.where(
                                        is_m_neginf,
                                        sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                        l_beta_k * tl.exp(m_beta_k - m_new_safe)
                                        + sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                    )
                                    m_beta_k = m_new

                                is_beta_k_neginf = m_beta_k < (NEG_INF + 1.0)
                                beta_k = tl.where(
                                    is_beta_k_neginf,
                                    NEG_INF,
                                    m_beta_k + tl.log(l_beta_k + 1e-10),
                                )
                                beta_k = tl.where(c_mask, beta_k, NEG_INF)

                                # Accumulate boundary marginals for this k
                                if RETURN_BOUNDARY_MARGINALS:
                                    tl.atomic_add(
                                        boundary_marginals_ptr
                                        + batch_idx * stride_bm_b
                                        + t * stride_bm_t,
                                        marginal_sum_all_k,
                                    )

                                # Accumulate beta_k into new_beta via logsumexp over k
                                max_new = tl.maximum(new_beta, beta_k)
                                is_both_neginf_beta = (new_beta < (NEG_INF + 1.0)) & (
                                    beta_k < (NEG_INF + 1.0)
                                )
                                max_new_safe = tl.where(is_both_neginf_beta, 0.0, max_new)
                                log_sum_exp_new = tl.log(
                                    tl.exp(new_beta - max_new_safe)
                                    + tl.exp(beta_k - max_new_safe)
                                    + 1e-10
                                )
                                new_beta = tl.where(
                                    is_both_neginf_beta, NEG_INF, max_new + log_sum_exp_new
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
        return_boundary_marginals: bool = False,
        accum_dtype: torch.dtype = torch.float64,
        num_warps: int = 4,
        validate_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Launch Triton backward kernel with buffer allocation.

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
            return_boundary_marginals (bool, optional): If ``True``, also compute
                and return boundary marginals. Default: ``False``
            accum_dtype (torch.dtype, optional): Dtype for gradient accumulation.
                Use ``torch.float64`` (default) for numerical stability at batch >= 128.
                Use ``torch.float32`` for lower memory at batch <= 64.
            num_warps (int, optional): Warps per block (2-8). Default: ``4``
            validate_cache (bool, optional): Validate Triton cache. Default: ``True``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of:
                - **grad_cum_scores** (Tensor): Shape :math:`(\text{batch}, T+1, C)`.
                - **grad_transition** (Tensor): Shape :math:`(C, C)` or :math:`(K, C, C)`.
                - **grad_duration_bias** (Tensor): Shape :math:`(K, C)`.
                - **grad_proj_start** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
                  if boundaries provided.
                - **grad_proj_end** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
                  if boundaries provided.
                - **boundary_marginals** (Tensor or None): Shape :math:`(\text{batch}, T)`
                  if ``return_boundary_marginals=True``.
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Validate cache if requested (include TILE_C for backward kernel)
        if validate_cache:
            config = TritonConfig(num_warps=num_warps, tile_c=16)
            validate_triton_cache(config)
            update_cache_sentinel(config)
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
            # Allocate gradient outputs for boundaries with C_PAD and float64 precision
            # (we slice back to C and convert to original dtype before returning)
            grad_proj_start = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float64)
            grad_proj_end = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float64)
        else:
            # Create dummy tensors for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0
            grad_proj_start = None
            grad_proj_end = None
            # Kernel uses proj_start/proj_end directly when HAS_BOUNDARY_PROJ=False

        # Pad checkpoints to C_PAD and convert to float64 for kernel consistency.
        # The kernel uses float64 for alpha computation - checkpoints must match.
        if ring_checkpoints.shape[-1] < C_PAD:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=torch.float64
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints.to(torch.float64)
        else:
            ring_ckpts_padded = ring_checkpoints.to(torch.float64).contiguous()

        # Allocate working memory in float64 to match kernel computation precision.
        # The kernel computes alpha/beta in float64 (tl.float64) - using float32 buffers
        # causes precision loss on store/load cycles, accumulating errors over T×K ops.
        alpha_buffer = torch.full(
            (batch, segment_size, C_PAD), NEG_INF, device=device, dtype=torch.float64
        )
        beta_ring = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=torch.float64)

        # Gradient accumulation dtype (float64 for stability at large batch, float32 for lower memory)
        grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=accum_dtype)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=accum_dtype)

        # Per-batch workspace buffers (C_PAD for OOB safety)
        if has_duration_transitions:
            # Duration-dependent: (batch, K, C_PAD, C_PAD)
            grad_tr_workspace = torch.zeros(
                batch, K, C_PAD, C_PAD, device=device, dtype=accum_dtype
            )
        else:
            # Static: (batch, C_PAD, C_PAD)
            grad_tr_workspace = torch.zeros(batch, C_PAD, C_PAD, device=device, dtype=accum_dtype)
        grad_db_workspace = torch.zeros(batch, K, C_PAD, device=device, dtype=accum_dtype)

        # Allocate boundary marginals output if requested
        if return_boundary_marginals:
            boundary_marginals = torch.zeros(batch, T, device=device, dtype=dtype)
            stride_bm_b, stride_bm_t = boundary_marginals.stride()
        else:
            boundary_marginals = grad_cum_scores[:, :T, 0]  # Dummy (won't be written)
            stride_bm_b, stride_bm_t = 0, 0

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
                boundary_marginals,
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
                return_boundary_marginals,  # RETURN_BOUNDARY_MARGINALS constexpr
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
                stride_bm_b,
                stride_bm_t,
                TILE_C=16,
                num_warps=num_warps,
            )

        # Reduce per-batch gradients: grad_θ = Σ_b[grad_output[b] × workspace[b]]
        grad_output_f64 = grad_output.to(torch.float64)
        if has_duration_transitions:
            grad_tr_workspace = grad_tr_workspace[:, :, :C, :C]
            grad_transition = torch.einsum("bkij, b -> kij", grad_tr_workspace, grad_output_f64)
        else:
            grad_tr_workspace = grad_tr_workspace[:, :C, :C]
            grad_transition = torch.einsum("bij, b -> ij", grad_tr_workspace, grad_output_f64)

        grad_db_workspace = grad_db_workspace[:, :, :C]
        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace, grad_output_f64)

        # Slice padded gradients back to actual class count C
        # All gradient workspaces use float64 → convert to original dtype
        # IMPORTANT: Clamp gradients to a reasonable range before conversion.
        # Float64 gradients can be extremely large; even values that fit in float32
        # (< 3.4e38) can corrupt model parameters during optimizer.step().
        # Use 1e10 as a conservative maximum - still very large but won't cause NaN.
        _GRAD_MAX = 1e10
        grad_cum_scores = grad_cum_scores[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
        grad_transition = grad_transition.clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
        grad_duration_bias = grad_duration_bias.clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
            grad_proj_end = grad_proj_end[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            grad_proj_start,
            grad_proj_end,
            boundary_marginals if return_boundary_marginals else None,
        )

    def launch_streaming_triton_backward_fused(
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
        return_boundary_marginals: bool = False,
        accum_dtype: torch.dtype = torch.float64,
        num_warps: int = 4,
        validate_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Launch fused Triton backward kernel with local gradient accumulation.

        This is an optimized version of :func:`launch_streaming_triton_backward`
        that eliminates atomic operations for transition and duration_bias gradients
        by accumulating in registers and writing once at the end.

        **Limitations:**
        - Only supports static transitions (C, C), NOT duration-dependent (K, C, C)

        **Performance:**
        - Reduces ~4.6M atomic operations to ~48K (for typical configs)
        - Expected 10-50x speedup on backward pass

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)`.
                **Duration-dependent (K, C, C) transitions are NOT supported.**
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
            return_boundary_marginals (bool, optional): If ``True``, also compute
                and return boundary marginals. Default: ``False``
            accum_dtype (torch.dtype, optional): Dtype for gradient accumulation.
                Use ``torch.float64`` (default) for numerical stability at batch >= 128.
                Use ``torch.float32`` for lower memory at batch <= 64.
            num_warps (int, optional): Warps per block (2-8). Default: ``4``
            validate_cache (bool, optional): Validate Triton cache. Default: ``True``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Same as
            :func:`launch_streaming_triton_backward`.

        Raises:
            ValueError: If transition has 3 dimensions (duration-dependent).
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Check that transition is static (C, C), not duration-dependent (K, C, C)
        if transition.ndim == 3:
            raise ValueError(
                "Fused backward kernel only supports static transitions (C, C). "
                f"Got transition with shape {transition.shape}. "
                "Use launch_streaming_triton_backward() for duration-dependent transitions."
            )

        if validate_cache:
            config = TritonConfig(num_warps=num_warps, tile_c=16)
            validate_triton_cache(config)
            update_cache_sentinel(config)

        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        K = duration_bias.shape[0]
        device = cum_scores.device
        dtype = cum_scores.dtype

        num_checkpoints = ring_checkpoints.shape[1]
        C_PAD = _next_power_of_2(C)
        segment_size = checkpoint_interval + K

        has_boundaries = proj_start is not None and proj_end is not None

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
            grad_proj_start = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float64)
            grad_proj_end = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float64)
        else:
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0
            grad_proj_start = None
            grad_proj_end = None

        # Pad checkpoints to C_PAD and convert to float64
        if ring_checkpoints.shape[-1] < C_PAD:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=torch.float64
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints.to(torch.float64)
        else:
            ring_ckpts_padded = ring_checkpoints.to(torch.float64).contiguous()

        # Allocate working memory
        alpha_buffer = torch.full(
            (batch, segment_size, C_PAD), NEG_INF, device=device, dtype=torch.float64
        )
        beta_ring = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=torch.float64)

        # Gradient accumulation buffers
        grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=accum_dtype)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=accum_dtype)

        # Per-batch workspace buffers (static transitions only)
        grad_tr_workspace = torch.zeros(batch, C_PAD, C_PAD, device=device, dtype=accum_dtype)
        grad_db_workspace = torch.zeros(batch, K, C_PAD, device=device, dtype=accum_dtype)

        # Boundary marginals
        if return_boundary_marginals:
            boundary_marginals = torch.zeros(batch, T, device=device, dtype=dtype)
            stride_bm_b, stride_bm_t = boundary_marginals.stride()
        else:
            boundary_marginals = grad_cum_scores[:, :T, 0]
            stride_bm_b, stride_bm_t = 0, 0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()
        stride_tr_src, stride_tr_dst = transition.stride()
        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_ckpts_padded.stride()
        stride_ab_b, stride_ab_t, stride_ab_c = alpha_buffer.stride()
        stride_br_b, stride_br_k, stride_br_c = beta_ring.stride()
        stride_gcs_b, stride_gcs_t, stride_gcs_c = grad_cum_scores.stride()
        stride_gtw_b, stride_gtw_src, stride_gtw_dst = grad_tr_workspace.stride()
        stride_gdbw_b, stride_gdbw_k, stride_gdbw_c = grad_db_workspace.stride()

        # Kernel pointers for boundaries
        grad_ps_for_kernel = grad_proj_start if has_boundaries else grad_cum_scores
        grad_pe_for_kernel = grad_proj_end if has_boundaries else grad_cum_scores

        # Launch fused kernel
        grid = (batch,)
        with torch.cuda.device(device):
            semi_crf_streaming_backward_fused_kernel[grid](
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
                boundary_marginals,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                segment_size,
                has_boundaries,
                return_boundary_marginals,
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
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
                stride_gtw_src,
                stride_gtw_dst,
                stride_gdbw_b,
                stride_gdbw_k,
                stride_gdbw_c,
                stride_bm_b,
                stride_bm_t,
                TILE_C=16,
                num_warps=num_warps,
            )

        # Reduce per-batch gradients
        grad_output_f64 = grad_output.to(torch.float64)
        grad_tr_workspace = grad_tr_workspace[:, :C, :C]
        grad_transition = torch.einsum("bij, b -> ij", grad_tr_workspace, grad_output_f64)

        grad_db_workspace = grad_db_workspace[:, :, :C]
        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace, grad_output_f64)

        # Slice and clamp gradients
        _GRAD_MAX = 1e10
        grad_cum_scores = grad_cum_scores[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
        grad_transition = grad_transition.clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
        grad_duration_bias = grad_duration_bias.clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)
            grad_proj_end = grad_proj_end[:, :, :C].clamp(-_GRAD_MAX, _GRAD_MAX).to(dtype)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            grad_proj_start,
            grad_proj_end,
            boundary_marginals if return_boundary_marginals else None,
        )

    def launch_streaming_triton_marginals(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        log_Z: torch.Tensor,
        ring_checkpoints: torch.Tensor,
        checkpoint_interval: int,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Compute boundary marginals via backward kernel.

        Runs backward with ``return_boundary_marginals=True``, discarding gradients.

        Args:
            cum_scores: Shape :math:`(\text{batch}, T+1, C)`.
            transition: Shape :math:`(C, C)` or :math:`(K, C, C)`.
            duration_bias: Shape :math:`(K, C)`.
            lengths: Shape :math:`(\text{batch},)`.
            log_Z: Partition values, shape :math:`(\text{batch},)`.
            ring_checkpoints: Shape :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
            checkpoint_interval: Interval from forward pass.
            proj_start: Optional boundary scores :math:`(\text{batch}, T, C)`.
            proj_end: Optional boundary scores :math:`(\text{batch}, T, C)`.

        Returns:
            Boundary marginals of shape :math:`(\text{batch}, T)`.
        """
        batch = cum_scores.shape[0]
        # Use ones for grad_output since we only want marginals, not scaled gradients
        grad_output = torch.ones(batch, device=cum_scores.device, dtype=cum_scores.dtype)

        _, _, _, _, _, boundary_marginals = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_checkpoints,
            checkpoint_interval,
            grad_output,
            proj_start,
            proj_end,
            return_boundary_marginals=True,
        )

        return boundary_marginals
