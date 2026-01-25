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
    # NOTE: @triton.autotune removed - corrupts gradient buffers during benchmarking.
    # See docs/DEBUGGING_NAN.md "Autotuning Limitations" for details.
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
        r"""Streaming Semi-CRF backward kernel with loop tiling and online logsumexp.

        This kernel computes gradients for the Semi-CRF partition function using
        the forward-backward algorithm with memory-efficient streaming checkpoints.

        Algorithm
        ---------
        For each checkpoint segment (processed in reverse order):

        **Phase 1 - Alpha Recomputation:**
            Load ring buffer checkpoint for segment start, then recompute
            alpha values forward through the segment using the recurrence:

            .. math::
                \alpha[t, c] = \text{logsumexp}_k \left(
                    \alpha[t-k, :] + \text{edge}[t-k \to t, :, c]
                \right)

        **Phase 2 - Beta Backward with Gradient Accumulation:**
            Process positions in reverse (t = seg_end-1 down to seg_start).
            For each position, compute marginal probabilities and accumulate
            gradients while updating beta via:

            .. math::
                \beta[t, c] = \text{logsumexp}_k \left(
                    \text{edge}[t \to t+k, c, :] + \beta[t+k, :]
                \right)

        Numerical Stability
        -------------------
        - **Float64 accumulation**: Gradient tensors use float64 to prevent
          non-determinism from atomic_add floating-point non-associativity.
          Error scales as O(sqrt(T×K×C)) per operation; float64 reduces this
          from ~1e-3 (float32) to ~1e-10 (negligible).

        - **NEG_INF guards**: When all logsumexp inputs are NEG_INF (-1e9),
          the subtraction `scores - max` yields 0 instead of staying at NEG_INF.
          Guards detect this case (max < NEG_INF + 1) and return NEG_INF directly.

        - **Log-marginal clamping**: Before exp(), log-marginals are clamped to
          [-700, 700] to prevent float64 overflow (exp(710) ≈ inf).

        - **Input clamping**: Alpha, beta, and edge values clamped to [-1e6, 1e6]
          before marginal computation to prevent extreme intermediate values.

        Loop Tiling (Register Pressure Reduction)
        -----------------------------------------
        The marginal computation requires a (C_PAD × C_PAD) matrix, which at
        C_PAD=64 demands ~384 registers/thread. With num_warps=4+, this exceeds
        available registers and causes spilling to slow local memory.

        **Solution**: Process C_PAD in tiles of TILE_C (typically 16 or 32):
            - Load (TILE_C × C_PAD) tile of marginal matrix
            - Accumulate gradients from tile
            - Use online logsumexp for beta update across tiles (Flash Attention pattern)

        This reduces peak register demand to ~120/thread, enabling num_warps=4-8.

        Memory Access Patterns
        ----------------------
        - **Transition matrix**: Loaded once per (t, k) pair, broadcast over c_src
        - **Cumulative scores**: Coalesced reads via c_idx stride
        - **Gradients**: Float64 workspace with atomic_add per (t, k, tile)

        Gradient Scaling Semantics
        --------------------------
        - **Per-batch parameters** (cum_scores): Scaled by grad_output[batch_idx]
          inside kernel
        - **Shared parameters** (transition, duration_bias): Accumulated unscaled,
          then reduced via einsum after kernel: ``grad = einsum("bij, b -> ij", ws, grad_out)``

        Performance Notes
        -----------------
        - **Single-config autotune only**: Multi-config autotuning is NOT safe for
          backward kernels with atomic operations. Triton's pre_hook/reset_to_zero
          only runs during benchmarking, NOT after selecting the best config
          (see Triton issue #7181). This causes the final run to accumulate on
          garbage data from the last benchmarked config, producing errors of ~10^23.

        - **Do NOT add more configs**: If you're tempted to add TILE_C=32 or
          different num_warps configs, DON'T. The first kernel call will produce
          wrong results due to buffer corruption during autotuning.

        - **L40/L40S optimal**: Fixed at num_warps=4, TILE_C=16

        - **One program per batch element**: grid = (batch_size,)

        See Also
        --------
        - Flash Attention: Online softmax pattern for memory-efficient attention
        - Mamba SSM: Loop tiling for register-pressure reduction in state space models
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
                        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

                        # === TILED BACKWARD COMPUTATION ===
                        # Process c_dst dimension in tiles of TILE_C to reduce register pressure.
                        # This enables higher num_warps (4-8) without register spilling.
                        #
                        # Pattern: Instead of computing full (C_PAD, C_PAD) marginal matrix,
                        # compute (TILE_C, C_PAD) tiles and accumulate gradients per-tile.
                        # For beta update, use online logsumexp across tiles (Flash Attention pattern).

                        # tl.maximum ensures K=1 processes at least one duration
                        for k in tl.range(1, tl.maximum(K, 2)):
                            end_pos = t + k
                            # Only process valid end positions
                            if end_pos <= seq_len and end_pos <= T:
                                end_ring_idx = end_pos % K
                                dur_idx = tl.minimum(k, K - 1)

                                # === Accumulators for this k iteration ===
                                # Boundary marginals accumulator (scalar)
                                marginal_sum_all_k = 0.0

                                # Online logsumexp accumulators for beta_k (indexed by c_src)
                                # Accumulates logsumexp over c_dst tiles
                                # IMPORTANT: Use float64 to match new_beta precision and avoid
                                # accumulation errors with higher num_warps
                                m_beta_k = tl.full([C_PAD], NEG_INF, dtype=tl.float64)
                                l_beta_k = tl.zeros([C_PAD], dtype=tl.float64)

                                # Clamp alpha_t once per k (reused across tiles)
                                alpha_t_clamped = tl.minimum(tl.maximum(alpha_t, -1e6), 1e6)

                                # === Tile loop over c_dst dimension ===
                                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                                    # Tile indices
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

                                    # === Compute marginal tile (TILE_C, C_PAD) ===
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

                                    # === Accumulate gradients from this tile ===

                                    # Boundary marginals: sum over both dims
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
                                    # grad_cum_scores[t]: -marginal (same position for all k)
                                    # Note: We use atomic_add here because scatter with constexpr
                                    # indices is not supported in Triton. This is still more efficient
                                    # than the non-tiled version since we write (TILE_C,) per tile.
                                    # FUTURE OPTIMIZATION: Accumulate locally across all k and tiles,
                                    # then write once after the k-loop. Requires restructuring loops
                                    # (tile outside k, or C_PAD-sized local accumulator with scatter).
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

                                    # === Online logsumexp for beta_k (Flash Attention pattern) ===
                                    # Instead of materializing full (C_PAD, C_PAD) and reducing,
                                    # we accumulate logsumexp across tiles using online algorithm:
                                    #   m = running max, l = running sum of exp(x - m)
                                    #   For each new tile: m' = max(m, tile_max)
                                    #                      l' = l * exp(m - m') + tile_sum * exp(tile_max - m')
                                    # Final result: m + log(l)
                                    scores_for_beta_tile = edge_tile + beta_tile[:, None]
                                    scores_for_beta_tile = tl.where(
                                        tile_mask_2d, scores_for_beta_tile, NEG_INF
                                    )

                                    # Tile statistics: max and sum(exp) over c_dst dimension
                                    max_tile = tl.max(scores_for_beta_tile, axis=0)
                                    # NEG_INF guard: if all inputs are NEG_INF, max would be NEG_INF
                                    # and scores - max = 0, giving exp(0) = 1 (wrong!). Detect and handle.
                                    is_tile_neginf = max_tile < (NEG_INF + 1.0)
                                    max_tile_safe = tl.where(is_tile_neginf, 0.0, max_tile)

                                    sum_exp_tile = tl.sum(
                                        tl.exp(scores_for_beta_tile - max_tile_safe[None, :]),
                                        axis=0,
                                    )
                                    sum_exp_tile = tl.where(is_tile_neginf, 0.0, sum_exp_tile)

                                    # Online update: merge this tile's statistics with running accumulator
                                    m_new = tl.maximum(m_beta_k, max_tile)
                                    is_m_neginf = m_beta_k < (NEG_INF + 1.0)
                                    m_new_safe = tl.where(is_m_neginf & is_tile_neginf, 0.0, m_new)

                                    # Rescale previous sum and add new tile's contribution
                                    l_beta_k = tl.where(
                                        is_m_neginf,
                                        sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                        l_beta_k * tl.exp(m_beta_k - m_new_safe)
                                        + sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                    )
                                    m_beta_k = m_new

                                # === After all c_dst tiles: finalize beta_k ===
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
        num_warps: int = 4,
        validate_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""launch_streaming_triton_backward(cum_scores, transition, duration_bias, lengths, log_Z, ring_checkpoints, checkpoint_interval, grad_output, proj_start=None, proj_end=None, return_boundary_marginals=False, num_warps=4, validate_cache=True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

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
            return_boundary_marginals (bool, optional): If ``True``, also compute
                and return boundary marginals. Default: ``False``
            num_warps (int, optional): Number of warps per block for Triton kernel.
                Higher values increase parallelism but also register pressure.
                Recommended range: 2-8. Default: ``4``
            validate_cache (bool, optional): If True, validate Triton cache
                consistency and warn on config changes. Default: ``True``

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

        # NUMERICAL STABILITY FIX: Use float64 for gradient accumulation to reduce
        # floating-point non-associativity errors from atomic_add operations.
        # The error accumulates over T×K×C operations (e.g., 585,000 at T=500, K=30, C=39).
        # Float32 error: ~1e-7 per op → ~1e-3 accumulated
        # Float64 error: ~1e-16 per op → ~1e-10 accumulated
        # We convert back to original dtype before returning.
        accum_dtype = torch.float64

        # Allocate gradient outputs with C_PAD
        # grad_cum_scores uses float32: each batch writes to its own slice (no cross-batch atomics)
        # so float32 precision is sufficient. This reduces memory bandwidth by 50%.
        grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=torch.float32)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=accum_dtype)

        # Allocate per-batch workspace buffers with higher precision
        # IMPORTANT: Allocate with C_PAD to prevent OOB memory access from masked-out
        # threads (indices C to C_PAD-1).
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
                TILE_C=32,  # Increased from 16 to reduce tile iterations (4→2 for C_PAD=64)
                num_warps=num_warps,
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
        # Slice workspaces back to actual class count C before einsum reduction
        # (they were allocated with C_PAD to prevent OOB memory access)
        # Convert grad_output to float64 to match workspace dtype for einsum
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
        # grad_cum_scores: float32 → original dtype (may be no-op if dtype is float32)
        # grad_transition/grad_duration_bias: float64 → original dtype
        grad_cum_scores = grad_cum_scores[:, :, :C].to(dtype)
        grad_transition = grad_transition.to(dtype)
        grad_duration_bias = grad_duration_bias.to(dtype)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start[:, :, :C].to(dtype)
            grad_proj_end = grad_proj_end[:, :, :C].to(dtype)

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
        r"""launch_streaming_triton_marginals(cum_scores, transition, duration_bias, lengths, log_Z, ring_checkpoints, checkpoint_interval, proj_start=None, proj_end=None) -> Tensor

        Compute boundary marginals via Triton backward kernel.

        This is a convenience wrapper that runs the backward kernel with
        ``return_boundary_marginals=True`` and discards the gradient outputs.
        The boundary marginal at position t represents the probability that
        a segment starts at that position.

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
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``

        Returns:
            Tensor: Boundary marginals of shape :math:`(\text{batch}, T)`.
                Each value represents the probability that a segment starts
                at that position.

        .. note::
            Requires CUDA and Triton. For CPU computation, use
            :func:`~torch_semimarkov.streaming.semi_crf_streaming_marginals_pytorch`.

        Example::

            >>> # Setup (on CUDA)
            >>> cum_scores = torch.zeros(2, 101, 4, device='cuda')
            >>> cum_scores[:, 1:] = torch.cumsum(torch.randn(2, 100, 4, device='cuda'), dim=1)
            >>> transition = torch.randn(4, 4, device='cuda')
            >>> duration_bias = torch.randn(10, 4, device='cuda')  # K=10
            >>> lengths = torch.tensor([100, 80], device='cuda')
            >>> # Forward pass to get checkpoints
            >>> log_Z, ring_ckpts, interval = launch_streaming_triton_kernel(
            ...     cum_scores, transition, duration_bias, lengths, K=10
            ... )
            >>> # Compute boundary marginals
            >>> marginals = launch_streaming_triton_marginals(
            ...     cum_scores, transition, duration_bias, lengths,
            ...     log_Z, ring_ckpts, interval
            ... )
            >>> marginals.shape
            torch.Size([2, 100])

        See Also:
            :func:`launch_streaming_triton_backward`: Full backward pass with gradients
            :func:`~torch_semimarkov.streaming.semi_crf_streaming_marginals_pytorch`:
                PyTorch reference implementation (CPU compatible)
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
