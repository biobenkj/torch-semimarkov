"""Triton backward kernel for streaming Semi-CRF.

This module contains the Triton kernel for the backward pass and the launcher
function that allocates buffers and dispatches the kernel.
"""

from typing import Tuple

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
        grad_tr_workspace_ptr,  # (batch, C, C) - per-batch accumulator, no atomic needed
        grad_db_workspace_ptr,  # (batch, K, C) - per-batch accumulator, no atomic needed
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
        # Strides for grad_tr_workspace (batch, C, C)
        stride_gtw_b,
        stride_gtw_src,
        stride_gtw_dst,
        # Strides for grad_db_workspace (batch, K, C)
        stride_gdbw_b,
        stride_gdbw_k,
        stride_gdbw_c,
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
        grad_tr_ws_base = grad_tr_workspace_ptr + batch_idx * stride_gtw_b
        grad_db_ws_base = grad_db_workspace_ptr + batch_idx * stride_gdbw_b

        # Load transition matrix into registers
        transition_block = tl.load(
            transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
            mask=c_mask_2d,
            other=0.0,
        )  # (C_PAD, C_PAD) - this is transition.T

        # Initialize beta ring buffer at final positions
        final_pos = seq_len - 1
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
                for k_slot in tl.range(0, K):
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
                        for k in tl.range(1, K):
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

                        for k in tl.range(1, K):
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

                                # grad_transition: write to per-batch workspace (unscaled marginal)
                                # marginal is (C_dst, C_src), grad_transition[c_src, c_dst] += marginal[c_dst, c_src]
                                marginal_T = tl.trans(marginal)  # (C_src, C_dst) = (C_PAD, C_PAD)
                                # Compute 2D offsets for workspace: grad_tr_workspace[batch, src, dst]
                                # c_dst_idx is (C_PAD, 1) for src, c_src_idx is (1, C_PAD) for dst
                                tr_offsets_ws = c_dst_idx * stride_gtw_src + c_src_idx * stride_gtw_dst
                                # atomic_add still needed within batch (multiple t,k iterations add to same location)
                                # but no inter-batch contention since each batch has its own workspace slice
                                tl.atomic_add(grad_tr_ws_base + tr_offsets_ws, marginal_T, mask=c_mask_2d)

                                # grad_duration_bias: write to per-batch workspace (unscaled)
                                # grad_db_workspace[batch, k, c_dst] += sum over c_src
                                tl.atomic_add(
                                    grad_db_ws_base + k * stride_gdbw_k + c_idx * stride_gdbw_c,
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

        # Allocate per-batch workspace buffers to avoid atomic add contention
        # Each batch element accumulates to its own slice, then we sum after kernel
        grad_tr_workspace = torch.zeros(batch, C, C, device=device, dtype=dtype)
        grad_db_workspace = torch.zeros(batch, K, C, device=device, dtype=dtype)

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
            grad_tr_workspace,
            grad_db_workspace,
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
            stride_gtw_b,
            stride_gtw_src,
            stride_gtw_dst,
            stride_gdbw_b,
            stride_gdbw_k,
            stride_gdbw_c,
        )

        # Sum workspace buffers across batch dimension to get final shared gradients
        grad_transition = grad_tr_workspace.sum(dim=0)
        grad_duration_bias = grad_db_workspace.sum(dim=0)

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
