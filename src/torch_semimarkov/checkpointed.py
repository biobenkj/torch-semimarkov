r"""Checkpointed backward pass implementations for Semi-CRF.

This module provides memory-efficient backward pass implementations using
gradient checkpointing, optimized for large-scale sequences (T up to 400K+)
and long segment durations (K up to 3K+).

Memory Complexity
-----------------
The checkpoint interval S is computed as √(T×K) to minimize total memory:

    Memory = (T/S)×K×C + S×C + K×C

Taking derivative and solving: S* = √(T×K), giving:

    Memory* ≈ 2×√(T×K)×C + K×C

For T=400K, K=3K, C=24: ~7 MB/batch (vs 182 MB with naive √T interval)

Implementations
---------------
1. Optimized Checkpointing (O(T) compute, O(√(T×K)×C) memory):
   - Saves entire ring buffer state at every √(T×K) positions
   - Recomputes α only within each segment
   - Uses `semi_crf_optimized_checkpointed_backward`
   - Best for: large K, long sequences

2. Triton Checkpointed (O(T) compute, O(√(T×K)×C) memory, GPU-accelerated):
   - Same strategy as optimized, but with Triton kernels for backward
   - Uses `semi_crf_triton_checkpointed_backward`
   - Best for: GPU training with long sequences (RECOMMENDED)
"""

import math
from typing import Optional

import torch

from .backward import (
    HAS_TRITON,
    NEG_INF,
    _next_power_of_2,
)

# Import Triton if available
if HAS_TRITON:
    import triton
    import triton.language as tl


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_checkpoint_interval(T: int, K: int = 1) -> int:
    """Compute optimal checkpoint interval to minimize total memory.

    The optimal interval S minimizes: Memory = (T/S)×K×C + S×C + K×C

    Taking d/dS = 0 gives: S* = √(T×K)

    This is critical for large K: with K=3000, T=400000:
    - Old formula √T = 632 → 633 checkpoints × 72K floats = 182 MB
    - New formula √(T×K) = 34641 → 11.5 checkpoints × 72K floats = 3.3 MB

    The interval must also be >= K to ensure the ring buffer state at
    each checkpoint contains all required α values for recomputation.

    Args:
        T: Sequence length.
        K: Maximum duration.

    Returns:
        Optimal checkpoint interval (at least K).
    """
    # Optimal interval is √(T×K), but must be at least K for correctness
    optimal = int(math.sqrt(T * K))
    return max(K, optimal)


# =============================================================================
# Optimized Checkpointed Backward: O(T) compute, O(√(T×K) × K × C) memory
# =============================================================================


def semi_crf_forward_with_ring_checkpoints(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Forward pass that saves ring buffer state at checkpoint intervals.

    This saves the entire ring buffer (K × C values) at each checkpoint,
    not just α[checkpoint_pos] (C values). This enables O(T) backward
    instead of O(T^1.5) at the cost of K× more checkpoint memory.

    Memory: O(√(T×K) × K × C) for checkpoints
    Compute: O(T) for forward

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √(T×K).
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
        ring_checkpoints: Ring buffer states at checkpoints, shape (batch, num_checkpoints, K, C).
        checkpoint_interval: The actual interval used.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device
    dtype = edge.dtype

    # Determine checkpoint interval
    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(T, K)
    else:
        checkpoint_interval = max(checkpoint_interval, K)

    # Number of checkpoints needed
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

    # Allocate ring checkpoint storage: (batch, num_checkpoints, K, C)
    ring_checkpoints = torch.full(
        (batch, num_checkpoints, K, C), NEG_INF, device=device, dtype=dtype
    )

    # Use ring buffer for forward pass
    alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
    alpha_ring[:, 0, :] = 0.0  # α[0] = 0

    # Store initial ring buffer state (checkpoint 0)
    ring_checkpoints[:, 0, :, :] = alpha_ring

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)
    final_positions = lengths - 1

    # Handle sequences of length 1
    len_1_mask = lengths == 1
    if len_1_mask.any():
        final_alpha[len_1_mask] = 0.0

    for t in range(1, T):
        active_mask = t < lengths

        k_eff = min(K - 1, t)
        scores_all = []

        for k in range(1, k_eff + 1):
            start = t - k
            ring_idx = start % K
            alpha_prev = alpha_ring[:, ring_idx, :]
            edge_k = edge[:, start, k, :, :]
            scores = alpha_prev.unsqueeze(-2) + edge_k
            scores_all.append(scores)

        scores_stacked = torch.stack(scores_all, dim=1)

        if semiring == "log":
            scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
            alpha_t = torch.logsumexp(scores_over_src, dim=1)
        else:
            scores_over_src = torch.max(scores_stacked, dim=-1)[0]
            alpha_t = torch.max(scores_over_src, dim=1)[0]

        # Update ring buffer
        ring_idx_t = t % K
        alpha_ring[:, ring_idx_t, :] = torch.where(
            active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
        )

        # Save ring buffer state at checkpoint positions
        if t % checkpoint_interval == 0:
            ckpt_idx = t // checkpoint_interval
            if ckpt_idx < num_checkpoints:
                # Save entire ring buffer state
                for k_slot in range(K):
                    ring_checkpoints[:, ckpt_idx, k_slot, :] = torch.where(
                        active_mask.view(batch, 1),
                        alpha_ring[:, k_slot, :],
                        ring_checkpoints[:, ckpt_idx, k_slot, :],
                    )

        # Track final alpha
        is_final = t == final_positions
        if is_final.any():
            final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)

    if semiring == "log":
        partition = torch.logsumexp(final_alpha, dim=-1)
    else:
        partition = torch.max(final_alpha, dim=-1)[0]

    return partition, ring_checkpoints, checkpoint_interval


def semi_crf_backward_from_ring_checkpoints(
    edge: torch.Tensor,
    ring_checkpoints: torch.Tensor,
    log_Z: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: int,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Backward pass using ring buffer checkpoints for O(T) compute.

    This recomputes α values only within each segment (not from position 0),
    using the saved ring buffer state at each checkpoint as the starting point.

    Compute: O(T) total (only recompute within segments)
    Memory: O(interval × C + K × C) working memory

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        ring_checkpoints: Saved ring buffer states, shape (batch, num_checkpoints, K, C).
        log_Z: Log partition values of shape (batch,).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints.
        semiring: Either "log" or "max".

    Returns:
        grad_edge: Gradient w.r.t. edge of shape (batch, T-1, K, C, C).
    """
    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device
    dtype = edge.dtype

    effective_interval = max(checkpoint_interval, K)

    # Initialize gradient output
    grad_edge = torch.zeros_like(edge)

    # Segment buffer for α values
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # β ring buffer
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize β at final positions
    final_positions = lengths - 1
    for b in range(batch):
        final_ring_idx = final_positions[b].item() % K
        beta_ring[b, final_ring_idx, :] = 0.0

    num_checkpoints = ring_checkpoints.shape[1]

    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = min((ckpt_idx + 1) * checkpoint_interval, T)

        # Clear segment buffer
        alpha_segment.fill_(NEG_INF)

        # === Phase 1: Recompute α from checkpoint's ring buffer state ===
        # Load the saved ring buffer state for this checkpoint
        alpha_ring = ring_checkpoints[:, ckpt_idx, :, :].clone()

        # Store α[seg_start] at local position 0
        # α[seg_start] is in ring_checkpoints[:, ckpt_idx, seg_start % K, :]
        alpha_segment[:, 0, :] = alpha_ring[:, seg_start % K, :]

        # Recompute α for positions seg_start+1 to seg_end-1
        for t in range(seg_start + 1, seg_end):
            active_mask = t < lengths

            k_eff = min(K - 1, t)
            scores_all = []

            for k in range(1, k_eff + 1):
                start = t - k
                ring_idx = start % K
                alpha_prev = alpha_ring[:, ring_idx, :]
                edge_k = edge[:, start, k, :, :]
                scores = alpha_prev.unsqueeze(-2) + edge_k
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

        # === Phase 2: Compute β backward and gradients ===
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
                edge_k = edge[:, t, k, :, :]

                # Gradient computation
                log_marginal = (
                    alpha_t.unsqueeze(-2)
                    + edge_k
                    + beta_next.unsqueeze(-1)
                    - log_Z.view(batch, 1, 1)
                )
                marginal = torch.exp(log_marginal)
                marginal = torch.where(
                    valid_mask.view(batch, 1, 1), marginal, torch.zeros_like(marginal)
                )
                grad_edge[:, t, k, :, :] = marginal

                # β contribution
                scores_for_beta = edge_k + beta_next.unsqueeze(-1)
                scores_for_beta = torch.where(
                    valid_mask.view(batch, 1, 1),
                    scores_for_beta,
                    torch.full_like(scores_for_beta, NEG_INF),
                )
                new_beta_scores.append(scores_for_beta)

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

    return grad_edge


class SemiCRFOptimizedCheckpointedBackward(torch.autograd.Function):
    r"""Optimized checkpointed backward with O(T) compute.

    This achieves O(√(T×K) × K × C) memory by saving ring buffer state at checkpoints,
    enabling O(T) backward compute instead of O(T^1.5).

    Memory: O(√(T×K) × K × C) for checkpoints + O(interval × C) working memory
    Compute: O(T) for backward (only recompute within segments)
    """

    @staticmethod
    def forward(
        ctx,
        edge: torch.Tensor,
        lengths: torch.Tensor,
        checkpoint_interval: Optional[int] = None,
        semiring: str = "log",
    ) -> torch.Tensor:
        partition, ring_checkpoints, actual_interval = semi_crf_forward_with_ring_checkpoints(
            edge.detach(), lengths, checkpoint_interval, semiring
        )

        ctx.save_for_backward(edge, lengths, ring_checkpoints, partition)
        ctx.checkpoint_interval = actual_interval
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Optional[torch.Tensor], None, None, None]:
        edge, lengths, ring_checkpoints, partition = ctx.saved_tensors
        checkpoint_interval = ctx.checkpoint_interval
        semiring = ctx.semiring

        marginals = semi_crf_backward_from_ring_checkpoints(
            edge, ring_checkpoints, partition, lengths, checkpoint_interval, semiring
        )

        grad_edge = marginals * grad_output.view(-1, 1, 1, 1, 1)

        return grad_edge, None, None, None


def semi_crf_optimized_checkpointed_backward(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition with optimized O(T) checkpointed backward.

    This saves ring buffer state (K × C values) at each checkpoint instead of
    just α[checkpoint_pos] (C values), enabling O(T) backward compute.

    Memory: O(√(T×K) × K × C) for checkpoints + O(interval × C) working memory
    Compute: O(T) for backward (only recompute within segments)

    For T=400K, K=3K, C=24: ~7 MB/batch

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √(T×K).
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFOptimizedCheckpointedBackward.apply(edge, lengths, checkpoint_interval, semiring)


# =============================================================================
# Triton Checkpointed Backward Kernels (GPU only)
# =============================================================================

if HAS_TRITON:

    @triton.jit
    def _semi_crf_ckpt_segment_forward_kernel(
        # Inputs
        edge_ptr,  # (batch, T-1, K, C, C)
        ring_checkpoints_ptr,  # (batch, num_checkpoints, K, C_PAD)
        lengths_ptr,  # (batch,)
        # Outputs
        alpha_segment_ptr,  # (batch, checkpoint_interval, C_PAD) - segment alpha buffer
        # Segment info
        ckpt_idx,  # which checkpoint/segment to process
        # Dimensions
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        checkpoint_interval: tl.constexpr,
        USE_FP64: tl.constexpr,
        # Strides for edge tensor (batch, T-1, K, C, C)
        stride_eb,
        stride_et,
        stride_ek,
        stride_ec1,
        stride_ec2,
        # Strides for ring_checkpoints (batch, num_checkpoints, K, C_PAD)
        stride_rcb,
        stride_rci,
        stride_rck,
        stride_rcc,
        # Strides for alpha_segment (batch, checkpoint_interval, C_PAD)
        stride_asb,
        stride_ast,
        stride_asc,
    ):
        """
        Forward pass within a segment: recompute α values from checkpoint.

        This kernel loads the ring buffer state from the checkpoint and
        forward propagates to compute α values for all positions in the segment.
        Results are stored in alpha_segment_ptr for use by the backward kernel.
        """
        NEG_INF: tl.constexpr = -1e9

        if USE_FP64:
            DTYPE = tl.float64
        else:
            DTYPE = tl.float32

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # Label indices
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C
        c_dest = tl.arange(0, C_PAD)[:, None]
        c_src = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dest < C) & (c_src < C)

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Segment boundaries
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = tl.minimum((ckpt_idx + 1) * checkpoint_interval, seq_len)

        # Skip if segment is beyond sequence
        if seg_start >= seq_len:
            return

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        ckpt_base = ring_checkpoints_ptr + batch_idx * stride_rcb
        alpha_seg_base = alpha_segment_ptr + batch_idx * stride_asb

        # Process all positions in segment
        for t_local in tl.range(0, checkpoint_interval):
            t = seg_start + t_local
            active = t < seg_end

            if active:
                if t == seg_start:
                    # First position: alpha[seg_start] is in the checkpoint
                    ring_slot = seg_start % K
                    alpha_t = tl.load(
                        ckpt_base
                        + ckpt_idx * stride_rci
                        + ring_slot * stride_rck
                        + c_idx * stride_rcc,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                else:
                    # Compute alpha[t] = logsumexp over k of contribution from alpha[t-k]
                    alpha_t = tl.full([C_PAD], NEG_INF, dtype=DTYPE)

                    for k in tl.range(1, K):
                        k_valid = (k <= t) & (k <= K - 1)
                        start_pos = t - k

                        # Compute safe index for loads
                        start_pos_safe = tl.maximum(start_pos, 0)
                        ring_slot = start_pos_safe % K

                        # Determine source: checkpoint (before segment) or segment buffer
                        from_checkpoint = k_valid & (start_pos < seg_start)
                        from_segment = k_valid & (start_pos >= seg_start)

                        # Load alpha[start_pos] from appropriate source
                        alpha_ckpt = tl.load(
                            ckpt_base
                            + ckpt_idx * stride_rci
                            + ring_slot * stride_rck
                            + c_idx * stride_rcc,
                            mask=from_checkpoint & c_mask,
                            other=NEG_INF,
                        )
                        local_idx = tl.maximum(start_pos - seg_start, 0)
                        alpha_seg = tl.load(
                            alpha_seg_base + local_idx * stride_ast + c_idx * stride_asc,
                            mask=from_segment & c_mask,
                            other=NEG_INF,
                        )
                        alpha_prev = tl.where(from_checkpoint, alpha_ckpt, alpha_seg)

                        # Load edge[start_pos, k, :, :]
                        edge_offset_2d = (
                            edge_base
                            + start_pos_safe * stride_et
                            + k * stride_ek
                            + c_dest * stride_ec1
                            + c_src * stride_ec2
                        )
                        edge_block = tl.load(
                            edge_offset_2d,
                            mask=k_valid & c_mask_2d,
                            other=NEG_INF,
                        )

                        # Compute scores
                        scores = alpha_prev[None, :] + edge_block
                        scores = tl.where(c_mask_2d, scores, NEG_INF)

                        # Logsumexp over source labels (axis=1)
                        max_scores = tl.max(scores, axis=1)
                        score_for_k = max_scores + tl.log(
                            tl.sum(tl.exp(scores - max_scores[:, None]), axis=1)
                        )
                        score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                        # Accumulate into alpha_t
                        max_ab = tl.maximum(alpha_t, score_for_k)
                        alpha_t = max_ab + tl.log(
                            tl.exp(alpha_t - max_ab) + tl.exp(score_for_k - max_ab)
                        )

                # Store alpha[t] to segment buffer for backward pass
                tl.store(
                    alpha_seg_base + t_local * stride_ast + c_idx * stride_asc,
                    alpha_t,
                    mask=c_mask,
                )

    @triton.jit
    def _semi_crf_ckpt_segment_backward_kernel(
        # Inputs
        edge_ptr,  # (batch, T-1, K, C, C)
        alpha_segment_ptr,  # (batch, checkpoint_interval, C_PAD) - pre-computed alpha
        log_Z_ptr,  # (batch,)
        lengths_ptr,  # (batch,)
        beta_ring_ptr,  # (batch, K, C_PAD) - persistent beta ring buffer
        # Outputs
        grad_edge_ptr,  # (batch, T-1, K, C, C)
        # Segment info
        ckpt_idx,  # which checkpoint/segment to process
        # Dimensions
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        checkpoint_interval: tl.constexpr,
        USE_FP64: tl.constexpr,
        # Strides for edge tensor
        stride_eb,
        stride_et,
        stride_ek,
        stride_ec1,
        stride_ec2,
        # Strides for alpha_segment
        stride_asb,
        stride_ast,
        stride_asc,
        # Strides for beta_ring
        stride_brb,
        stride_brk,
        stride_brc,
        # Strides for grad_edge
        stride_gb,
        stride_gt,
        stride_gk,
        stride_gc1,
        stride_gc2,
    ):
        """
        Backward pass within a segment: compute β and gradients.

        This kernel uses pre-computed α values from alpha_segment_ptr and
        computes β values backward through the segment while computing gradients.
        The β ring buffer is updated and persists across segments.
        """
        NEG_INF: tl.constexpr = -1e9

        if USE_FP64:
            DTYPE = tl.float64
        else:
            DTYPE = tl.float32

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # Label indices
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C
        c_dest = tl.arange(0, C_PAD)[:, None]
        c_src = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dest < C) & (c_src < C)

        # Load sequence length and log_Z
        seq_len = tl.load(lengths_ptr + batch_idx)
        log_Z = tl.load(log_Z_ptr + batch_idx)

        # Segment boundaries
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = tl.minimum((ckpt_idx + 1) * checkpoint_interval, seq_len)

        # Skip if segment is beyond sequence
        if seg_start >= seq_len:
            return

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        alpha_seg_base = alpha_segment_ptr + batch_idx * stride_asb
        beta_base = beta_ring_ptr + batch_idx * stride_brb
        grad_base = grad_edge_ptr + batch_idx * stride_gb

        # Process positions backward
        for t_offset in tl.range(0, checkpoint_interval):
            t = (seg_end - 1) - t_offset
            t_active = (t >= seg_start) & (t < seq_len - 1)

            if t_active:
                # Load alpha[t] from segment buffer
                t_local = t - seg_start
                t_local_safe = tl.maximum(t_local, 0)
                alpha_t = tl.load(
                    alpha_seg_base + t_local_safe * stride_ast + c_idx * stride_asc,
                    mask=c_mask,
                    other=NEG_INF,
                )

                # Compute new beta[t] and gradients
                new_beta = tl.full([C_PAD], NEG_INF, dtype=DTYPE)
                k_max = tl.minimum(K - 1, seq_len - 1 - t)

                # Safe t for edge indexing
                t_safe = tl.minimum(tl.maximum(t, 0), T - 2)

                for k in tl.range(1, K):
                    end_pos = t + k
                    k_active = (k <= k_max) & (end_pos < seq_len)
                    end_pos_safe = tl.minimum(tl.maximum(end_pos, 0), T - 1)

                    # Load beta[end_pos] from ring buffer
                    ring_k_idx = end_pos_safe % K
                    beta_end = tl.load(
                        beta_base + ring_k_idx * stride_brk + c_idx * stride_brc,
                        mask=k_active & c_mask,
                        other=NEG_INF,
                    )

                    # Load edge[t, k, :, :]
                    edge_offset_2d = (
                        edge_base
                        + t_safe * stride_et
                        + k * stride_ek
                        + c_dest * stride_ec1
                        + c_src * stride_ec2
                    )
                    edge_block = tl.load(
                        edge_offset_2d,
                        mask=k_active & c_mask_2d,
                        other=NEG_INF,
                    )

                    # === Gradient computation ===
                    log_marginal = alpha_t[None, :] + edge_block + beta_end[:, None] - log_Z
                    marginal = tl.exp(log_marginal)
                    marginal = tl.where(k_active & c_mask_2d, marginal, 0.0)

                    # Store gradient
                    grad_offset_2d = (
                        grad_base
                        + t_safe * stride_gt
                        + k * stride_gk
                        + c_dest * stride_gc1
                        + c_src * stride_gc2
                    )
                    tl.store(grad_offset_2d, marginal, mask=k_active & c_mask_2d)

                    # === Beta contribution ===
                    scores_for_beta = edge_block + beta_end[:, None]
                    scores_for_beta = tl.where(k_active & c_mask_2d, scores_for_beta, NEG_INF)

                    # logsumexp over c_dest (axis=0) to get [C_PAD] indexed by c_src
                    max_s = tl.max(scores_for_beta, axis=0)
                    beta_contrib = max_s + tl.log(
                        tl.sum(tl.exp(scores_for_beta - max_s[None, :]), axis=0)
                    )
                    beta_contrib = tl.where(k_active & c_mask, beta_contrib, NEG_INF)

                    # Accumulate into new_beta
                    max_nb = tl.maximum(new_beta, beta_contrib)
                    new_beta_candidate = max_nb + tl.log(
                        tl.exp(new_beta - max_nb) + tl.exp(beta_contrib - max_nb)
                    )
                    new_beta = tl.where(k_active, new_beta_candidate, new_beta)

                # Store beta[t] to ring buffer
                ring_t_idx = t % K
                tl.store(
                    beta_base + ring_t_idx * stride_brk + c_idx * stride_brc,
                    new_beta,
                    mask=c_mask,
                )

    def launch_triton_checkpointed_backward_kernel(
        edge: torch.Tensor,
        ring_checkpoints: torch.Tensor,
        log_Z: torch.Tensor,
        lengths: torch.Tensor,
        checkpoint_interval: int,
    ) -> torch.Tensor:
        r"""Launch the Triton checkpointed backward kernel.

        This uses a two-pass approach per segment:
        1. Forward pass: recompute α values within segment from checkpoint
        2. Backward pass: compute β and gradients using stored α values

        This achieves O(T) compute by only recomputing within segments.

        Args:
            edge: Log potentials of shape (batch, T-1, K, C, C).
            ring_checkpoints: Saved ring buffer states of shape (batch, num_checkpoints, K, C).
            log_Z: Log partition values of shape (batch,).
            lengths: Sequence lengths of shape (batch,).
            checkpoint_interval: Interval between checkpoints.

        Returns:
            grad_edge: Gradient w.r.t. edge of shape (batch, T-1, K, C, C).
        """
        batch, T_minus_1, K, C, _ = edge.shape
        T = T_minus_1 + 1
        num_checkpoints = ring_checkpoints.shape[1]

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Ensure inputs are contiguous
        edge = edge.contiguous()
        lengths = lengths.contiguous()

        # Pad ring_checkpoints if needed
        if C_PAD != C:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD),
                NEG_INF,
                device=edge.device,
                dtype=edge.dtype,
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints
        else:
            ring_ckpts_padded = ring_checkpoints.contiguous()

        # Allocate alpha segment buffer (reused across segments)
        alpha_segment = torch.full(
            (batch, checkpoint_interval, C_PAD),
            NEG_INF,
            device=edge.device,
            dtype=edge.dtype,
        )

        # Allocate beta ring buffer (persists across segment kernels)
        beta_ring = torch.full((batch, K, C_PAD), NEG_INF, device=edge.device, dtype=edge.dtype)

        # Initialize beta at final positions
        for b in range(batch):
            final_pos = lengths[b].item() - 1
            final_ring_idx = final_pos % K
            beta_ring[b, final_ring_idx, :C] = 0.0

        # Allocate output gradient tensor
        grad_edge = torch.zeros_like(edge)

        # Get strides
        stride_eb, stride_et, stride_ek, stride_ec1, stride_ec2 = edge.stride()
        stride_rcb, stride_rci, stride_rck, stride_rcc = ring_ckpts_padded.stride()
        stride_asb, stride_ast, stride_asc = alpha_segment.stride()
        stride_brb, stride_brk, stride_brc = beta_ring.stride()
        stride_gb, stride_gt, stride_gk, stride_gc1, stride_gc2 = grad_edge.stride()

        USE_FP64 = edge.dtype == torch.float64

        # Process segments in reverse order
        grid = (batch,)
        for ckpt_idx in range(num_checkpoints - 1, -1, -1):
            # Pass 1: Forward - recompute alpha values within segment
            _semi_crf_ckpt_segment_forward_kernel[grid](
                edge,
                ring_ckpts_padded,
                lengths,
                alpha_segment,
                ckpt_idx,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                USE_FP64,
                stride_eb,
                stride_et,
                stride_ek,
                stride_ec1,
                stride_ec2,
                stride_rcb,
                stride_rci,
                stride_rck,
                stride_rcc,
                stride_asb,
                stride_ast,
                stride_asc,
            )

            # Pass 2: Backward - compute beta and gradients
            _semi_crf_ckpt_segment_backward_kernel[grid](
                edge,
                alpha_segment,
                log_Z,
                lengths,
                beta_ring,
                grad_edge,
                ckpt_idx,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                USE_FP64,
                stride_eb,
                stride_et,
                stride_ek,
                stride_ec1,
                stride_ec2,
                stride_asb,
                stride_ast,
                stride_asc,
                stride_brb,
                stride_brk,
                stride_brc,
                stride_gb,
                stride_gt,
                stride_gk,
                stride_gc1,
                stride_gc2,
            )

        return grad_edge


# =============================================================================
# Triton Checkpointed Autograd Function
# =============================================================================


class SemiCRFTritonCheckpointedBackward(torch.autograd.Function):
    r"""Autograd function using Triton kernel for checkpointed backward.

    This combines:
    - Forward: PyTorch implementation saving ring buffer checkpoints
    - Backward: Triton kernel for GPU-accelerated checkpointed backward

    Memory: O(√(T×K) × K × C) for checkpoints
    Compute: O(T) for backward (Triton kernel)

    Falls back to PyTorch implementation on CPU.
    """

    @staticmethod
    def forward(
        ctx,
        edge: torch.Tensor,
        lengths: torch.Tensor,
        checkpoint_interval: Optional[int] = None,
        semiring: str = "log",
    ) -> torch.Tensor:
        # Compute forward pass with ring buffer checkpointing
        partition, ring_checkpoints, actual_interval = semi_crf_forward_with_ring_checkpoints(
            edge.detach(), lengths, checkpoint_interval, semiring
        )

        # Save for backward
        ctx.save_for_backward(edge, lengths, ring_checkpoints, partition)
        ctx.checkpoint_interval = actual_interval
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Optional[torch.Tensor], None, None, None]:
        edge, lengths, ring_checkpoints, partition = ctx.saved_tensors
        checkpoint_interval = ctx.checkpoint_interval
        semiring = ctx.semiring

        # Use Triton kernel if available and on CUDA
        if HAS_TRITON and edge.is_cuda and semiring == "log":
            marginals = launch_triton_checkpointed_backward_kernel(
                edge, ring_checkpoints, partition, lengths, checkpoint_interval
            )
        else:
            # Fall back to PyTorch implementation
            marginals = semi_crf_backward_from_ring_checkpoints(
                edge, ring_checkpoints, partition, lengths, checkpoint_interval, semiring
            )

        # Scale by upstream gradient
        grad_edge = marginals * grad_output.view(-1, 1, 1, 1, 1)

        return grad_edge, None, None, None


def semi_crf_triton_checkpointed_backward(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition using Triton kernel for checkpointed backward.

    This is the recommended function for GPU training with long sequences.
    It uses:
    - Forward: Ring buffer checkpointing (O(√(T×K) × K × C) memory)
    - Backward: Triton kernel for GPU-accelerated gradient computation (O(T) compute)

    For T=400K, K=3K, C=24: ~7 MB/batch

    Falls back to PyTorch implementation on CPU.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √(T×K).
        semiring: Either "log" or "max". Note: Triton kernel currently
            only supports "log" semiring.

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFTritonCheckpointedBackward.apply(edge, lengths, checkpoint_interval, semiring)
