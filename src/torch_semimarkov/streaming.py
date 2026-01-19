r"""Golden Rule Streaming API for memory-efficient Semi-CRF.

This module implements the "Golden Rule" optimization for Semi-CRF inference:
edge potentials are computed on-the-fly from pre-projected cumulative scores,
eliminating the need to materialize the full (batch, T-1, K, C, C) edge tensor.

Memory Complexity
-----------------
- Pre-computed edge API: O(T × K × C²) - 2.76 TB for T=400K, K=3K, C=24
- Golden Rule API: O(T × C + K × C + C²) - ~50 MB for same dimensions

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
"""

import math
import warnings
from typing import Optional, Tuple

import torch


NEG_INF = -1e9


def _compute_checkpoint_interval(T: int, K: int) -> int:
    """Compute optimal checkpoint interval to minimize total memory.

    The optimal interval S minimizes: Memory = (T/S)×K×C + S×C + K×C
    Taking d/dS = 0 gives: S* = √(T×K)

    Args:
        T: Sequence length.
        K: Maximum duration.

    Returns:
        Optimal checkpoint interval (at least K).
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
    """Compute edge block on-the-fly using the Golden Rule.

    This computes the edge potential for segments starting at position t
    with duration k, without materializing the full edge tensor.

    edge[c_dest, c_src] = segment_score[c_dest] + transition[c_src, c_dest]

    where segment_score = content_score + duration_bias + boundaries

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
            Must be float32 and zero-centered for numerical stability.
        transition: Label transition scores of shape (C, C).
            transition[c_src, c_dest] is the score for c_src -> c_dest.
        duration_bias: Duration-specific label bias of shape (K, C).
        t: Segment start position.
        k: Segment duration.
        proj_start: Optional start boundary scores of shape (batch, T, C).
        proj_end: Optional end boundary scores of shape (batch, T, C).

    Returns:
        edge_block: Edge potentials of shape (batch, C_dest, C_src).
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
    """Forward pass with Golden Rule edge computation.

    Computes the log partition function using a ring buffer with O(KC) memory.
    Edge potentials are computed on-the-fly from cumulative scores.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
            Must be float32. Should be zero-centered before cumsum.
        transition: Label transition scores of shape (C, C).
        duration_bias: Duration-specific label bias of shape (K, C).
        lengths: Sequence lengths of shape (batch,).
        K: Maximum segment duration.
        semiring: "log" (logsumexp) or "max" (Viterbi).
        proj_start: Optional start boundary scores of shape (batch, T, C).
        proj_end: Optional end boundary scores of shape (batch, T, C).
        checkpoint_interval: Interval for saving ring buffer state.
            Defaults to √(T×K).

    Returns:
        partition: Log partition function of shape (batch,).
        ring_checkpoints: Saved ring buffer states for backward.
        checkpoint_interval: Actual interval used.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Warn if cum_scores appears non-zero-centered
    endpoint_magnitude = cum_scores[:, -1, :].abs().mean().item()
    if endpoint_magnitude > 1000:
        warnings.warn(
            f"cum_scores endpoint magnitude {endpoint_magnitude:.0f} suggests "
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
    """Backward pass computing gradients via marginals.

    Uses the forward-backward algorithm with checkpointing. Recomputes alpha
    within segments from saved ring buffer checkpoints, then computes beta
    backward while accumulating gradients.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        duration_bias: Duration-specific label bias of shape (K, C).
        lengths: Sequence lengths of shape (batch,).
        K: Maximum segment duration.
        log_Z: Log partition values of shape (batch,).
        ring_checkpoints: Saved ring buffer states of shape (batch, num_checkpoints, K, C).
        checkpoint_interval: Interval between checkpoints.
        semiring: "log" or "max".
        proj_start: Optional start boundary scores of shape (batch, T, C).
        proj_end: Optional end boundary scores of shape (batch, T, C).

    Returns:
        grad_cum_scores: Gradient w.r.t. cum_scores of shape (batch, T+1, C).
        grad_transition: Gradient w.r.t. transition of shape (C, C).
        grad_duration_bias: Gradient w.r.t. duration_bias of shape (K, C).
        grad_proj_start: Gradient w.r.t. proj_start if provided, else None.
        grad_proj_end: Gradient w.r.t. proj_end if provided, else None.
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
    """Autograd function for streaming Semi-CRF with Golden Rule edge computation.

    This wraps the forward and backward passes to enable automatic differentiation.
    Memory usage is O(KC) for the ring buffer, independent of sequence length T.
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


def semi_crf_streaming_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Semi-CRF partition function with Golden Rule streaming.

    This is the main entry point for the streaming API. Edge potentials are
    computed on-the-fly from cumulative scores, eliminating the need for the
    full (batch, T-1, K, C, C) edge tensor.

    Memory: O(KC) ring buffer, independent of sequence length T.
    Compute: O(T × K × C²) same as standard Semi-CRF.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
            MUST be float32 for numerical stability at T>100K.
            SHOULD be zero-centered before cumsum to prevent precision loss.
        transition: Label transition scores of shape (C, C).
            transition[c_src, c_dest] is the score for c_src -> c_dest.
        duration_bias: Duration-specific label bias of shape (K, C).
            Required to compensate for sum-pooling length bias.
        lengths: Sequence lengths of shape (batch,).
        K: Maximum segment duration.
        semiring: "log" (logsumexp for partition) or "max" (Viterbi decoding).
        proj_start: Optional start boundary scores of shape (batch, T, C).
        proj_end: Optional end boundary scores of shape (batch, T, C).

    Returns:
        partition: Log partition function (or max score) of shape (batch,).

    Example:
        >>> # Encoder output
        >>> h = encoder(x)  # (batch, T, hidden_dim)
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
        >>> # Streaming forward
        >>> partition = semi_crf_streaming_forward(
        ...     cum_scores, transition, duration_bias, lengths, K
        ... )
    """
    return SemiCRFStreaming.apply(
        cum_scores, transition, duration_bias, lengths, K, semiring,
        proj_start, proj_end,
    )
