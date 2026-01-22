r"""PyTorch reference implementation for streaming Semi-CRF.

This module contains the pure PyTorch implementations used for:

- CPU fallback when Triton is not available
- Boundary projection support (``proj_start``, ``proj_end``)
- Reference implementation for testing Triton kernels

Functions:
    _compute_checkpoint_interval: Compute optimal checkpoint interval.
    compute_edge_block_streaming: Compute edge block on-the-fly.
    semi_crf_streaming_forward_pytorch: Forward pass with ring buffer.
    semi_crf_streaming_backward_pytorch: Backward pass via forward-backward algorithm.
"""

import math
import warnings
from typing import Optional

import torch

from .constants import NEG_INF


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


def compute_edge_block_streaming(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    t: int,
    k: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""compute_edge_block_streaming(cum_scores, transition, duration_bias, t, k, proj_start=None, proj_end=None) -> Tensor

    Compute edge block on-the-fly using prefix-sum decomposition.

    This computes the edge potential for segments starting at position ``t``
    with duration ``k``, without materializing the full edge tensor:

    .. math::
        \text{edge}[c_{\text{dest}}, c_{\text{src}}] = \text{segment\_score}[c_{\text{dest}}]
        + \text{transition}[c_{\text{src}}, c_{\text{dest}}]

    where :math:`\text{segment\_score} = \text{content\_score} + \text{duration\_bias} + \text{boundaries}`.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        t (int): Segment start position.
        k (int): Segment duration.
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``

    Returns:
        Tensor: Edge potentials of shape :math:`(\text{batch}, C, C)`.
    """
    # Content score via cumsum difference: (batch, C)
    content_score = cum_scores[:, t + k, :] - cum_scores[:, t, :]

    # Add duration bias (clamp k to valid range for K=1 case)
    K = duration_bias.shape[0]
    dur_idx = min(k, K - 1)
    segment_score = content_score + duration_bias[dur_idx]

    # Add boundary scores if provided
    if proj_start is not None:
        segment_score = segment_score + proj_start[:, t, :]
    if proj_end is not None:
        segment_score = segment_score + proj_end[:, t + k - 1, :]

    # Build edge block: segment_score[c_dest] + transition[c_src, c_dest]
    # segment_score: (batch, C) -> unsqueeze to (batch, C, 1) for c_dest
    # transition: (C_src, C_dest) -> transpose to (C_dest, C_src) -> (1, C_dest, C_src)
    # Result: (batch, C_dest, C_src)
    if transition.ndim == 2:
        # Static transitions: (C, C)
        trans_k = transition
    else:
        # Duration-dependent transitions: (K, C, C) - index by k
        trans_k = transition[k]

    edge_block = segment_score.unsqueeze(-1) + trans_k.T.unsqueeze(0)

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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""semi_crf_streaming_forward_pytorch(cum_scores, transition, duration_bias, lengths, K, semiring="log", proj_start=None, proj_end=None, checkpoint_interval=None) -> tuple[Tensor, Tensor, int]

    Forward pass with streaming edge computation (pure PyTorch reference).

    Computes the log partition function using a ring buffer with :math:`O(KC)` memory.
    Edge potentials are computed on-the-fly from cumulative scores.

    The forward recurrence is:

    .. math::
        \alpha[t, c] = \bigoplus_{k=1}^{K-1} \bigoplus_{c'} \alpha[t-k, c'] + \text{edge}[t-k, k, c, c']

    where :math:`\bigoplus` is logsumexp (log semiring) or max (max semiring).

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` (logsumexp) or ``"max"`` (Viterbi).
            Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        checkpoint_interval (int, optional): Interval for saving ring buffer state.
            If ``None``, uses :math:`\sqrt{T \times K}`. Default: ``None``

    Returns:
        tuple[Tensor, Tensor, int]: Tuple of:
            - **partition** (Tensor): Log partition values of shape :math:`(\text{batch},)`.
            - **ring_checkpoints** (Tensor): Saved ring buffer states of shape
              :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
            - **checkpoint_interval** (int): Actual interval used.
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
                "Zero-center before cumsum: projected = projected - projected.mean(dim=1, keepdim=True)",
                stacklevel=2,
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

    # Main forward loop
    for t in range(1, T + 1):
        # Include t == lengths to compute alpha at final position
        active_mask = t <= lengths

        # Number of valid durations at this position
        k_eff = min(K - 1, t)

        scores_all = []
        for k in range(1, max(k_eff + 1, 2)):  # max ensures K=1 processes duration 1
            start = t - k

            # Get alpha[start] from ring buffer
            ring_idx = start % K
            alpha_prev = alpha_ring[:, ring_idx, :]  # (batch, C_src)

            # Compute edge block on-the-fly (streaming)
            edge_block = compute_edge_block_streaming(
                cum_scores, transition, duration_bias, start, k, proj_start, proj_end
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

        # Track final alpha for sequences at their final position (t == lengths)
        # At iteration t, alpha_t represents segments ending at position t-1
        # For sequence of length L, we need alpha at t=L (segments ending at L-1)
        is_final = t == lengths
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
]:
    r"""semi_crf_streaming_backward_pytorch(cum_scores, transition, duration_bias, lengths, K, log_Z, ring_checkpoints, checkpoint_interval, semiring="log", proj_start=None, proj_end=None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    Backward pass computing gradients via marginals (pure PyTorch reference).

    Uses the forward-backward algorithm with checkpointing. Recomputes alpha
    within segments from saved ring buffer checkpoints, then computes beta
    backward while accumulating gradients.

    The marginal probability for a segment is:

    .. math::
        P(\text{segment}[t, k, c_{\text{dst}}, c_{\text{src}}]) =
        \frac{\exp(\alpha[t, c_{\text{src}}] + \text{edge}[t, k, c_{\text{dst}}, c_{\text{src}}]
        + \beta[t+k, c_{\text{dst}}])}{\exp(\log Z)}

    Gradients are accumulated as weighted sums of these marginals.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        log_Z (Tensor): Log partition values of shape :math:`(\text{batch},)`.
        ring_checkpoints (Tensor): Saved ring buffer states from forward pass.
        checkpoint_interval (int): Interval between checkpoints.
        semiring (str, optional): ``"log"`` or ``"max"``. Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of gradients:
            - **grad_cum_scores** (Tensor): Shape :math:`(\text{batch}, T+1, C)`.
            - **grad_transition** (Tensor): Shape :math:`(\text{batch}, C, C)` or
              :math:`(\text{batch}, K, C, C)`.
            - **grad_duration_bias** (Tensor): Shape :math:`(\text{batch}, K, C)`.
            - **grad_proj_start** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
              if boundaries provided.
            - **grad_proj_end** (Tensor or None): Shape :math:`(\text{batch}, T, C)`
              if boundaries provided.
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    effective_interval = max(checkpoint_interval, K)

    # Initialize gradient accumulators
    # Note: grad_transition and grad_duration_bias are per-batch to allow proper
    # weighting by grad_output in the autograd wrapper. The caller (autograd.py)
    # will apply einsum to compute: grad = Σ_b[grad_output[b] × grad_per_batch[b]]
    grad_cum_scores = torch.zeros_like(cum_scores)
    if transition.ndim == 2:
        grad_transition = torch.zeros(batch, C, C, device=device, dtype=dtype)
    else:
        grad_transition = torch.zeros(batch, K, C, C, device=device, dtype=dtype)
    grad_duration_bias = torch.zeros(batch, K, C, device=device, dtype=dtype)
    grad_proj_start = torch.zeros_like(proj_start) if proj_start is not None else None
    grad_proj_end = torch.zeros_like(proj_end) if proj_end is not None else None

    # Segment buffer for alpha values
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # Beta ring buffer
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    # After the forward fix, final alpha is captured at t=lengths, so beta
    # should be initialized at position lengths (not lengths-1)
    for b in range(batch):
        final_ring_idx = lengths[b].item() % K
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

            for k in range(1, max(k_eff + 1, 2)):  # max ensures K=1 processes duration 1
                start = t - k
                ring_idx = start % K
                alpha_prev = alpha_ring[:, ring_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, start, k, proj_start, proj_end
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
            local_t = t - seg_start
            alpha_t = alpha_segment[:, local_t, :]

            # Active if we can start a segment at position t
            # (need at least one position for segment to cover)
            active_mask = t < lengths
            if not active_mask.any():
                continue

            # Maximum duration: segments can end at position lengths (using cum_scores[:, lengths, :])
            max_k = min(K - 1, T - t)
            new_beta_scores = []

            for k in range(1, max(max_k + 1, 2)):  # max ensures K=1 processes duration 1
                end_pos = t + k
                # Include segments where end_pos == lengths (covering positions t to lengths-1)
                valid_mask = (end_pos <= lengths) & active_mask

                if not valid_mask.any():
                    continue

                ring_k_idx = end_pos % K
                beta_next = beta_ring[:, ring_k_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, t, k, proj_start, proj_end
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
                marginal_sum_dest = marginal.sum(dim=-1)  # (batch, C_dest)

                # grad_cum_scores[end_pos, c_dest] += sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, end_pos, :] += marginal_sum_dest
                # grad_cum_scores[t, c_dest] -= sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, t, :] -= marginal_sum_dest

                # grad_transition: per-batch accumulation (don't sum over batch)
                # marginal is (batch, C_dest, C_src), transpose to (batch, C_src, C_dest)
                # For duration-dependent transitions (K, C, C), index by k
                if transition.ndim == 2:
                    grad_transition += marginal.transpose(-1, -2)  # (batch, C_src, C_dest)
                else:
                    grad_transition[:, k] += marginal.transpose(
                        -1, -2
                    )  # (batch, C_src, C_dest) at index k

                # grad_duration_bias: per-batch accumulation
                # marginal.sum(dim=-1) sums over C_src -> (batch, C_dest)
                # Clamp k to valid index range (for K=1 case where k=1 but K-1=0)
                dur_idx = min(k, K - 1)
                grad_duration_bias[:, dur_idx, :] += marginal.sum(dim=-1)  # (batch, C_dest)

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
