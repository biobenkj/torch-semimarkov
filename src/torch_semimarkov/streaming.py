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


NEG_INF = -1e9


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
    r"""semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K, semiring='log', proj_start=None, proj_end=None) -> Tensor

    Compute Semi-CRF partition function with Golden Rule streaming.

    This is the main entry point for the streaming API. Edge potentials are
    computed on-the-fly from cumulative scores, eliminating the need for the
    full :math:`(\text{batch}, T-1, K, C, C)` edge tensor.

    Memory: :math:`O(KC)` ring buffer, independent of sequence length :math:`T`.
    Compute: :math:`O(T \times K \times C^2)` same as standard Semi-CRF.

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
        >>> # Streaming forward
        >>> partition = semi_crf_streaming_forward(
        ...     cum_scores, transition, duration_bias, lengths, K
        ... )
        >>> partition.shape
        torch.Size([2])

    See Also:
        :class:`~torch_semimarkov.SemiMarkov`: Pre-computed edge tensor API
        :func:`compute_edge_block_golden_rule`: On-the-fly edge computation helper
    """
    return SemiCRFStreaming.apply(
        cum_scores, transition, duration_bias, lengths, K, semiring,
        proj_start, proj_end,
    )
