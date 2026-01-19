r"""PyTorch reference implementation for Semi-CRF backward pass.

This module provides explicit backward pass computation for the Semi-Markov CRF
log-partition function. The backward computes gradients as marginal probabilities
using the forward-backward algorithm.

Mathematical Background
-----------------------

Forward (alpha):
    alpha[t, c] = logsumexp over k in 1..K, c' in 1..C of:
        alpha[t-k, c'] + edge[t-k, k, c', c]

    where alpha[t, c] is the log-sum of all paths ending at position t with label c.

Backward (beta):
    beta[t, c] = logsumexp over k in 1..K, c' in 1..C of:
        edge[t, k, c, c'] + beta[t+k, c']

    where beta[t, c] is the log-sum of all paths starting from position t with label c.

Gradient (marginal probability):
    d(log Z) / d(edge[t, k, c', c]) = P(segment (t, k, c', c) is used)
                                    = exp(alpha[t, c'] + edge[t, k, c', c] + beta[t+k, c] - log_Z)

This module provides a reference implementation for testing/verification.
For production use with long sequences, use the checkpointed implementations
in checkpointed.py which have much better memory characteristics.
"""

import torch
from typing import Tuple, Optional


NEG_INF = -1e9

# Triton availability check (used by checkpointed.py)
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    p = 1
    while p < n:
        p *= 2
    return p


def semi_crf_forward_with_alpha(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Forward pass that returns both partition and all alpha values.

    This is a modified version of the forward pass that stores all intermediate
    alpha values for use in the backward pass.

    Edge tensor convention: edge[b, t, k, c_dest, c_src] is the score for a
    segment starting at position t, with duration k, transitioning FROM c_src
    TO c_dest. Note: destination label is indexed BEFORE source label.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        semiring: Either "log" (logsumexp) or "max" (Viterbi).

    Returns:
        partition: Log partition function of shape (batch,).
        alpha: Forward values of shape (batch, T, C) where alpha[b, t, c] is
            the log-sum of all paths ending at position t with label c.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device
    dtype = edge.dtype

    # Store all alpha values: (batch, T, C)
    alpha = torch.full((batch, T, C), NEG_INF, device=device, dtype=dtype)
    alpha[:, 0, :] = 0.0  # Initial state: all labels equally likely

    # Main forward loop
    for t in range(1, T):
        # Number of valid durations at this position
        k_eff = min(K - 1, t)

        # Accumulate contributions from all valid durations
        scores_all = []

        for k in range(1, k_eff + 1):
            start = t - k

            # alpha[t, c_dest] = logsumexp over c_src of: alpha[start, c_src] + edge[start, k, c_dest, c_src]
            # alpha_prev: (batch, C) indexed by c_src
            alpha_prev = alpha[:, start, :]  # (batch, C_src)

            # edge_k: (batch, C_dest, C_src)
            edge_k = edge[:, start, k, :, :]  # (batch, C_dest, C_src)

            # We want: alpha_prev[c_src] + edge_k[c_dest, c_src] for all c_dest, c_src
            # Then logsumexp over c_src to get result indexed by c_dest
            # alpha_prev.unsqueeze(-2): (batch, 1, C_src) - broadcasts over c_dest
            # scores: (batch, C_dest, C_src)
            scores = alpha_prev.unsqueeze(-2) + edge_k

            scores_all.append(scores)

        # Stack: (batch, k_eff, C_dest, C_src)
        scores_stacked = torch.stack(scores_all, dim=1)

        if semiring == "log":
            # logsumexp over (k, c_src) dimensions to get (batch, C_dest)
            # First logsumexp over c_src (dim=-1), then over k (dim=1)
            scores_over_src = torch.logsumexp(scores_stacked, dim=-1)  # (batch, k_eff, C_dest)
            alpha_t = torch.logsumexp(scores_over_src, dim=1)  # (batch, C_dest)
        else:  # max
            scores_over_src = torch.max(scores_stacked, dim=-1)[0]
            alpha_t = torch.max(scores_over_src, dim=1)[0]

        alpha[:, t, :] = alpha_t

    # Extract final alpha for each sequence based on length
    # final_alpha[b] = alpha[b, lengths[b]-1, :]
    batch_idx = torch.arange(batch, device=device)
    final_positions = lengths - 1
    final_alpha = alpha[batch_idx, final_positions, :]  # (batch, C)

    # Compute partition
    if semiring == "log":
        partition = torch.logsumexp(final_alpha, dim=-1)
    else:
        partition = torch.max(final_alpha, dim=-1)[0]

    return partition, alpha


def semi_crf_backward_beta(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute backward beta values.

    beta[t, c_src] = logsumexp over k in 1..K, c_dest in 1..C of:
        edge[t, k, c_dest, c_src] + beta[t+k, c_dest]

    Edge tensor convention: edge[b, t, k, c_dest, c_src] where c_dest is the
    destination label and c_src is the source label.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        semiring: Either "log" or "max".

    Returns:
        beta: Backward values of shape (batch, T, C) where beta[b, t, c] is
            the log-sum of all paths starting from position t with label c.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device
    dtype = edge.dtype

    # Store all beta values: (batch, T, C)
    beta = torch.full((batch, T, C), NEG_INF, device=device, dtype=dtype)

    # Initialize: beta[T-1, c] = 0 for each sequence's final position
    # For variable lengths, we need to set beta[lengths[b]-1, :] = 0
    batch_idx = torch.arange(batch, device=device)
    final_positions = lengths - 1
    beta[batch_idx, final_positions, :] = 0.0

    # Backward loop: t from T-2 down to 0
    # We need to handle variable lengths: only update if t < lengths[b] - 1
    for t in range(T - 2, -1, -1):
        # Mask for which batch elements are active at this position
        # Active if t < lengths - 1 (i.e., there are positions after t)
        active_mask = t < (lengths - 1)  # (batch,)

        if not active_mask.any():
            continue

        # Maximum duration from position t
        max_k = min(K - 1, T - 1 - t)

        scores_all = []

        for k in range(1, max_k + 1):
            end_pos = t + k

            # Only valid if end_pos <= lengths - 1
            valid_mask = (end_pos <= lengths - 1) & active_mask  # (batch,)

            # beta[t, c_src] needs: edge[t, k, c_dest, c_src] + beta[t+k, c_dest]
            # edge_k: (batch, C_dest, C_src)
            edge_k = edge[:, t, k, :, :]  # (batch, C_dest, C_src)

            # beta_next: (batch, C_dest)
            beta_next = beta[:, end_pos, :]  # (batch, C_dest)

            # We want: edge_k[c_dest, c_src] + beta_next[c_dest] for all c_dest, c_src
            # Then logsumexp over c_dest to get result indexed by c_src
            # beta_next.unsqueeze(-1): (batch, C_dest, 1) - broadcasts over c_src
            # scores: (batch, C_dest, C_src)
            scores = edge_k + beta_next.unsqueeze(-1)

            # Mask invalid entries
            scores = torch.where(
                valid_mask.view(batch, 1, 1),
                scores,
                torch.full_like(scores, NEG_INF),
            )

            scores_all.append(scores)

        if not scores_all:
            continue

        # Stack: (batch, max_k, C_dest, C_src)
        scores_stacked = torch.stack(scores_all, dim=1)

        if semiring == "log":
            # logsumexp over (k, c_dest) dimensions to get (batch, C_src)
            # First logsumexp over c_dest (dim=-2), then over k (dim=1)
            scores_over_dest = torch.logsumexp(scores_stacked, dim=-2)  # (batch, max_k, C_src)
            beta_t = torch.logsumexp(scores_over_dest, dim=1)  # (batch, C_src)
        else:  # max
            scores_over_dest = torch.max(scores_stacked, dim=-2)[0]
            beta_t = torch.max(scores_over_dest, dim=1)[0]

        # Only update active positions
        beta[:, t, :] = torch.where(
            active_mask.view(batch, 1), beta_t, beta[:, t, :]
        )

    return beta


def semi_crf_compute_marginals(
    edge: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    log_Z: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    r"""Compute marginal probabilities (gradients) for each edge.

    The marginal probability of using segment (t, k, c_dest, c_src) is:
        P = exp(alpha[t, c_src] + edge[t, k, c_dest, c_src] + beta[t+k, c_dest] - log_Z)

    Edge tensor convention: edge[b, t, k, c_dest, c_src] where:
    - c_src is the source label (at position t)
    - c_dest is the destination label (at position t+k)

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        alpha: Forward values of shape (batch, T, C).
        beta: Backward values of shape (batch, T, C).
        log_Z: Log partition values of shape (batch,).
        lengths: Sequence lengths of shape (batch,).

    Returns:
        marginals: Marginal probabilities of shape (batch, T-1, K, C, C).
            Same shape as edge tensor.
    """
    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device

    # Initialize marginals to 0
    marginals = torch.zeros_like(edge)

    for t in range(T - 1):
        for k in range(1, K):
            end_pos = t + k

            # Only valid if end_pos < T and within sequence length
            # Valid if end_pos <= lengths - 1
            valid_mask = end_pos <= (lengths - 1)  # (batch,)

            if not valid_mask.any():
                continue

            # alpha[t, c_src]: (batch, C_src)
            alpha_t = alpha[:, t, :]  # (batch, C_src)

            # beta[t+k, c_dest]: (batch, C_dest)
            beta_end = beta[:, end_pos, :]  # (batch, C_dest)

            # edge[t, k, c_dest, c_src]: (batch, C_dest, C_src)
            edge_tk = edge[:, t, k, :, :]

            # log_marginal[c_dest, c_src] = alpha[t, c_src] + edge[t, k, c_dest, c_src] + beta[t+k, c_dest] - log_Z
            # Shape: (batch, C_dest, C_src)
            log_marginal = (
                alpha_t.unsqueeze(-2)  # (batch, 1, C_src)
                + edge_tk  # (batch, C_dest, C_src)
                + beta_end.unsqueeze(-1)  # (batch, C_dest, 1)
                - log_Z.view(batch, 1, 1)  # (batch, 1, 1)
            )

            # Convert to probability
            marginal = torch.exp(log_marginal)

            # Mask invalid entries
            marginal = torch.where(
                valid_mask.view(batch, 1, 1), marginal, torch.zeros_like(marginal)
            )

            marginals[:, t, k, :, :] = marginal

    return marginals


def semi_crf_backward_pytorch(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Complete backward pass: compute partition and gradients.

    This is the main entry point for the PyTorch reference backward.
    It computes:
    1. Forward pass to get alpha values and partition
    2. Backward pass to get beta values
    3. Marginals (gradients) from alpha, beta, and edge

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
        grad_edge: Gradient w.r.t. edge of shape (batch, T-1, K, C, C).
    """
    # Forward pass
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring)

    # Backward pass
    beta = semi_crf_backward_beta(edge, lengths, semiring)

    # Compute marginals (gradients)
    grad_edge = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    return partition, grad_edge


class SemiCRFBackward(torch.autograd.Function):
    r"""Autograd function using explicit forward-backward algorithm.

    This custom autograd function computes gradients using the explicit
    forward-backward algorithm instead of relying on autograd through
    the forward computation.

    This serves as:
    1. Reference implementation for verifying correctness
    2. Baseline for comparing against Triton checkpointed backward
    """

    @staticmethod
    def forward(
        ctx,
        edge: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        # Compute forward pass and store values for backward
        partition, alpha = semi_crf_forward_with_alpha(
            edge.detach(), lengths, semiring
        )

        # Save for backward
        ctx.save_for_backward(edge, lengths, alpha, partition)
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None]:
        edge, lengths, alpha, partition = ctx.saved_tensors
        semiring = ctx.semiring

        # Compute beta
        beta = semi_crf_backward_beta(edge, lengths, semiring)

        # Compute marginals
        marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

        # Scale by upstream gradient
        grad_edge = marginals * grad_output.view(-1, 1, 1, 1, 1)

        return grad_edge, None, None


def semi_crf_forward_backward(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition using explicit forward-backward for gradients.

    This function uses the explicit forward-backward algorithm for computing
    gradients, rather than relying on autograd through the forward pass.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFBackward.apply(edge, lengths, semiring)
