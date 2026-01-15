r"""PyTorch reference implementation for Semi-CRF backward pass.

This module provides explicit backward pass computation for the Semi-Markov CRF
log-partition function. The backward computes gradients as marginal probabilities
using the forward-backward algorithm.

Mathematical Background
-----------------------

Forward (α):
    α[t, c] = logsumexp over k in 1..K, c' in 1..C of:
        α[t-k, c'] + edge[t-k, k, c', c]

    where α[t, c] is the log-sum of all paths ending at position t with label c.

Backward (β):
    β[t, c] = logsumexp over k in 1..K, c' in 1..C of:
        edge[t, k, c, c'] + β[t+k, c']

    where β[t, c] is the log-sum of all paths starting from position t with label c.

Gradient (marginal probability):
    ∂(log Z) / ∂(edge[t, k, c', c]) = P(segment (t, k, c', c) is used)
                                    = exp(α[t, c'] + edge[t, k, c', c] + β[t+k, c] - log_Z)

This reference implementation uses full storage (Option C from roadmap) for clarity.
The Triton kernel will use recomputation (Option A) to maintain O(KC) memory.
"""

import torch
from typing import Tuple, Optional


NEG_INF = -1e9

# Triton is optional - kernel only available when installed and on GPU
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def semi_crf_forward_with_alpha(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Forward pass that returns both partition and all α values.

    This is a modified version of the forward pass that stores all intermediate
    α values for use in the backward pass.

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

            # α[t, c_dest] = logsumexp over c_src of: α[start, c_src] + edge[start, k, c_dest, c_src]
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
    r"""Compute backward β values.

    β[t, c_src] = logsumexp over k in 1..K, c_dest in 1..C of:
        edge[t, k, c_dest, c_src] + β[t+k, c_dest]

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

    # Initialize: β[T-1, c] = 0 for each sequence's final position
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

            # β[t, c_src] needs: edge[t, k, c_dest, c_src] + β[t+k, c_dest]
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
        P = exp(α[t, c_src] + edge[t, k, c_dest, c_src] + β[t+k, c_dest] - log_Z)

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
    dtype = edge.dtype

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

            # α[t, c_src]: (batch, C_src)
            alpha_t = alpha[:, t, :]  # (batch, C_src)

            # β[t+k, c_dest]: (batch, C_dest)
            beta_end = beta[:, end_pos, :]  # (batch, C_dest)

            # edge[t, k, c_dest, c_src]: (batch, C_dest, C_src)
            edge_tk = edge[:, t, k, :, :]

            # log_marginal[c_dest, c_src] = α[t, c_src] + edge[t, k, c_dest, c_src] + β[t+k, c_dest] - log_Z
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
    1. Forward pass to get α values and partition
    2. Backward pass to get β values
    3. Marginals (gradients) from α, β, and edge

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
    2. Baseline for comparing against future Triton backward kernel
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


# =============================================================================
# Triton Kernel (GPU only, optional) - Non-checkpointed backward
# =============================================================================

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


if HAS_TRITON:

    @triton.jit
    def semi_crf_backward_kernel(
        # Inputs
        edge_ptr,  # (batch, T-1, K, C, C) - edge potentials
        alpha_ptr,  # (batch, T, C_PAD) - pre-computed forward values (padded)
        log_Z_ptr,  # (batch,) - partition values
        lengths_ptr,  # (batch,) - sequence lengths
        # Outputs
        grad_edge_ptr,  # (batch, T-1, K, C, C) - gradient output
        beta_ring_ptr,  # (batch, K, C_PAD) - ring buffer for beta
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        USE_FP64: tl.constexpr,  # whether to use float64
        # Strides for edge tensor (batch, T-1, K, C, C)
        stride_eb,
        stride_et,
        stride_ek,
        stride_ec1,
        stride_ec2,
        # Strides for alpha tensor (batch, T, C_PAD)
        stride_ab,
        stride_at,
        stride_ac,
        # Strides for grad_edge tensor (same as edge)
        stride_gb,
        stride_gt,
        stride_gk,
        stride_gc1,
        stride_gc2,
        # Strides for beta ring buffer (batch, K, C_PAD)
        stride_rb,
        stride_rk,
        stride_rc,
    ):
        """
        Fused Semi-Markov CRF backward pass with gradient computation.

        This kernel computes:
        1. β values using a ring buffer (going backward from T-1 to 0)
        2. Gradient contributions at each position using α, edge, β, and log_Z

        Each program handles one batch element.

        Ring buffer layout: ring[batch, k, c_pad]
        - Stores β[t+1], β[t+2], ..., β[t+K] for current position t
        - We rotate through the buffer as we iterate backward

        Edge tensor convention: edge[b, t, k, c_dest, c_src]
        - c_src: source label at position t
        - c_dest: destination label at position t+k
        """
        NEG_INF: tl.constexpr = -1e9

        # Select dtype based on USE_FP64 flag
        if USE_FP64:
            DTYPE = tl.float64
        else:
            DTYPE = tl.float32

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        # 2D indices for [C_PAD, C_PAD] blocks
        c_dest = tl.arange(0, C_PAD)[:, None]  # [C_PAD, 1] - destination labels
        c_src = tl.arange(0, C_PAD)[None, :]  # [1, C_PAD] - source labels
        c_mask_2d = (c_dest < C) & (c_src < C)  # [C_PAD, C_PAD]

        # Load sequence length and log_Z
        seq_len = tl.load(lengths_ptr + batch_idx)
        log_Z = tl.load(log_Z_ptr + batch_idx)

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        alpha_base = alpha_ptr + batch_idx * stride_ab
        grad_base = grad_edge_ptr + batch_idx * stride_gb
        ring_base = beta_ring_ptr + batch_idx * stride_rb

        # Initialize beta ring buffer
        # β[seq_len-1] = 0 (at final position)
        # All other slots start as NEG_INF
        for k_init in tl.static_range(0, K):
            ring_offset = ring_base + k_init * stride_rk + c_idx * stride_rc
            # Only the slot for the final position gets 0, rest get NEG_INF
            # We'll handle this dynamically based on seq_len
            tl.store(ring_offset, tl.full([C_PAD], NEG_INF, dtype=DTYPE), mask=c_mask)

        # Set β[final_pos] = 0 in the ring buffer
        final_ring_slot = (seq_len - 1) % K
        final_ring_offset = ring_base + final_ring_slot * stride_rk + c_idx * stride_rc
        tl.store(final_ring_offset, tl.where(c_mask, tl.zeros([C_PAD], dtype=DTYPE), tl.full([C_PAD], NEG_INF, dtype=DTYPE)), mask=c_mask)

        # Main backward loop: t from seq_len-2 down to 0
        # We iterate backward through positions
        for t_offset in tl.range(0, T - 1):
            # Compute actual position (going backward)
            t = (seq_len - 2) - t_offset

            # Skip if t < 0 (for short sequences)
            active = t >= 0

            # Accumulate new_beta = logsumexp over (k, c_dest) - shape [C_PAD]
            # This is β[t, c_src]
            new_beta = tl.full([C_PAD], NEG_INF, dtype=DTYPE)

            # Load alpha[t, :] for gradient computation
            alpha_t = tl.load(
                alpha_base + t * stride_at + c_idx * stride_ac,
                mask=active & c_mask,
                other=NEG_INF,
            )  # [C_PAD] indexed by c_src

            # Loop over durations k = 1, 2, ..., K-1
            for k in tl.range(1, K):
                end_pos = t + k

                # Valid if end_pos <= seq_len - 1
                k_valid = active & (k <= K - 1) & (end_pos < seq_len)

                # Load β[t+k] from ring buffer
                # Ring index for position end_pos
                ring_k_idx = end_pos % K
                beta_end = tl.load(
                    ring_base + ring_k_idx * stride_rk + c_idx * stride_rc,
                    mask=k_valid & c_mask,
                    other=NEG_INF,
                )  # [C_PAD] indexed by c_dest

                # Load edge[t, k, :, :] - shape [C_PAD, C_PAD] = [c_dest, c_src]
                edge_offset_2d = (
                    edge_base
                    + t * stride_et
                    + k * stride_ek
                    + c_dest * stride_ec1
                    + c_src * stride_ec2
                )
                edge_block = tl.load(
                    edge_offset_2d, mask=k_valid & c_mask_2d, other=NEG_INF
                )  # [C_PAD, C_PAD]

                # === Compute β contribution ===
                # β[t, c_src] += logsumexp over c_dest of: edge[t,k,c_dest,c_src] + β[t+k,c_dest]
                # scores[c_dest, c_src] = edge[c_dest, c_src] + β[c_dest]
                scores_for_beta = edge_block + beta_end[:, None]  # [C_PAD, C_PAD]
                scores_for_beta = tl.where(c_mask_2d, scores_for_beta, NEG_INF)

                # logsumexp over c_dest (axis=0) to get [C_PAD] indexed by c_src
                max_scores = tl.max(scores_for_beta, axis=0)  # [C_PAD]
                beta_contrib = max_scores + tl.log(
                    tl.sum(tl.exp(scores_for_beta - max_scores[None, :]), axis=0)
                )  # [C_PAD]
                beta_contrib = tl.where(k_valid & c_mask, beta_contrib, NEG_INF)

                # Accumulate into new_beta via logsumexp
                max_nb = tl.maximum(new_beta, beta_contrib)
                new_beta = max_nb + tl.log(
                    tl.exp(new_beta - max_nb) + tl.exp(beta_contrib - max_nb)
                )

                # === Compute gradient contribution ===
                # grad[t, k, c_dest, c_src] = exp(α[t, c_src] + edge[t,k,c_dest,c_src] + β[t+k, c_dest] - log_Z)
                # log_marginal[c_dest, c_src] = α[c_src] + edge[c_dest, c_src] + β[c_dest] - log_Z
                log_marginal = (
                    alpha_t[None, :]  # [1, C_PAD] - broadcasts over c_dest
                    + edge_block  # [C_PAD, C_PAD]
                    + beta_end[:, None]  # [C_PAD, 1] - broadcasts over c_src
                    - log_Z
                )
                marginal = tl.exp(log_marginal)
                marginal = tl.where(k_valid & c_mask_2d, marginal, 0.0)

                # Store gradient
                grad_offset_2d = (
                    grad_base
                    + t * stride_gt
                    + k * stride_gk
                    + c_dest * stride_gc1
                    + c_src * stride_gc2
                )
                tl.store(grad_offset_2d, marginal, mask=k_valid & c_mask_2d)

            # Store new_beta to ring buffer at position t
            ring_t_idx = t % K
            new_beta_masked = tl.where(active & c_mask, new_beta, NEG_INF)
            tl.store(
                ring_base + ring_t_idx * stride_rk + c_idx * stride_rc,
                new_beta_masked,
                mask=active & c_mask,
            )

    def launch_triton_backward_kernel(
        edge: torch.Tensor,
        alpha: torch.Tensor,
        log_Z: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        r"""Launch the Triton backward kernel.

        Args:
            edge: Log potentials of shape (batch, T-1, K, C, C).
            alpha: Pre-computed forward values of shape (batch, T, C).
            log_Z: Log partition values of shape (batch,).
            lengths: Sequence lengths of shape (batch,).

        Returns:
            grad_edge: Gradient w.r.t. edge of shape (batch, T-1, K, C, C).
        """
        batch, T_minus_1, K, C, _ = edge.shape
        T = T_minus_1 + 1

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Ensure inputs are contiguous
        edge = edge.contiguous()
        lengths = lengths.contiguous()

        # Pad alpha to C_PAD if needed
        if C_PAD != C:
            alpha_padded = torch.full(
                (batch, T, C_PAD), NEG_INF, device=edge.device, dtype=edge.dtype
            )
            alpha_padded[:, :, :C] = alpha
        else:
            alpha_padded = alpha.contiguous()

        # Allocate ring buffer for beta
        beta_ring = torch.empty(
            (batch, K, C_PAD), device=edge.device, dtype=edge.dtype
        )

        # Allocate output gradient tensor
        grad_edge = torch.zeros_like(edge)

        # Get strides
        stride_eb, stride_et, stride_ek, stride_ec1, stride_ec2 = edge.stride()
        stride_ab, stride_at, stride_ac = alpha_padded.stride()
        stride_gb, stride_gt, stride_gk, stride_gc1, stride_gc2 = grad_edge.stride()
        stride_rb, stride_rk, stride_rc = beta_ring.stride()

        # Determine if using float64
        USE_FP64 = edge.dtype == torch.float64

        # Launch kernel
        grid = (batch,)
        semi_crf_backward_kernel[grid](
            edge,
            alpha_padded,
            log_Z,
            lengths,
            grad_edge,
            beta_ring,
            batch,
            T,
            K,
            C,
            C_PAD,
            USE_FP64,
            stride_eb,
            stride_et,
            stride_ek,
            stride_ec1,
            stride_ec2,
            stride_ab,
            stride_at,
            stride_ac,
            stride_gb,
            stride_gt,
            stride_gk,
            stride_gc1,
            stride_gc2,
            stride_rb,
            stride_rk,
            stride_rc,
        )

        return grad_edge


class SemiCRFTritonBackward(torch.autograd.Function):
    r"""Autograd function using Triton kernel for backward pass.

    This uses the custom Triton kernel for computing gradients, providing
    GPU-accelerated backward pass computation.

    Forward: Uses the existing Triton forward kernel (or PyTorch fallback)
    Backward: Uses the custom Triton backward kernel
    """

    @staticmethod
    def forward(
        ctx,
        edge: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        # Compute forward pass with alpha storage
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

        # Use Triton kernel if available and on CUDA
        if HAS_TRITON and edge.is_cuda and semiring == "log":
            marginals = launch_triton_backward_kernel(edge, alpha, partition, lengths)
        else:
            # Fall back to PyTorch implementation
            beta = semi_crf_backward_beta(edge, lengths, semiring)
            marginals = semi_crf_compute_marginals(
                edge, alpha, beta, partition, lengths
            )

        # Scale by upstream gradient
        grad_edge = marginals * grad_output.view(-1, 1, 1, 1, 1)

        return grad_edge, None, None


def semi_crf_triton_backward(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition using Triton kernel for backward pass.

    This function uses the custom Triton backward kernel for computing
    gradients when on GPU, with PyTorch fallback for CPU.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        semiring: Either "log" or "max". Note: Triton kernel currently
            only supports "log" semiring.

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFTritonBackward.apply(edge, lengths, semiring)
