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
# Triton Kernels (GPU only, optional)
# =============================================================================

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

        # === Load ring buffer state from checkpoint ===
        # We need to track K alpha values for the ring buffer
        # Load initial state from checkpoint

        # For each position in segment, we'll compute alpha and store it
        # We maintain a ring buffer of the last K alpha values

        # === FORWARD PASS: Recompute alpha values within segment ===
        # This follows the same logic as triton_scan.py's working forward kernel.
        #
        # Key insight: we read alpha[t-k] from:
        # - Checkpoint ring buffer if t-k < seg_start (positions computed before this segment)
        # - Segment buffer if t-k >= seg_start (positions we just computed in this kernel)
        #
        # The checkpoint stores the ring buffer state AT seg_start, containing
        # alpha values for positions [seg_start - K + 1, seg_start].

        # Process all positions in segment
        for t_local in tl.range(0, checkpoint_interval):
            t = seg_start + t_local
            active = t < seg_end

            if active:
                if t == seg_start:
                    # First position: alpha[seg_start] is in the checkpoint
                    ring_slot = seg_start % K
                    alpha_t = tl.load(
                        ckpt_base + ckpt_idx * stride_rci + ring_slot * stride_rck + c_idx * stride_rcc,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                else:
                    # Compute alpha[t] = logsumexp over k of contribution from alpha[t-k]
                    # Same algorithm as triton_scan.py
                    alpha_t = tl.full([C_PAD], NEG_INF, dtype=DTYPE)

                    for k in tl.range(1, K):
                        # k is valid if k <= t and k <= K-1 (same as triton_scan.py)
                        k_valid = (k <= t) & (k <= K - 1)
                        start_pos = t - k

                        # Compute safe index for loads (even though we mask, avoid UB)
                        start_pos_safe = tl.maximum(start_pos, 0)
                        ring_slot = start_pos_safe % K

                        # Determine source: checkpoint (before segment) or segment buffer
                        from_checkpoint = k_valid & (start_pos < seg_start)
                        from_segment = k_valid & (start_pos >= seg_start)

                        # Load alpha[start_pos] from appropriate source
                        # Use separate loads with proper masks (Triton idiom)
                        alpha_ckpt = tl.load(
                            ckpt_base + ckpt_idx * stride_rci + ring_slot * stride_rck + c_idx * stride_rcc,
                            mask=from_checkpoint & c_mask,
                            other=NEG_INF,
                        )
                        local_idx = tl.maximum(start_pos - seg_start, 0)
                        alpha_seg = tl.load(
                            alpha_seg_base + local_idx * stride_ast + c_idx * stride_asc,
                            mask=from_segment & c_mask,
                            other=NEG_INF,
                        )
                        # Combine: if from_checkpoint, use alpha_ckpt; else use alpha_seg
                        # When both are false, both have NEG_INF which is correct
                        alpha_prev = tl.where(from_checkpoint, alpha_ckpt, alpha_seg)

                        # Load edge[start_pos, k, :, :] - same pattern as triton_scan.py
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

                        # Compute scores - same as triton_scan.py
                        scores = alpha_prev[None, :] + edge_block  # [C_PAD, C_PAD]
                        scores = tl.where(c_mask_2d, scores, NEG_INF)

                        # Logsumexp over source labels (axis=1) - same as triton_scan.py
                        max_scores = tl.max(scores, axis=1)
                        score_for_k = max_scores + tl.log(
                            tl.sum(tl.exp(scores - max_scores[:, None]), axis=1)
                        )
                        score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                        # Accumulate into alpha_t - same as triton_scan.py
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

        # Process positions backward: from seg_end-1 down to seg_start
        # At each position t, we:
        # 1. Load alpha[t] from segment buffer
        # 2. Compute gradient using alpha, edge, beta, log_Z
        # 3. Compute new beta[t] and store to ring buffer

        # Use tl.range instead of tl.static_range to avoid compilation issues
        for t_offset in tl.range(0, checkpoint_interval):
            t = (seg_end - 1) - t_offset
            t_active = (t >= seg_start) & (t < seq_len - 1)

            if t_active:
                # Load alpha[t] from segment buffer
                t_local = t - seg_start
                t_local_safe = tl.maximum(t_local, 0)  # Ensure non-negative
                alpha_t = tl.load(
                    alpha_seg_base + t_local_safe * stride_ast + c_idx * stride_asc,
                    mask=c_mask,
                    other=NEG_INF,
                )

                # Compute new beta[t] and gradients
                new_beta = tl.full([C_PAD], NEG_INF, dtype=DTYPE)
                k_max = tl.minimum(K - 1, seq_len - 1 - t)

                # Safe t for edge indexing (ensure non-negative and in bounds)
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
                    # grad[t, k, c_dest, c_src] = exp(alpha[c_src] + edge[c_dest, c_src] + beta[c_dest] - log_Z)
                    log_marginal = (
                        alpha_t[None, :]  # [1, C_PAD] broadcasts over c_dest
                        + edge_block  # [C_PAD, C_PAD]
                        + beta_end[:, None]  # [C_PAD, 1] broadcasts over c_src
                        - log_Z
                    )
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
                    # beta[t, c_src] += logsumexp over c_dest of: edge[c_dest, c_src] + beta[c_dest]
                    scores_for_beta = edge_block + beta_end[:, None]  # [C_PAD, C_PAD]
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
        beta_ring = torch.full(
            (batch, K, C_PAD), NEG_INF, device=edge.device, dtype=edge.dtype
        )

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


# =============================================================================
# Phase 3: Checkpointed Forward-Backward (Memory Optimized)
# =============================================================================


def _compute_checkpoint_interval(T: int, K: int = 1) -> int:
    """Compute optimal checkpoint interval (approximately √T, but at least K).

    The checkpoint interval must be >= K to ensure all required α values
    for recomputation are available within the current segment.

    Args:
        T: Sequence length.
        K: Maximum duration (for ensuring interval >= K).

    Returns:
        Optimal checkpoint interval.
    """
    import math
    return max(K, int(math.sqrt(T)))


def semi_crf_forward_with_checkpoints(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Forward pass that saves α at checkpoint intervals.

    Instead of storing all T α values, we store α at positions:
    0, checkpoint_interval, 2*checkpoint_interval, ...

    This reduces memory from O(T·C) to O(√T·C) when checkpoint_interval = √T.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √T.
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
        checkpoints: Checkpointed α values of shape (batch, num_checkpoints, C).
        checkpoint_interval: The actual interval used.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_minus_1, K, C, _ = edge.shape
    T = T_minus_1 + 1
    device = edge.device
    dtype = edge.dtype

    # Determine checkpoint interval (must be >= K for correctness)
    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(T, K)
    else:
        checkpoint_interval = max(checkpoint_interval, K)

    # Number of checkpoints needed
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

    # Allocate checkpoint storage: (batch, num_checkpoints, C)
    checkpoints = torch.full(
        (batch, num_checkpoints, C), NEG_INF, device=device, dtype=dtype
    )

    # Use ring buffer for forward pass (O(K·C) memory)
    alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
    alpha_ring[:, 0, :] = 0.0  # α[0] = 0

    # Store initial checkpoint at position 0
    checkpoints[:, 0, :] = 0.0

    # Track final alpha for each batch element (for variable lengths)
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)
    final_positions = lengths - 1

    # Initialize final_alpha for sequences of length 1
    len_1_mask = (lengths == 1)
    if len_1_mask.any():
        final_alpha[len_1_mask] = 0.0

    for t in range(1, T):
        # Which batch elements are still active at this position
        active_mask = t < lengths  # (batch,)

        k_eff = min(K - 1, t)
        scores_all = []

        for k in range(1, k_eff + 1):
            start = t - k
            ring_idx = start % K

            alpha_prev = alpha_ring[:, ring_idx, :]  # (batch, C_src)
            edge_k = edge[:, start, k, :, :]  # (batch, C_dest, C_src)
            scores = alpha_prev.unsqueeze(-2) + edge_k  # (batch, C_dest, C_src)
            scores_all.append(scores)

        scores_stacked = torch.stack(scores_all, dim=1)  # (batch, k_eff, C_dest, C_src)

        if semiring == "log":
            scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
            alpha_t = torch.logsumexp(scores_over_src, dim=1)
        else:
            scores_over_src = torch.max(scores_stacked, dim=-1)[0]
            alpha_t = torch.max(scores_over_src, dim=1)[0]

        # Only update ring buffer for active sequences
        ring_idx_t = t % K
        alpha_ring[:, ring_idx_t, :] = torch.where(
            active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
        )

        # Save checkpoint if at checkpoint position (only for active sequences)
        if t % checkpoint_interval == 0:
            ckpt_idx = t // checkpoint_interval
            if ckpt_idx < num_checkpoints:
                checkpoints[:, ckpt_idx, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, checkpoints[:, ckpt_idx, :]
                )

        # Save final alpha when we reach the final position for each sequence
        is_final = (t == final_positions)  # (batch,)
        if is_final.any():
            final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)

    if semiring == "log":
        partition = torch.logsumexp(final_alpha, dim=-1)
    else:
        partition = torch.max(final_alpha, dim=-1)[0]

    return partition, checkpoints, checkpoint_interval


def semi_crf_backward_from_checkpoints(
    edge: torch.Tensor,
    checkpoints: torch.Tensor,
    log_Z: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: int,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Backward pass using checkpointed α values.

    This recomputes α values within each segment from checkpoints,
    then computes β and gradients. Memory usage is O(interval·C + K·C).

    Algorithm:
    1. Process segments in REVERSE order
    2. For each segment:
       a. Recompute α forward from the beginning (position 0) to populate ring buffer
       b. Store α values for this segment in segment buffer
       c. Compute β backward and gradients (β ring buffer persists)

    Note: To correctly compute α[t], we need α[t-1], α[t-2], ..., α[t-K+1].
    These values may span multiple previous segments, so we recompute from
    position 0 to ensure correctness. This makes the algorithm O(T * num_segments)
    = O(T * √T) = O(T^1.5) instead of O(T), but guarantees correct gradients.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        checkpoints: Checkpointed α values of shape (batch, num_checkpoints, C).
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

    # Segment buffer for α values: (batch, interval + K, C)
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # β ring buffer: (batch, K, C)
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize β at final positions
    final_positions = lengths - 1
    for b in range(batch):
        final_ring_idx = final_positions[b].item() % K
        beta_ring[b, final_ring_idx, :] = 0.0

    # Process segments in REVERSE order
    num_checkpoints = checkpoints.shape[1]

    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = min((ckpt_idx + 1) * checkpoint_interval, T)

        # Clear segment buffer
        alpha_segment.fill_(NEG_INF)

        # === Phase 1: Recompute α forward from position 0 to seg_end ===
        # We need to recompute from the beginning to get correct α values
        # because α[t] depends on α[t-1], ..., α[t-K+1] which may span
        # multiple previous segments.

        # Initialize α ring buffer starting from position 0
        alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
        alpha_ring[:, 0, :] = 0.0  # α[0] = 0

        # Recompute α from position 1 to seg_end-1
        for t in range(1, seg_end):
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

                # Store in segment buffer if within current segment
                if t >= seg_start:
                    local_t = t - seg_start
                    alpha_segment[:, local_t, :] = torch.where(
                        active_mask.view(batch, 1), alpha_t, alpha_segment[:, local_t, :]
                    )

        # Store α[seg_start] = checkpoint at local position 0
        # (This handles the case where seg_start = 0)
        if seg_start == 0:
            alpha_segment[:, 0, :] = 0.0
        else:
            # α[seg_start] was already stored in the loop above
            pass

        # === Phase 2: Compute β backward and gradients for this segment ===

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


class SemiCRFCheckpointedBackward(torch.autograd.Function):
    r"""Autograd function using checkpointed forward-backward algorithm.

    This achieves O(√T·C + K·C) memory by:
    1. Saving α at every √T positions during forward
    2. Recomputing α within segments during backward

    Compute cost: O(T^1.5) for backward (recompute α from start for each segment)
    Memory: O(√T·C) instead of O(T·C)

    Note: The compute cost is higher than optimal (O(T)) because we recompute
    α from position 0 for each segment to ensure correctness. A more efficient
    O(T) implementation would require saving ring buffer state at each checkpoint.
    """

    @staticmethod
    def forward(
        ctx,
        edge: torch.Tensor,
        lengths: torch.Tensor,
        checkpoint_interval: Optional[int] = None,
        semiring: str = "log",
    ) -> torch.Tensor:
        # Compute forward pass with checkpointing
        partition, checkpoints, actual_interval = semi_crf_forward_with_checkpoints(
            edge.detach(), lengths, checkpoint_interval, semiring
        )

        # Save for backward
        ctx.save_for_backward(edge, lengths, checkpoints, partition)
        ctx.checkpoint_interval = actual_interval
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        edge, lengths, checkpoints, partition = ctx.saved_tensors
        checkpoint_interval = ctx.checkpoint_interval
        semiring = ctx.semiring

        # Compute gradients using checkpointed backward
        marginals = semi_crf_backward_from_checkpoints(
            edge, checkpoints, partition, lengths, checkpoint_interval, semiring
        )

        # Scale by upstream gradient
        grad_edge = marginals * grad_output.view(-1, 1, 1, 1, 1)

        return grad_edge, None, None, None


def semi_crf_checkpointed_backward(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition with memory-efficient checkpointed backward.

    This achieves O(√T·C + K·C) memory instead of O(T·C) by:
    1. Saving α at every √T positions during forward
    2. Recomputing α within segments during backward

    Trade-off: O(T^1.5) compute for ~√T memory reduction.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √T.
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFCheckpointedBackward.apply(
        edge, lengths, checkpoint_interval, semiring
    )


# =============================================================================
# Optimized Checkpointed Backward: O(T) compute, O(√T × K × C) memory
# =============================================================================


def semi_crf_forward_with_ring_checkpoints(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint_interval: Optional[int] = None,
    semiring: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Forward pass that saves ring buffer state at checkpoint intervals.

    This saves the entire ring buffer (K × C values) at each checkpoint,
    not just α[checkpoint_pos] (C values). This enables O(T) backward
    instead of O(T^1.5) at the cost of K× more checkpoint memory.

    Memory: O(√T × K × C) for checkpoints
    Compute: O(T) for forward

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √T.
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
    len_1_mask = (lengths == 1)
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
        is_final = (t == final_positions)
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

    This achieves O(√T × K × C) memory by saving ring buffer state at checkpoints,
    enabling O(T) backward compute instead of O(T^1.5).

    Memory: O(√T × K × C) for checkpoints + O(interval × C) working memory
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
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
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

    Trade-offs vs basic checkpointing:
    - Memory: O(√T × K × C) vs O(√T × C) — K× more checkpoint storage
    - Compute: O(T) vs O(T^1.5) — much faster for large T

    For typical K (4-16) and large T (1000+), this is a better trade-off.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √T.
        semiring: Either "log" or "max".

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFOptimizedCheckpointedBackward.apply(
        edge, lengths, checkpoint_interval, semiring
    )


# =============================================================================
# Phase 3 Triton: Checkpointed Backward with Triton Kernel
# =============================================================================


class SemiCRFTritonCheckpointedBackward(torch.autograd.Function):
    r"""Autograd function using Triton kernel for checkpointed backward.

    This combines:
    - Forward: PyTorch implementation saving ring buffer checkpoints
    - Backward: Triton kernel for GPU-accelerated checkpointed backward

    Memory: O(√T × K × C) for checkpoints
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
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
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
    - Forward: Ring buffer checkpointing (O(√T × K × C) memory)
    - Backward: Triton kernel for GPU-accelerated gradient computation (O(T) compute)

    Falls back to PyTorch implementation on CPU.

    Args:
        edge: Log potentials of shape (batch, T-1, K, C, C).
        lengths: Sequence lengths of shape (batch,).
        checkpoint_interval: Interval between checkpoints. Defaults to √T.
        semiring: Either "log" or "max". Note: Triton kernel currently
            only supports "log" semiring.

    Returns:
        partition: Log partition function of shape (batch,).
    """
    return SemiCRFTritonCheckpointedBackward.apply(
        edge, lengths, checkpoint_interval, semiring
    )
