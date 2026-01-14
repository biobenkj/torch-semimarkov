r"""Fused streaming Semi-Markov CRF forward scan implementations.

This module provides optimized implementations of the streaming forward scan
for Semi-Markov CRFs. The key optimization is fusing the O(N) loop into
a single GPU kernel, keeping the :math:`K \times C` frontier in fast memory.

Two implementations are provided:

1. **PyTorch reference** (CPU/GPU): Always available, used for testing and as fallback
2. **Triton kernel** (GPU only): Requires triton package, ~45x faster when feasible

The Triton kernel uses a fused scan that:

- Processes one batch element per thread block
- Maintains the ring buffer in L1/L2 cache
- Avoids Python loop overhead entirely

.. note::
    The Triton kernel is automatically used when available and the input is on CUDA.
    Falls back to PyTorch implementation on CPU or when Triton is not installed.

Examples::

    >>> from torch_semimarkov.triton_scan import semi_crf_triton_forward
    >>> edge = torch.randn(4, 99, 8, 6, 6).cuda()
    >>> lengths = torch.full((4,), 100).cuda()
    >>> partition = semi_crf_triton_forward(edge, lengths)
"""

import torch

# Triton is optional - kernel only available when installed and on GPU
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# =============================================================================
# PyTorch Reference Implementation (CPU + GPU)
# =============================================================================


def semi_crf_forward_pytorch(edge, lengths, semiring="log"):
    r"""semi_crf_forward_pytorch(edge, lengths, semiring="log") -> Tensor

    Reference PyTorch implementation of the streaming forward scan.

    Implements the same O(KC) ring buffer algorithm as :meth:`SemiMarkov._dp_scan_streaming`,
    but with pure PyTorch operations. Used as the reference implementation for correctness
    validation and as fallback when Triton is not available.

    .. math::
        \beta[n, c] = \text{logsumexp}_{k=1}^{\min(K-1,n)} \sum_{c'} \left(
            \beta[n-k, c'] + \text{edge}[n-k, k, c, c']
        \right)

    Args:
        edge (Tensor): Log potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        semiring (str, optional): Semiring to use for computation. Either ``"log"``
            (logsumexp, default) or ``"max"`` (Viterbi/max-plus). Default: ``"log"``

    Returns:
        Tensor: Log partition function (log semiring) or Viterbi score (max semiring)
            of shape :math:`(\text{batch},)`.

    .. note::
        This implementation supports gradient computation via autograd and works
        on both CPU and GPU.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")
    batch, N_1, K, C, _ = edge.shape
    N = N_1 + 1
    device = edge.device
    dtype = edge.dtype

    NEG_INF = -1e9

    # Ring buffer as list of tensors to avoid in-place updates
    # This allows proper gradient tracking
    ring_len = K
    initial_beta = torch.zeros((batch, C), device=device, dtype=dtype)
    beta_ring = [initial_beta] + [
        torch.full((batch, C), NEG_INF, device=device, dtype=dtype) for _ in range(ring_len - 1)
    ]
    head = 0

    # Duration indices (reused each iteration)
    dur_full = torch.arange(1, K, device=device)

    # Final beta storage (captured at each batch's sequence end)
    final_beta = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Handle length=1: partition = logsumexp(0, 0, ..., 0) = log(C)
    mask_len1 = (lengths == 1).view(batch, 1)
    final_beta = torch.where(mask_len1, initial_beta, final_beta)

    # Main scan loop
    for n in range(1, N):
        # Number of valid durations at this position
        k_eff = min(K - 1, n)
        dur = dur_full[:k_eff]  # [1, 2, ..., k_eff]
        start = n - dur  # positions where segments start

        # Get previous betas from ring buffer
        # ring_idx[i] = (head - (dur[i] - 1)) % ring_len
        ring_idx = [(head - (d.item() - 1)) % ring_len for d in dur]
        beta_prev = torch.stack([beta_ring[i] for i in ring_idx], dim=1)  # (batch, k_eff, C)

        # Get edge potentials
        edge_slice = edge[:, start, dur, :, :]  # (batch, k_eff, C, C)

        # First reduction: over c_prev (source labels)
        if semiring == "log":
            scores = torch.logsumexp(
                beta_prev.unsqueeze(-2) + edge_slice, dim=-1
            )  # (batch, k_eff, C)
        else:  # max semiring
            scores = torch.max(beta_prev.unsqueeze(-2) + edge_slice, dim=-1)[0]  # (batch, k_eff, C)

        # Second reduction: over duration dimension
        if semiring == "log":
            beta_n = torch.logsumexp(scores, dim=1)  # (batch, C)
        else:  # max semiring
            beta_n = torch.max(scores, dim=1)[0]  # (batch, C)

        # Capture final beta for sequences ending at this position
        mask_end = (lengths == (n + 1)).view(batch, 1)
        final_beta = torch.where(mask_end, beta_n, final_beta)

        # Update ring buffer (replace entry, don't modify in-place)
        head = (head + 1) % ring_len
        beta_ring[head] = beta_n

    # Final partition: reduction over labels
    if semiring == "log":
        partition = torch.logsumexp(final_beta, dim=-1)
    else:  # max semiring
        partition = torch.max(final_beta, dim=-1)[0]
    return partition


# =============================================================================
# Triton Kernels (GPU only, optional)
# =============================================================================

if HAS_TRITON:

    @triton.jit
    def semi_crf_scan_kernel(
        # Inputs
        edge_ptr,  # (batch, N-1, K, C, C) - edge potentials
        ring_ptr,  # (batch, K, C_PAD) - ring buffer (read/write)
        out_ptr,  # (batch,) - output partition
        lengths_ptr,  # (batch,) - sequence lengths
        # Dimensions
        batch_size,
        N: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        # Strides for edge tensor
        stride_eb,
        stride_en,
        stride_ek,
        stride_ec1,
        stride_ec2,
        # Strides for ring buffer (uses C_PAD)
        stride_rb,
        stride_rk,
        stride_rc,
    ):
        """
        Fused Semi-Markov CRF forward scan with arbitrary K support.

        Uses global memory ring buffer (L2/L1 cached) for the DP state.
        Each program handles one batch element.
        Loads full [C, C] edge blocks for better numerical stability.
        C_PAD is padded to power of 2 for Triton's tl.arange requirement.

        Ring buffer layout: ring[batch, k, c_pad]
        - k=0 is head (most recent beta)
        - k=1..K-1 are older betas
        - We rotate head pointer instead of shifting data
        """
        NEG_INF: tl.constexpr = -1e9  # Match PyTorch reference

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C  # mask for valid label indices

        # 2D indices for [C_PAD, C_PAD] edge block loads
        c_dst = tl.arange(0, C_PAD)[:, None]  # [C_PAD, 1]
        c_src = tl.arange(0, C_PAD)[None, :]  # [1, C_PAD]
        c_mask_2d = (c_dst < C) & (c_src < C)  # [C_PAD, C_PAD]

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        ring_base = ring_ptr + batch_idx * stride_rb

        # Initialize ring buffer: slot 0 = 0.0, rest = NEG_INF
        for k_init in tl.static_range(0, K):
            val = 0.0 if k_init == 0 else NEG_INF
            ring_offset = ring_base + k_init * stride_rk + c_idx * stride_rc
            tl.store(ring_offset, tl.where(c_mask, val, NEG_INF), mask=c_mask)

        # Track final beta for each batch - shape [C_PAD]
        final_beta = tl.where(c_mask, 0.0, NEG_INF).to(tl.float32)

        # Main loop over sequence positions
        for n in tl.range(1, N):
            # Use mask instead of break (Triton doesn't support break)
            active = n < seq_len

            # Accumulate new_beta = logsumexp over (k, c_prev) - shape [C_PAD]
            new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Loop over durations k = 1, 2, ..., K-1
            for k in tl.range(1, K):
                # Skip if duration exceeds position
                k_valid = (k <= n) & (k <= K - 1)

                start_pos = n - k

                # Ring index for beta[n-k]: (n-k) % K
                ring_k_idx = (n - k) % K

                # Load beta_prev for ALL labels [C_PAD] from ring buffer
                beta_prev_all = tl.load(
                    ring_base + ring_k_idx * stride_rk + c_idx * stride_rc,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )  # shape [C_PAD]

                # Load entire [C_PAD, C_PAD] edge block for this (start_pos, k)
                # Only load valid [C, C] portion
                edge_offset_2d = (
                    edge_base
                    + start_pos * stride_en
                    + k * stride_ek
                    + c_dst * stride_ec1
                    + c_src * stride_ec2
                )  # [C_PAD, C_PAD]

                edge_block = tl.load(
                    edge_offset_2d, mask=active & k_valid & c_mask_2d, other=NEG_INF
                )  # [C_PAD, C_PAD]

                # Compute scores: scores[c, cp] = beta_prev[cp] + edge[c, cp]
                scores = beta_prev_all[None, :] + edge_block  # [C_PAD, C_PAD]

                # Mask out invalid source labels before reduction
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Numerically stable logsumexp over source labels (axis=1)
                max_scores = tl.max(scores, axis=1)  # [C_PAD]
                score_for_k = max_scores + tl.log(
                    tl.sum(tl.exp(scores - max_scores[:, None]), axis=1)
                )  # [C_PAD]

                # Mask invalid durations and invalid destination labels
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Accumulate this duration into new_beta via logsumexp
                max_nb = tl.maximum(new_beta, score_for_k)
                new_beta = max_nb + tl.log(tl.exp(new_beta - max_nb) + tl.exp(score_for_k - max_nb))

            # Store new_beta to ring buffer at current head position
            new_head = n % K
            new_beta_masked = tl.where(active & c_mask, new_beta, NEG_INF)
            tl.store(
                ring_base + new_head * stride_rk + c_idx * stride_rc,
                new_beta_masked,
                mask=active & c_mask,
            )

            # Capture final beta at sequence end
            is_final = n == seq_len - 1
            final_beta = tl.where(is_final & c_mask, new_beta_masked, final_beta)

        # Final reduction: logsumexp over labels (only valid ones)
        final_beta_masked = tl.where(c_mask, final_beta, NEG_INF)
        max_val = tl.max(final_beta_masked, axis=0)
        exp_fb = tl.where(c_mask, tl.exp(final_beta - max_val), 0.0)
        sum_exp = tl.sum(exp_fb, axis=0)
        partition = max_val + tl.log(sum_exp)

        # Store result (partition is a scalar)
        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_scan_kernel_max(
        # Inputs
        edge_ptr,  # (batch, N-1, K, C, C) - edge potentials
        ring_ptr,  # (batch, K, C_PAD) - ring buffer (read/write)
        out_ptr,  # (batch,) - output partition
        lengths_ptr,  # (batch,) - sequence lengths
        # Dimensions
        batch_size,
        N: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        # Strides for edge tensor
        stride_eb,
        stride_en,
        stride_ek,
        stride_ec1,
        stride_ec2,
        # Strides for ring buffer (uses C_PAD)
        stride_rb,
        stride_rk,
        stride_rc,
    ):
        """
        Fused Semi-Markov CRF forward scan using max semiring (Viterbi).

        Same structure as semi_crf_scan_kernel but uses max instead of logsumexp.
        This computes the Viterbi score (best path score) instead of log partition.

        Ring buffer layout: ring[batch, k, c_pad]
        - k=0 is head (most recent beta)
        - k=1..K-1 are older betas
        - We rotate head pointer instead of shifting data
        """
        NEG_INF: tl.constexpr = -1e9  # Match PyTorch reference

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C  # mask for valid label indices

        # 2D indices for [C_PAD, C_PAD] edge block loads
        c_dst = tl.arange(0, C_PAD)[:, None]  # [C_PAD, 1]
        c_src = tl.arange(0, C_PAD)[None, :]  # [1, C_PAD]
        c_mask_2d = (c_dst < C) & (c_src < C)  # [C_PAD, C_PAD]

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        ring_base = ring_ptr + batch_idx * stride_rb

        # Initialize ring buffer: slot 0 = 0.0, rest = NEG_INF
        for k_init in tl.static_range(0, K):
            val = 0.0 if k_init == 0 else NEG_INF
            ring_offset = ring_base + k_init * stride_rk + c_idx * stride_rc
            tl.store(ring_offset, tl.where(c_mask, val, NEG_INF), mask=c_mask)

        # Track final beta for each batch - shape [C_PAD]
        final_beta = tl.where(c_mask, 0.0, NEG_INF).to(tl.float32)

        # Main loop over sequence positions
        for n in tl.range(1, N):
            # Use mask instead of break (Triton doesn't support break)
            active = n < seq_len

            # Accumulate new_beta = max over (k, c_prev) - shape [C_PAD]
            new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Loop over durations k = 1, 2, ..., K-1
            for k in tl.range(1, K):
                # Skip if duration exceeds position
                k_valid = (k <= n) & (k <= K - 1)

                start_pos = n - k

                # Ring index for beta[n-k]: (n-k) % K
                ring_k_idx = (n - k) % K

                # Load beta_prev for ALL labels [C_PAD] from ring buffer
                beta_prev_all = tl.load(
                    ring_base + ring_k_idx * stride_rk + c_idx * stride_rc,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )  # shape [C_PAD]

                # Load entire [C_PAD, C_PAD] edge block for this (start_pos, k)
                # Only load valid [C, C] portion
                edge_offset_2d = (
                    edge_base
                    + start_pos * stride_en
                    + k * stride_ek
                    + c_dst * stride_ec1
                    + c_src * stride_ec2
                )  # [C_PAD, C_PAD]

                edge_block = tl.load(
                    edge_offset_2d, mask=active & k_valid & c_mask_2d, other=NEG_INF
                )  # [C_PAD, C_PAD]

                # Compute scores: scores[c, cp] = beta_prev[cp] + edge[c, cp]
                scores = beta_prev_all[None, :] + edge_block  # [C_PAD, C_PAD]

                # Mask out invalid source labels before reduction
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max over source labels (axis=1) - Viterbi instead of logsumexp
                score_for_k = tl.max(scores, axis=1)  # [C_PAD]

                # Mask invalid durations and invalid destination labels
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Accumulate this duration into new_beta via max
                new_beta = tl.maximum(new_beta, score_for_k)

            # Store new_beta to ring buffer at current head position
            new_head = n % K
            new_beta_masked = tl.where(active & c_mask, new_beta, NEG_INF)
            tl.store(
                ring_base + new_head * stride_rk + c_idx * stride_rc,
                new_beta_masked,
                mask=active & c_mask,
            )

            # Capture final beta at sequence end
            is_final = n == seq_len - 1
            final_beta = tl.where(is_final & c_mask, new_beta_masked, final_beta)

        # Final reduction: max over labels (only valid ones)
        final_beta_masked = tl.where(c_mask, final_beta, NEG_INF)
        partition = tl.max(final_beta_masked, axis=0)

        # Store result (partition is a scalar)
        tl.store(out_ptr + batch_idx, partition)

    def _next_power_of_2(n):
        """Return the smallest power of 2 >= n."""
        if n <= 0:
            return 1
        # Handle powers of 2
        if n & (n - 1) == 0:
            return n
        # Find next power of 2
        p = 1
        while p < n:
            p *= 2
        return p

    def launch_triton_kernel(edge, lengths, semiring="log"):
        """
        Launch the Triton kernel with proper buffer allocation.

        Args:
            edge: (batch, N-1, K, C, C) contiguous CUDA tensor
            lengths: (batch,) CUDA tensor
            semiring: "log" for logsumexp (partition function) or "max" for Viterbi

        Returns:
            partition: (batch,) log partition function or Viterbi score
        """
        batch, N_1, K, C, _ = edge.shape
        N = N_1 + 1

        # Pad C to next power of 2 (Triton requirement for tl.arange)
        C_PAD = _next_power_of_2(C)

        # Ensure contiguous
        edge = edge.contiguous()

        # Allocate ring buffer with padded C (small, will be L2 cached)
        ring_buffer = torch.empty((batch, K, C_PAD), device=edge.device, dtype=edge.dtype)

        # Output buffer
        partition = torch.empty(batch, device=edge.device, dtype=edge.dtype)

        # Get strides
        stride_eb, stride_en, stride_ek, stride_ec1, stride_ec2 = edge.stride()
        stride_rb, stride_rk, stride_rc = ring_buffer.stride()

        # Launch kernel - select based on semiring
        grid = (batch,)
        kernel = semi_crf_scan_kernel if semiring == "log" else semi_crf_scan_kernel_max
        kernel[grid](
            edge,
            ring_buffer,
            partition,
            lengths,
            batch,
            N,
            K,
            C,
            C_PAD,
            stride_eb,
            stride_en,
            stride_ek,
            stride_ec1,
            stride_ec2,
            stride_rb,
            stride_rk,
            stride_rc,
        )

        return partition


# =============================================================================
# Autograd Function
# =============================================================================


class SemiCRFTritonForward(torch.autograd.Function):
    r"""Autograd function with gradient checkpointing for Semi-CRF forward.

    This custom autograd function enables using the fast Triton kernel for forward
    computation while supporting gradients via checkpointing (recomputing forward
    during backward).

    Forward:
        Uses Triton kernel when available and on CUDA, otherwise PyTorch fallback.

    Backward:
        Recomputes forward with autograd using the PyTorch implementation.
        This trades compute for memory (no intermediate activations stored).
    """

    @staticmethod
    def forward(ctx, edge, lengths, use_triton=True, semiring="log"):
        # Check if Triton kernel is applicable
        use_triton_kernel = HAS_TRITON and use_triton and edge.is_cuda

        if use_triton_kernel:
            partition = launch_triton_kernel(edge, lengths, semiring=semiring)
        else:
            partition = semi_crf_forward_pytorch(edge.detach(), lengths, semiring=semiring)

        ctx.save_for_backward(edge, lengths)
        ctx.use_triton = use_triton_kernel
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output):
        edge, lengths = ctx.saved_tensors
        semiring = ctx.semiring

        # Recompute forward with gradients (checkpointing)
        edge_grad = edge.detach().requires_grad_(True)

        with torch.enable_grad():
            partition = semi_crf_forward_pytorch(edge_grad, lengths, semiring=semiring)

            # Use grad_outputs to weight the gradients
            # This computes: sum_b(grad_output[b] * d(partition[b])/d(edge_grad))
            grad_edge = torch.autograd.grad(
                outputs=partition, inputs=edge_grad, grad_outputs=grad_output, create_graph=False
            )[0]

        return grad_edge, None, None, None


def semi_crf_triton_forward(edge, lengths, use_triton=True, validate=False, semiring="log"):
    r"""semi_crf_triton_forward(edge, lengths, use_triton=True, validate=False, semiring="log") -> Tensor

    Compute Semi-Markov CRF forward scan with optional GPU acceleration.

    Main entry point for the fused streaming scan. Uses the Triton kernel when
    available and applicable (CUDA input), otherwise falls back to the optimized
    PyTorch implementation.

    Args:
        edge (Tensor): Log potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        use_triton (bool, optional): If ``True``, use Triton kernel when possible.
            Default: ``True``
        validate (bool, optional): If ``True``, use float64 PyTorch implementation
            for high-precision debugging. Useful for validating numerical accuracy.
            Returns result in original dtype. Default: ``False``
        semiring (str, optional): Semiring to use for computation. Either ``"log"``
            (logsumexp for partition function) or ``"max"`` (Viterbi/max-plus for
            best path score). Default: ``"log"``

    Returns:
        Tensor: Log partition function (log semiring) or Viterbi score (max semiring)
            of shape :math:`(\text{batch},)`.

    Examples::

        >>> edge = torch.randn(4, 99, 8, 6, 6)
        >>> lengths = torch.full((4,), 100)
        >>> # CPU: uses PyTorch fallback
        >>> partition_cpu = semi_crf_triton_forward(edge, lengths)
        >>> # GPU: uses Triton kernel if available
        >>> partition_gpu = semi_crf_triton_forward(edge.cuda(), lengths.cuda())
        >>> # Viterbi (max semiring) for best path score
        >>> viterbi_score = semi_crf_triton_forward(edge, lengths, semiring="max")

    .. note::
        Gradients are computed via checkpointing: the forward pass is recomputed
        during backward using the PyTorch implementation for proper autograd support.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    if validate:
        # Use float64 for high-precision validation
        orig_dtype = edge.dtype
        partition = semi_crf_forward_pytorch(edge.double(), lengths, semiring=semiring)
        return partition.to(orig_dtype)

    return SemiCRFTritonForward.apply(edge, lengths, use_triton, semiring)
