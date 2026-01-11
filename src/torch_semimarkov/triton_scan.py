"""
Fused streaming Semi-Markov CRF forward scan implementations.

This module provides optimized implementations of the streaming forward scan
for Semi-Markov CRFs. The key optimization is fusing the O(N) loop into
a single operation, keeping the K×C frontier in fast memory.

Implementations:
1. PyTorch reference (CPU/GPU) - always available, used for testing
2. Triton kernel (GPU only) - requires triton package, much faster when feasible

Usage:
    from torch_semimarkov.triton_scan import semi_crf_triton_forward
    partition = semi_crf_triton_forward(edge, lengths)
"""

import torch
import math

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


def semi_crf_forward_pytorch(edge, lengths):
    """
    Reference PyTorch implementation matching _dp_scan_streaming semantics.

    This implementation:
    - Works on CPU and GPU
    - Uses O(K×C) ring buffer (same as streaming scan)
    - Serves as reference for correctness validation
    - Used as fallback when Triton not available
    - Supports gradient computation via autograd

    Recurrence:
        beta[n, c] = logsumexp_{k=1..min(K-1,n), c_prev} (
            beta[n-k, c_prev] + edge[n-k, k, c, c_prev]
        )

    Args:
        edge: (batch, N-1, K, C, C) log potentials
        lengths: (batch,) sequence lengths

    Returns:
        partition: (batch,) log partition function
    """
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

        # First logsumexp: over c_prev (source labels)
        scores = torch.logsumexp(beta_prev.unsqueeze(-2) + edge_slice, dim=-1)  # (batch, k_eff, C)

        # Second logsumexp: over duration dimension
        beta_n = torch.logsumexp(scores, dim=1)  # (batch, C)

        # Capture final beta for sequences ending at this position
        mask_end = (lengths == (n + 1)).view(batch, 1)
        final_beta = torch.where(mask_end, beta_n, final_beta)

        # Update ring buffer (replace entry, don't modify in-place)
        head = (head + 1) % ring_len
        beta_ring[head] = beta_n

    # Final partition: logsumexp over labels
    partition = torch.logsumexp(final_beta, dim=-1)
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

    def launch_triton_kernel(edge, lengths):
        """
        Launch the Triton kernel with proper buffer allocation.

        Args:
            edge: (batch, N-1, K, C, C) contiguous CUDA tensor
            lengths: (batch,) CUDA tensor

        Returns:
            partition: (batch,) log partition function
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

        # Launch kernel
        grid = (batch,)
        semi_crf_scan_kernel[grid](
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
    """
    Autograd wrapper with gradient checkpointing.

    Forward: Triton kernel (if available) or PyTorch fallback
    Backward: Recompute forward with gradients (checkpointing)
    """

    @staticmethod
    def forward(ctx, edge, lengths, use_triton=True):
        # Check if Triton kernel is applicable
        use_triton_kernel = HAS_TRITON and use_triton and edge.is_cuda

        if use_triton_kernel:
            partition = launch_triton_kernel(edge, lengths)
        else:
            partition = semi_crf_forward_pytorch(edge.detach(), lengths)

        ctx.save_for_backward(edge, lengths)
        ctx.use_triton = use_triton_kernel

        return partition

    @staticmethod
    def backward(ctx, grad_output):
        edge, lengths = ctx.saved_tensors

        # Recompute forward with gradients (checkpointing)
        edge_grad = edge.detach().requires_grad_(True)

        with torch.enable_grad():
            partition = semi_crf_forward_pytorch(edge_grad, lengths)

            # Use grad_outputs to weight the gradients
            # This computes: sum_b(grad_output[b] * d(partition[b])/d(edge_grad))
            grad_edge = torch.autograd.grad(
                outputs=partition, inputs=edge_grad, grad_outputs=grad_output, create_graph=False
            )[0]

        return grad_edge, None, None


def semi_crf_triton_forward(edge, lengths, use_triton=True, validate=False):
    """
    Main entry point for Semi-Markov CRF forward scan.

    Uses Triton kernel when available and applicable, otherwise
    falls back to optimized PyTorch implementation.

    Args:
        edge: (batch, N-1, K, C, C) log potentials
        lengths: (batch,) sequence lengths
        use_triton: If True, use Triton when possible
        validate: If True, use float64 PyTorch implementation for
            high-precision debugging. Useful for validating numerical
            accuracy. Returns result in original dtype.

    Returns:
        partition: (batch,) log partition function
    """
    if validate:
        # Use float64 for high-precision validation
        orig_dtype = edge.dtype
        partition = semi_crf_forward_pytorch(edge.double(), lengths)
        return partition.to(orig_dtype)

    return SemiCRFTritonForward.apply(edge, lengths, use_triton)


# =============================================================================
# Testing
# =============================================================================


def test_against_library():
    """Test against the library's streaming implementation."""
    try:
        from torch_semimarkov import SemiMarkov
        from torch_semimarkov.semirings import LogSemiring
    except ImportError:
        print("torch_semimarkov not fully installed, skipping library test")
        return True

    print("Testing against library _dp_scan_streaming...")

    test_cases = [
        (1, 10, 4, 2),
        (1, 20, 8, 4),
        (4, 50, 16, 8),
        (2, 100, 32, 4),
        (2, 100, 64, 8),  # Larger K
        (2, 50, 16, 16),  # Larger C
        (2, 50, 16, 32),  # C=32
    ]

    all_passed = True
    for batch, N, K, C in test_cases:
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        # Library reference
        sm = SemiMarkov(LogSemiring)
        lib_partition, _, _ = sm._dp_scan_streaming(edge, lengths)
        lib_partition = LogSemiring.unconvert(lib_partition)

        # Our PyTorch reference
        our_partition = semi_crf_forward_pytorch(edge, lengths)

        max_diff = (lib_partition - our_partition).abs().max().item()
        passed = max_diff < 1e-4

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: batch={batch}, N={N}, K={K}, C={C}, max_diff={max_diff:.2e}")

        if not passed:
            print(f"    Library: {lib_partition}")
            print(f"    Ours:    {our_partition}")
            all_passed = False

    return all_passed


def test_triton_kernel():
    """Test Triton kernel against PyTorch reference."""
    if not HAS_TRITON:
        print("Triton not available, skipping kernel test")
        return True

    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel test")
        return True

    print("Testing Triton kernel vs PyTorch reference...")

    test_cases = [
        # Basic tests
        (1, 20, 4, 4),
        (4, 50, 8, 8),
        (2, 100, 16, 8),
        # Large K tests
        (1, 100, 32, 8),
        (2, 100, 64, 8),
        # Large C tests (up to 32 labels)
        (2, 50, 8, 16),
        (2, 50, 16, 24),
        (2, 50, 16, 32),
        # Combined large K and C
        (1, 100, 32, 16),
        (1, 100, 64, 32),
    ]

    all_passed = True
    for batch, N, K, C in test_cases:
        edge = torch.randn(batch, N - 1, K, C, C, device="cuda")
        lengths = torch.full((batch,), N, dtype=torch.long, device="cuda")

        # PyTorch reference
        ref = semi_crf_forward_pytorch(edge, lengths)

        # Triton kernel
        triton_out = launch_triton_kernel(edge, lengths)

        max_diff = (ref - triton_out).abs().max().item()
        passed = max_diff < 1e-3

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: batch={batch}, N={N}, K={K}, C={C}, max_diff={max_diff:.2e}")

        if not passed:
            print(f"    PyTorch: {ref.tolist()}")
            print(f"    Triton:  {triton_out.tolist()}")
            all_passed = False

    return all_passed


def test_validate_mode():
    """Test validate mode with float64 precision."""
    print("Testing validate mode (float64)...")

    if not torch.cuda.is_available() or not HAS_TRITON:
        print("  Skipping (requires CUDA + Triton)")
        return True

    # Use a larger case where precision differences are visible
    batch, N, K, C = 4, 500, 64, 8
    edge = torch.randn(batch, N - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), N, dtype=torch.long, device="cuda")

    # Triton (float32)
    triton_out = semi_crf_triton_forward(edge, lengths, use_triton=True)

    # PyTorch float32
    pytorch_f32 = semi_crf_triton_forward(edge, lengths, use_triton=False)

    # Validate mode (float64)
    validate_out = semi_crf_triton_forward(edge, lengths, validate=True)

    # Compare
    triton_vs_f32 = (triton_out - pytorch_f32).abs().max().item()
    triton_vs_f64 = (triton_out - validate_out).abs().max().item()
    f32_vs_f64 = (pytorch_f32 - validate_out).abs().max().item()

    print(f"  Config: batch={batch}, N={N}, K={K}, C={C}")
    print(f"  Triton vs PyTorch(f32): {triton_vs_f32:.2e}")
    print(f"  Triton vs validate(f64): {triton_vs_f64:.2e}")
    print(f"  PyTorch(f32) vs validate(f64): {f32_vs_f64:.2e}")

    return True


def test_gradients():
    """Test gradient computation."""
    print("Testing gradients...")

    batch, N, K, C = 2, 20, 16, 4
    edge = torch.randn(batch, N - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), N, dtype=torch.long)

    # Forward
    partition = semi_crf_triton_forward(edge, lengths, use_triton=False)
    loss = partition.sum()

    # Backward
    loss.backward()

    print(f"  Forward passed: partition = {partition.tolist()}")
    print(f"  Backward passed: grad shape = {edge.grad.shape}")
    print(f"  Grad stats: min={edge.grad.min():.4f}, max={edge.grad.max():.4f}")

    return True


def benchmark_against_semimarkov(batch=4, N=500, K=64, C=8, n_iters=10):
    """
    Benchmark Triton fused kernel against SemiMarkov._dp_standard_vectorized.

    This compares:
    1. Triton fused kernel (this module)
    2. SemiMarkov._dp_standard_vectorized (streaming linear scan from semimarkov.py)
    3. PyTorch reference (this module)
    """
    import time

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    # Try to import SemiMarkov from the local torch_semimarkov package
    try:
        from torch_semimarkov import SemiMarkov
        from torch_semimarkov.semirings import LogSemiring
    except ImportError as e:
        print(f"Could not import torch_semimarkov SemiMarkov: {e}")
        print("Skipping comparison benchmark")
        return

    print(f"Benchmark: Triton vs SemiMarkov._dp_standard_vectorized")
    print(f"Config: batch={batch}, N={N}, K={K}, C={C}")
    print("-" * 60)

    device = "cuda"
    edge = torch.randn(batch, N - 1, K, C, C, device=device)
    lengths = torch.full((batch,), N, dtype=torch.long, device=device)

    # Create SemiMarkov instance with LogSemiring
    sm = SemiMarkov(LogSemiring)

    # Warmup all implementations
    for _ in range(3):
        _ = launch_triton_kernel(edge, lengths)
        _, _, _ = sm._dp_standard_vectorized(edge, lengths)
        _ = semi_crf_forward_pytorch(edge, lengths)
    torch.cuda.synchronize()

    # Benchmark Triton fused kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        triton_out = launch_triton_kernel(edge, lengths)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark SemiMarkov._dp_standard_vectorized
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        sm_out, _, _ = sm._dp_standard_vectorized(edge, lengths)
    torch.cuda.synchronize()
    sm_vectorized_ms = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark PyTorch reference (this module)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        pytorch_out = semi_crf_forward_pytorch(edge, lengths)
    torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - start) / n_iters * 1000

    # Convert outputs for comparison
    # SemiMarkov returns semiring-converted value, need to unconvert
    sm_partition = LogSemiring.unconvert(sm_out)

    # Check consistency
    triton_vs_sm = (triton_out - sm_partition).abs().max().item()
    triton_vs_pytorch = (triton_out - pytorch_out).abs().max().item()
    sm_vs_pytorch = (sm_partition - pytorch_out).abs().max().item()

    print(f"Triton fused kernel:              {triton_ms:8.2f} ms")
    print(f"SemiMarkov._dp_standard_vectorized: {sm_vectorized_ms:8.2f} ms")
    print(f"PyTorch reference (this module):  {pytorch_ms:8.2f} ms")
    print()
    print(f"Speedup Triton vs SemiMarkov:     {sm_vectorized_ms/triton_ms:.2f}x")
    print(f"Speedup Triton vs PyTorch ref:    {pytorch_ms/triton_ms:.2f}x")
    print()
    print("Consistency check (max abs diff):")
    print(f"  Triton vs SemiMarkov:           {triton_vs_sm:.2e}")
    print(f"  Triton vs PyTorch ref:          {triton_vs_pytorch:.2e}")
    print(f"  SemiMarkov vs PyTorch ref:      {sm_vs_pytorch:.2e}")

    return {
        "triton_ms": triton_ms,
        "sm_vectorized_ms": sm_vectorized_ms,
        "pytorch_ms": pytorch_ms,
        "triton_vs_sm": triton_vs_sm,
    }


def benchmark(batch=4, N=1024, K=64, C=8, n_iters=10, device="cpu"):
    """Benchmark implementations."""
    import time

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available, skipping GPU benchmark")
        return None

    edge = torch.randn(batch, N - 1, K, C, C, device=device)
    lengths = torch.full((batch,), N, dtype=torch.long, device=device)

    # Warmup PyTorch
    for _ in range(3):
        _ = semi_crf_forward_pytorch(edge, lengths)
    if device == "cuda":
        torch.cuda.synchronize()

    # Time PyTorch
    start = time.perf_counter()
    for _ in range(n_iters):
        partition_pt = semi_crf_forward_pytorch(edge, lengths)
    if device == "cuda":
        torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - start) / n_iters * 1000

    throughput = batch * N / (pytorch_ms / 1000) / 1e6

    print(f"PyTorch ({device}):")
    print(f"  Config: batch={batch}, N={N}, K={K}, C={C}")
    print(f"  Time: {pytorch_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2f} M positions/sec")

    # Triton benchmark (GPU only)
    if device == "cuda" and HAS_TRITON:
        for _ in range(3):
            _ = launch_triton_kernel(edge, lengths)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            partition_tr = launch_triton_kernel(edge, lengths)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / n_iters * 1000

        # Check correctness
        max_diff = (partition_pt - partition_tr).abs().max().item()

        print(f"Triton:")
        print(f"  Time: {triton_ms:.2f} ms")
        print(f"  Speedup vs PyTorch: {pytorch_ms/triton_ms:.2f}x")
        print(f"  Max diff from PyTorch: {max_diff:.2e}")

        return pytorch_ms, triton_ms

    return pytorch_ms, None


if __name__ == "__main__":
    print("=" * 60)
    print("Semi-Markov CRF Fused Streaming Scan")
    print("=" * 60)
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\n1. Testing against library implementation (CPU):")
    test_against_library()

    print("\n2. Testing Triton kernel (GPU):")
    test_triton_kernel()

    print("\n3. Testing validate mode (float64):")
    test_validate_mode()

    print("\n4. Testing gradients:")
    test_gradients()

    print("\n5. Benchmarking on CPU:")
    benchmark(batch=2, N=100, K=16, C=8, device="cpu")

    if torch.cuda.is_available():
        print("\n6. Benchmarking on GPU (C=8):")
        benchmark(batch=4, N=500, K=64, C=8, device="cuda")

        print("\n7. Benchmarking on GPU (C=32):")
        benchmark(batch=4, N=500, K=64, C=32, device="cuda")

        print("\n8. Comparison with SemiMarkov._dp_standard_vectorized:")
        benchmark_against_semimarkov(batch=4, N=500, K=64, C=8)
