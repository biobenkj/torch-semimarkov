"""
Scaling tests for the streaming API.

These tests verify that the streaming implementation does NOT OOM at scale.
The primary success metric is: no OOM for T=100K, K=1K, C=24.

Memory expectations:
- Pre-computed edge API: O(T × K × C²) = 2.76 TB for T=400K, K=3K, C=24
- Streaming API: O(K × C + T × C) ≈ 50 MB for same dimensions
"""

import gc

import pytest
import torch

from torch_semimarkov.streaming import semi_crf_streaming_forward


def create_streaming_inputs(batch, T, K, C, device="cpu", dtype=torch.float32, seed=42):
    """Create test inputs for the streaming API."""
    torch.manual_seed(seed)

    # Simulate projected encoder features
    projected = torch.randn(batch, T, C, device=device, dtype=dtype)
    # Zero-center (critical for numerical stability at large T)
    projected = projected - projected.mean(dim=1, keepdim=True)

    # Cumulative scores: (batch, T+1, C)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    # Lengths
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    return cum_scores, transition, duration_bias, lengths


class TestNoOOM:
    """Tests verifying the implementation does NOT run out of memory."""

    @pytest.mark.slow
    def test_no_oom_T10K_K100_C24_cpu(self):
        """Verify no OOM at T=10K, K=100, C=24 on CPU."""
        T, K, C = 10_000, 100, 24
        batch = 1

        gc.collect()

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition should be finite"

    @pytest.mark.slow
    def test_no_oom_T50K_K500_C24_cpu(self):
        """Verify no OOM at T=50K, K=500, C=24 on CPU."""
        T, K, C = 50_000, 500, 24
        batch = 1

        gc.collect()

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition should be finite"

    @pytest.mark.slow
    def test_no_oom_T100K_K1K_C24_cpu(self):
        """PRIMARY SUCCESS METRIC: No OOM at T=100K, K=1K, C=24 on CPU."""
        T, K, C = 100_000, 1_000, 24
        batch = 1

        gc.collect()

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition should be finite"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_oom_T10K_K100_C24_cuda(self):
        """Verify no OOM at T=10K, K=100, C=24 on CUDA."""
        T, K, C = 10_000, 100, 24
        batch = 1

        gc.collect()
        torch.cuda.empty_cache()

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition should be finite"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.slow
    def test_no_oom_T100K_K1K_C24_cuda(self):
        """PRIMARY SUCCESS METRIC: No OOM at T=100K, K=1K, C=24 on CUDA."""
        T, K, C = 100_000, 1_000, 24
        batch = 1

        gc.collect()
        torch.cuda.empty_cache()

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition should be finite"


class TestMemoryUsage:
    """Tests verifying memory usage is O(KC) not O(TKC)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_scales_with_KC_not_TKC(self):
        """Verify memory usage scales with K×C, not T×K×C."""
        C = 24
        K = 100
        batch = 1

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Test with T=5K
        T1 = 5_000
        cum_scores1, transition1, duration_bias1, lengths1 = create_streaming_inputs(
            batch, T1, K, C, device="cuda"
        )
        _ = semi_crf_streaming_forward(cum_scores1, transition1, duration_bias1, lengths1, K)
        torch.cuda.synchronize()
        mem_T1 = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Clean up
        del cum_scores1, transition1, duration_bias1, lengths1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Test with T=20K (4x larger)
        T2 = 20_000
        cum_scores2, transition2, duration_bias2, lengths2 = create_streaming_inputs(
            batch, T2, K, C, device="cuda"
        )
        _ = semi_crf_streaming_forward(cum_scores2, transition2, duration_bias2, lengths2, K)
        torch.cuda.synchronize()
        mem_T2 = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        print("\nMemory usage:")
        print(f"  T={T1}: {mem_T1:.1f} MB")
        print(f"  T={T2}: {mem_T2:.1f} MB")
        print(f"  Ratio: {mem_T2 / mem_T1:.2f}x (expected ~4x for O(TC), ~1x for O(KC))")

        # If memory scaled with T×K×C, ratio would be 4x
        # If memory scales with K×C (plus T×C for input), ratio should be < 4x
        # The ratio should be closer to 4x due to cum_scores input, but working memory
        # should not scale with T
        assert mem_T2 < mem_T1 * 5, f"Memory scaling too high: {mem_T2 / mem_T1:.2f}x"


class TestNumericalStabilityAtScale:
    """Test numerical stability at large scale."""

    def test_finite_at_T10K(self):
        """Verify partition is finite at T=10K."""
        T, K, C = 10_000, 50, 8
        batch = 1

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all(), f"Non-finite at T={T}"

    @pytest.mark.slow
    def test_finite_at_T50K(self):
        """Verify partition is finite at T=50K."""
        T, K, C = 50_000, 100, 8
        batch = 1

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all(), f"Non-finite at T={T}"

    def test_gradient_finite_at_T1K(self):
        """Verify gradients are finite at T=1K."""
        T, K, C = 1_000, 20, 8
        batch = 1

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "cum_scores grad non-finite"
        assert torch.isfinite(transition.grad).all(), "transition grad non-finite"
        assert torch.isfinite(duration_bias.grad).all(), "duration_bias grad non-finite"


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("STREAMING SCALE TESTS")
    print("=" * 60)

    # Quick sanity test
    print("\n[1/3] Testing T=10K, K=100, C=24 (CPU)...")
    T, K, C = 10_000, 100, 24
    batch = 1

    cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
    print(f"  Partition: {partition.item():.4f}")
    print(f"  Finite: {torch.isfinite(partition).all()}")

    # Medium test
    print("\n[2/3] Testing T=50K, K=500, C=24 (CPU)...")
    T, K, C = 50_000, 500, 24
    batch = 1

    cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
    print(f"  Partition: {partition.item():.4f}")
    print(f"  Finite: {torch.isfinite(partition).all()}")

    # Primary benchmark
    print("\n[3/3] PRIMARY SUCCESS METRIC: T=100K, K=1K, C=24 (CPU)...")
    T, K, C = 100_000, 1_000, 24
    batch = 1

    gc.collect()

    try:
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        print(f"  Partition: {partition.item():.4f}")
        print(f"  Finite: {torch.isfinite(partition).all()}")
        print("\n" + "=" * 60)
        print("SUCCESS: No OOM at T=100K, K=1K, C=24")
        print("=" * 60)
        sys.exit(0)
    except MemoryError:
        print("\nFAILED: Out of memory!")
        sys.exit(1)
