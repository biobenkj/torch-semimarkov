"""
Tests verifying CPU-only operation of torch-semimarkov.

This test suite ensures that all operations work correctly on CPU,
which is the only supported device for CI/CD pipelines (due to GPU cost).

The library includes Triton GPU kernels, but they automatically fall back
to pure PyTorch implementations when CUDA is not available.
"""

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring
from torch_semimarkov.streaming import semi_crf_streaming_forward


class TestCPUOnlyOperation:
    """Verify all operations work correctly on CPU."""

    def test_default_tensor_device_is_cpu(self):
        """Default tensor creation should use CPU."""
        tensor = torch.randn(2, 3)
        assert tensor.device.type == "cpu"

    def test_semimarkov_on_cpu(self):
        """SemiMarkov inference works on CPU tensors."""
        batch, T, K, C = 2, 8, 4, 3
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        assert edge.device.type == "cpu"
        assert lengths.device.type == "cpu"

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert v.device.type == "cpu"
        assert torch.isfinite(v).all()

    def test_streaming_scan_on_cpu(self):
        """Streaming scan operates correctly on CPU."""
        batch, T, K, C = 2, 6, 3, 2
        torch.manual_seed(42)
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        # Streaming (only backend)
        v, _, _ = sm._dp_scan_streaming(edge.clone(), lengths)
        assert v.device.type == "cpu"
        assert torch.isfinite(v).all()

    def test_streaming_forward_on_cpu(self):
        """Streaming forward works on CPU."""
        batch, T, K, C = 2, 6, 4, 3
        # Build cumulative scores for streaming API
        scores = torch.randn(batch, T, C)
        cum_scores = torch.zeros(batch, T + 1, C)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K)
        lengths = torch.full((batch,), T, dtype=torch.long)

        # This should work on CPU (uses PyTorch fallback when CUDA not available)
        log_Z = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert log_Z.device.type == "cpu"
        assert torch.isfinite(log_Z).all()

    def test_marginals_on_cpu(self):
        """Marginals computation works on CPU."""
        batch, T, K, C = 2, 8, 4, 3
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        assert marginals.device.type == "cpu"
        assert torch.isfinite(marginals).all()

    def test_gradients_on_cpu(self):
        """Gradient computation works on CPU."""
        batch, T, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)
        v.sum().backward()

        assert edge.grad is not None
        assert edge.grad.device.type == "cpu"
        assert torch.isfinite(edge.grad).all()


class TestCPUConsistency:
    """Verify CPU results are consistent across multiple runs."""

    def test_deterministic_results(self):
        """Same input should produce same output on CPU."""
        batch, T, K, C = 2, 8, 4, 3
        torch.manual_seed(42)
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        v1, _, _ = sm.logpartition(edge, lengths=lengths)
        v2, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.equal(v1, v2)

    def test_gradient_deterministic(self):
        """Gradients should be deterministic on CPU."""
        batch, T, K, C = 2, 6, 3, 2
        torch.manual_seed(42)
        edge_base = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        # First run
        edge1 = edge_base.clone().requires_grad_(True)
        v1, _, _ = sm.logpartition(edge1, lengths=lengths)
        v1.sum().backward()
        grad1 = edge1.grad.clone()

        # Second run
        edge2 = edge_base.clone().requires_grad_(True)
        v2, _, _ = sm.logpartition(edge2, lengths=lengths)
        v2.sum().backward()
        grad2 = edge2.grad.clone()

        assert torch.equal(grad1, grad2)


class TestCPUDocumentation:
    """Tests that serve as documentation for CPU-only usage."""

    def test_explicit_cpu_device_specification(self, cpu_device):
        """Example of explicitly specifying CPU device."""
        batch, T, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, T - 1, K, C, C, device=cpu_device)
        lengths = torch.full((batch,), T, dtype=torch.long, device=cpu_device)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert v.device == cpu_device

    def test_no_cuda_required(self):
        """Verify the test suite doesn't require CUDA to pass."""
        # This test documents that CUDA is not required
        # The full test suite should pass without CUDA installed

        # Create tensors (will be on CPU)
        batch, T, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        # Run inference (use linear scan to get consistent 3-tuple return)
        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # Verify success
        assert torch.isfinite(v).all()
