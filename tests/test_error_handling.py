"""
Tests for error handling and edge cases in SemiMarkov.

These tests verify that the library handles invalid inputs gracefully
and produces meaningful error messages or handles edge cases correctly.
"""

import pytest
import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


class TestInvalidDimensions:
    """Tests for invalid input dimensions."""

    def test_mismatched_c_dimensions(self):
        """Edge potentials must have matching C dimensions (last two dims)."""
        batch, N, K, C1, C2 = 2, 6, 3, 4, 5  # C1 != C2
        edge = torch.randn(batch, N - 1, K, C1, C2)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        with pytest.raises(AssertionError, match="Transition shape"):
            sm.logpartition(edge, lengths=lengths)

    def test_wrong_edge_ndim(self):
        """Edge potentials must be 5D tensor."""
        batch, N, K, C = 2, 6, 3, 4
        # Create 4D tensor (missing one dimension)
        edge = torch.randn(batch, N - 1, K, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            sm.logpartition(edge, lengths=lengths)


class TestLengthValidation:
    """Tests for sequence length validation."""

    def test_length_exceeds_n(self):
        """Length cannot exceed N (sequence dimension + 1)."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        # Length of 10 exceeds N=6
        lengths = torch.tensor([10, 6], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        with pytest.raises(AssertionError, match="Length longer than edge scores"):
            sm.logpartition(edge, lengths=lengths)

    def test_max_length_must_equal_n(self):
        """At least one sequence must have length equal to N."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        # Both lengths are less than N
        lengths = torch.tensor([4, 5], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        with pytest.raises(AssertionError, match="At least one in batch must be length N"):
            sm.logpartition(edge, lengths=lengths)

    def test_valid_variable_lengths(self):
        """Variable lengths work when max length equals N."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        # One at max length, one shorter
        lengths = torch.tensor([6, 4], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        # Should not raise
        v, _, _ = sm.logpartition(edge, lengths=lengths)
        assert torch.isfinite(v).all()


class TestNumericalEdgeCases:
    """Tests for numerical edge cases."""

    def test_nan_potentials_propagate(self):
        """NaN in potentials should propagate to output."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        edge[0, 2, 1, 0, 0] = float("nan")
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # NaN should propagate to at least the first batch item
        assert torch.isnan(v[..., 0]).any()
        # Second batch item should be finite (no NaN in its inputs)
        assert torch.isfinite(v[..., 1]).all()

    def test_inf_potentials(self):
        """Inf potentials should be handled (common in masked positions)."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        # Set some potentials to -inf (mask out certain transitions)
        edge[0, 0, 0, :, :] = float("-inf")
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # Should still produce finite results (other paths available)
        # Note: if ALL paths are masked, result will be -inf
        assert not torch.isnan(v).any()

    def test_very_large_potentials(self):
        """Very large potentials shouldn't cause overflow in log-space."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C) * 100  # Large values
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # Should be finite (log-space arithmetic handles large values)
        assert torch.isfinite(v).all()

    def test_very_small_potentials(self):
        """Very small potentials shouldn't cause underflow issues."""
        batch, N, K, C = 2, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C) * 0.001  # Small values
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(v).all()


class TestMinimalConfigurations:
    """Tests for minimal/edge case configurations."""

    def test_batch_size_1(self):
        """Works with batch size 1."""
        batch, N, K, C = 1, 6, 3, 4
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert v.shape[-1] == batch
        assert torch.isfinite(v).all()

    def test_c_equals_1(self):
        """Works with single class C=1."""
        batch, N, K, C = 2, 6, 3, 1
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(v).all()

    def test_k_equals_2(self):
        """Works with minimum K=2."""
        batch, N, K, C = 2, 6, 2, 3
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(v).all()

    def test_n_equals_2(self):
        """Works with minimum sequence N=2."""
        batch, N, K, C = 2, 2, 2, 3
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(v).all()

    def test_n_less_than_k(self):
        """Works when N < K (not all durations reachable)."""
        batch, N, K, C = 2, 4, 8, 3  # N=4 but K=8
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(v).all()


class TestStreamingEdgeCases:
    """Test that streaming backend handles edge cases correctly."""

    def test_streaming_handles_short_sequence(self):
        """Streaming produces finite results for short sequences."""
        batch, N, K, C = 2, 4, 3, 2
        torch.manual_seed(42)
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        v_streaming, _, _ = sm._dp_scan_streaming(edge.clone(), lengths)

        assert torch.isfinite(v_streaming).all()

    def test_streaming_handles_variable_lengths(self):
        """Streaming produces finite results for variable lengths."""
        batch, N, K, C = 3, 8, 4, 2
        torch.manual_seed(123)
        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.tensor([8, 6, 4], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        v_streaming, _, _ = sm._dp_scan_streaming(edge.clone(), lengths)

        assert torch.isfinite(v_streaming).all()


class TestGradientEdgeCases:
    """Test gradient computation edge cases."""

    def test_gradient_with_inf_potentials(self):
        """Gradients should handle -inf potentials (masked transitions)."""
        batch, N, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, N - 1, K, C, C, requires_grad=True)

        # Mask some transitions
        with torch.no_grad():
            edge.data[0, 0, 0, 0, :] = float("-inf")

        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)
        v.sum().backward()

        # Gradient should exist and be finite where potentials were finite
        assert edge.grad is not None
        # Masked positions may have 0 or nan gradient, that's OK
        # But unmasked positions should have finite gradients
        assert torch.isfinite(edge.grad[0, 1:]).all()  # After the masked position

    def test_gradient_with_requires_grad_false(self):
        """No error when requires_grad=False."""
        batch, N, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, N - 1, K, C, C, requires_grad=False)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # Should complete without error
        assert torch.isfinite(v).all()
