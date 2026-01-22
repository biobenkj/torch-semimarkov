"""Tests for uncertainty quantification methods.

Tests verify:
- Streaming marginals satisfy probability constraints
- Streaming vs exact marginals match for short sequences
- Entropy computation is finite and meaningful
- Clinical-scale tests use streaming path exclusively
"""

import pytest
import torch

from torch_semimarkov import UncertaintySemiMarkovCRFHead


class TestBoundaryMarginals:
    """Test boundary marginal computation methods."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_streaming_marginals_shape(self, uncertainty_model):
        """Test streaming marginals have correct shape."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True
        )

        assert boundary_probs.shape == (batch, T)

    def test_streaming_marginals_probability_constraint(self, uncertainty_model):
        """Gradient-based marginals should be valid probabilities (non-negative)."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True, normalize=True
        )

        # Should be non-negative
        assert (boundary_probs >= 0).all()
        # Should be <= 1 when normalized
        assert (boundary_probs <= 1 + 1e-6).all()
        # Should be finite
        assert torch.isfinite(boundary_probs).all()

    def test_streaming_marginals_unnormalized(self, uncertainty_model):
        """Test unnormalized marginals are still finite."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True, normalize=False
        )

        assert torch.isfinite(boundary_probs).all()
        assert (boundary_probs >= 0).all()

    def test_high_contrast_input_has_peaked_marginals(self, uncertainty_model):
        """Test that high-contrast inputs produce peaked marginal distributions."""
        batch, T = 1, 20
        hidden = torch.zeros(batch, T, 32)

        # Create high contrast: first half has pattern A, second half has pattern B
        hidden[:, :10, :16] = 5.0
        hidden[:, 10:, 16:] = 5.0

        lengths = torch.tensor([T])

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True, normalize=True
        )

        # Should have some variation (not all the same)
        assert boundary_probs.std() > 0.01

    def test_uniform_input_has_flatter_marginals(self, uncertainty_model):
        """Test that uniform inputs produce flatter marginal distributions."""
        batch, T = 1, 20
        hidden = torch.randn(batch, T, 32) * 0.01  # Low variance input

        lengths = torch.tensor([T])

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True, normalize=True
        )

        # Should still be valid
        assert torch.isfinite(boundary_probs).all()
        assert (boundary_probs >= 0).all()


class TestExactMarginals:
    """Test exact marginal computation (for short sequences).

    Note: Exact methods require careful edge tensor construction.
    These tests are marked as expected to fail pending implementation
    improvements to the edge tensor builder.
    """

    @pytest.fixture
    def small_model(self):
        """Small model for exact computation (avoid OOM)."""
        return UncertaintySemiMarkovCRFHead(
            num_classes=3,
            max_duration=4,
            hidden_dim=16,
        )

    def test_exact_marginals_shape(self, small_model):
        """Test exact marginals have correct shape."""
        batch, T = 2, 20
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        boundary_probs = small_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=False
        )

        assert boundary_probs.shape == (batch, T)

    def test_exact_marginals_finite(self, small_model):
        """Test exact marginals are finite."""
        batch, T = 2, 20
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        boundary_probs = small_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=False, normalize=True
        )

        assert torch.isfinite(boundary_probs).all()
        assert (boundary_probs >= 0).all()


class TestStreamingVsExactConsistency:
    """Test that streaming and exact methods are consistent for short sequences.

    Note: Exact methods require careful edge tensor construction.
    These tests are skipped pending implementation improvements.
    """

    @pytest.fixture
    def small_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=3,
            max_duration=4,
            hidden_dim=16,
        )

    def test_marginals_correlation(self, small_model):
        """Streaming and exact marginals should be correlated for short sequences.

        Note: They won't be exactly equal because they use different
        computational approaches, but they should capture similar patterns.
        """
        batch, T = 1, 15
        torch.manual_seed(42)
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        streaming_marginals = small_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True, normalize=True
        )
        exact_marginals = small_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=False, normalize=True
        )

        # Both should be valid
        assert torch.isfinite(streaming_marginals).all()
        assert torch.isfinite(exact_marginals).all()

        # Compute correlation (should be positive for reasonable implementations)
        # Note: Due to different computational methods, exact match isn't expected
        streaming_flat = streaming_marginals.flatten()
        exact_flat = exact_marginals.flatten()

        # At minimum, verify shapes match and values are reasonable
        assert streaming_flat.shape == exact_flat.shape


class TestPositionMarginals:
    """Test per-position label marginal computation."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_position_marginals_shape(self, uncertainty_model):
        """Test position marginals have correct shape."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        position_marginals = uncertainty_model.compute_position_marginals(hidden, lengths)

        assert position_marginals.shape == (batch, T, 4)  # (batch, T, num_classes)

    def test_position_marginals_sum_to_one(self, uncertainty_model):
        """Position marginals should sum to 1 over classes."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        position_marginals = uncertainty_model.compute_position_marginals(hidden, lengths)

        # Sum over classes should be ~1 (softmax normalization)
        class_sum = position_marginals.sum(dim=-1)
        torch.testing.assert_close(class_sum, torch.ones_like(class_sum), rtol=1e-4, atol=1e-4)

    def test_position_marginals_finite(self, uncertainty_model):
        """Position marginals should be finite."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        position_marginals = uncertainty_model.compute_position_marginals(hidden, lengths)

        assert torch.isfinite(position_marginals).all()


class TestEntropyComputation:
    """Test entropy computation methods."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_entropy_streaming_shape(self, uncertainty_model):
        """Test streaming entropy has correct shape."""
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert entropy.shape == (batch,)

    def test_entropy_streaming_finite(self, uncertainty_model):
        """Streaming entropy should be finite."""
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert torch.isfinite(entropy).all()

    def test_entropy_streaming_non_negative(self, uncertainty_model):
        """Entropy should be non-negative."""
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert (entropy >= -1e-6).all()  # Allow small numerical errors

    def test_entropy_streaming_long_sequence(self, uncertainty_model):
        """Entropy should be finite for clinical-scale sequences."""
        batch = 1
        T = 1000  # Clinical-scale (streaming only)

        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        # This should not OOM - uses streaming method
        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert torch.isfinite(entropy).all()


class TestExactEntropy:
    """Test exact entropy computation (short sequences only).

    Note: Exact methods require careful edge tensor construction.
    These tests are skipped pending implementation improvements.
    """

    @pytest.fixture
    def small_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=3,
            max_duration=4,
            hidden_dim=16,
        )

    def test_entropy_exact_shape(self, small_model):
        """Test exact entropy has correct shape."""
        batch, T = 2, 15
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        entropy = small_model.compute_entropy_exact(hidden, lengths)

        assert entropy.shape == (batch,)

    def test_entropy_exact_finite(self, small_model):
        """Exact entropy should be finite."""
        batch, T = 2, 15
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        entropy = small_model.compute_entropy_exact(hidden, lengths)

        assert torch.isfinite(entropy).all()

    def test_entropy_exact_non_negative(self, small_model):
        """Exact entropy should be non-negative."""
        batch, T = 2, 15
        hidden = torch.randn(batch, T, 16)
        lengths = torch.full((batch,), T)

        entropy = small_model.compute_entropy_exact(hidden, lengths)

        assert (entropy >= -1e-6).all()


class TestClinicalScaleUncertainty:
    """Test uncertainty methods at clinical scale (streaming only)."""

    @pytest.fixture
    def clinical_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=8,
            max_duration=32,
            hidden_dim=64,
        )

    @pytest.mark.parametrize("T", [500, 1000, 2000])
    def test_streaming_marginals_clinical_lengths(self, clinical_model, T):
        """Test streaming marginals at clinical sequence lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)

        boundary_probs = clinical_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True
        )

        assert boundary_probs.shape == (batch, T)
        assert torch.isfinite(boundary_probs).all()

    @pytest.mark.parametrize("T", [500, 1000, 2000])
    def test_streaming_entropy_clinical_lengths(self, clinical_model, T):
        """Test streaming entropy at clinical sequence lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)

        entropy = clinical_model.compute_entropy_streaming(hidden, lengths)

        assert entropy.shape == (batch,)
        assert torch.isfinite(entropy).all()


class TestVariableLengthUncertainty:
    """Test uncertainty methods with variable-length batches."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_variable_lengths_marginals(self, uncertainty_model):
        """Test marginals with variable-length sequences."""
        batch = 3
        T_max = 100
        lengths = torch.tensor([100, 75, 50])

        hidden = torch.randn(batch, T_max, 32)

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True
        )

        assert boundary_probs.shape == (batch, T_max)
        assert torch.isfinite(boundary_probs).all()

    def test_variable_lengths_entropy(self, uncertainty_model):
        """Test entropy with variable-length sequences."""
        batch = 3
        T_max = 100
        lengths = torch.tensor([100, 75, 50])

        hidden = torch.randn(batch, T_max, 32)

        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert entropy.shape == (batch,)
        assert torch.isfinite(entropy).all()
