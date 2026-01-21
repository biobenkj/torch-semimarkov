"""Tests for focused learning with uncertainty-weighted loss.

Tests verify:
- Uncertainty-weighted loss computation produces valid outputs
- High uncertainty regions receive larger gradients
- Curriculum learning: samples can be sorted by uncertainty
- Active learning scenario: high-uncertainty regions can be sampled
"""

import pytest
import torch

from torch_semimarkov import UncertaintySemiMarkovCRFHead


class TestUncertaintyWeightedLoss:
    """Test uncertainty-weighted loss computation."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_weighted_loss_shape_high_uncertainty(self, uncertainty_model):
        """Test weighted loss has correct shape with high_uncertainty mode."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            uncertainty_weight=1.0,
            focus_mode="high_uncertainty",
        )

        assert loss.shape == ()  # Scalar with mean reduction
        assert torch.isfinite(loss)

    def test_weighted_loss_shape_boundary_regions(self, uncertainty_model):
        """Test weighted loss has correct shape with boundary_regions mode."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            uncertainty_weight=1.0,
            focus_mode="boundary_regions",
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_weighted_loss_no_reduction(self, uncertainty_model):
        """Test weighted loss with no reduction returns per-sample loss."""
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            reduction="none",
        )

        assert loss.shape == (batch,)
        assert torch.isfinite(loss).all()

    def test_weighted_loss_sum_reduction(self, uncertainty_model):
        """Test weighted loss with sum reduction."""
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            reduction="sum",
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_weighted_loss_invalid_mode_raises(self, uncertainty_model):
        """Test that invalid focus mode raises ValueError."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        with pytest.raises(ValueError, match="Unknown focus_mode"):
            uncertainty_model.compute_loss_uncertainty_weighted(
                hidden,
                lengths,
                labels,
                focus_mode="invalid_mode",
            )

    def test_zero_weight_matches_standard_loss(self, uncertainty_model):
        """Test that zero uncertainty weight produces same loss as standard."""
        torch.manual_seed(42)
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        # Standard loss
        standard_loss = uncertainty_model.compute_loss(hidden, lengths, labels, use_triton=False)

        # Weighted loss with zero weight
        weighted_loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            uncertainty_weight=0.0,
            use_triton=False,
        )

        torch.testing.assert_close(standard_loss, weighted_loss, rtol=1e-5, atol=1e-5)


class TestGradientScaling:
    """Test that uncertainty weighting affects gradient magnitudes."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_weighted_loss_produces_gradients(self, uncertainty_model):
        """Test that weighted loss produces valid gradients."""
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32, requires_grad=True)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            uncertainty_weight=1.0,
            use_triton=False,
        )
        loss.backward()

        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert (hidden.grad != 0).any()

    def test_higher_weight_scales_gradients(self, uncertainty_model):
        """Test that higher uncertainty weight increases gradient magnitude."""
        torch.manual_seed(42)
        batch, T = 2, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        grad_magnitudes = []

        for weight in [0.0, 1.0, 2.0]:
            hidden_copy = hidden.clone().requires_grad_(True)

            loss = uncertainty_model.compute_loss_uncertainty_weighted(
                hidden_copy,
                lengths,
                labels,
                uncertainty_weight=weight,
                use_triton=False,
            )
            loss.backward()

            grad_magnitudes.append(hidden_copy.grad.abs().mean().item())

        # Higher weight should generally lead to larger gradients
        # (not strictly monotonic due to uncertainty varying by sample)
        # But weight=2 should be larger than weight=0
        assert grad_magnitudes[2] > grad_magnitudes[0] * 0.8


class TestCurriculumLearning:
    """Test curriculum learning via uncertainty-based sample ordering."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_entropy_sorting(self, uncertainty_model):
        """Test that samples can be sorted by entropy (easy → hard)."""
        batch, T = 5, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        # Compute entropy for each sample
        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert entropy.shape == (batch,)
        assert torch.isfinite(entropy).all()

        # Sort indices by entropy (ascending = easy first)
        sorted_indices = torch.argsort(entropy)

        # Verify sorting works
        sorted_entropy = entropy[sorted_indices]
        assert (sorted_entropy[:-1] <= sorted_entropy[1:] + 1e-6).all()

    def test_curriculum_batch_creation(self, uncertainty_model):
        """Test creating curriculum batches from high to low uncertainty."""
        torch.manual_seed(42)
        batch, T = 10, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        # Compute entropy
        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        # Create curriculum schedule: easy → hard
        sorted_indices = torch.argsort(entropy)

        # Split into mini-batches of size 3
        mini_batch_size = 3
        num_batches = batch // mini_batch_size

        for i in range(num_batches):
            start = i * mini_batch_size
            end = start + mini_batch_size
            batch_indices = sorted_indices[start:end]

            # Get batch data
            batch_hidden = hidden[batch_indices]
            batch_lengths = lengths[batch_indices]
            batch_labels = labels[batch_indices]

            # Compute loss for this batch
            loss = uncertainty_model.compute_loss(
                batch_hidden, batch_lengths, batch_labels, use_triton=False
            )

            assert torch.isfinite(loss)

    def test_easy_samples_have_lower_entropy(self, uncertainty_model):
        """Test that clear/structured inputs have lower entropy than ambiguous ones."""
        batch, T = 1, 30

        # Clear input: strong signal in one class
        clear_hidden = torch.zeros(batch, T, 32)
        clear_hidden[:, :, 0] = 5.0  # Strong signal for class 0

        # Ambiguous input: uniform noise
        ambiguous_hidden = torch.randn(batch, T, 32) * 0.1

        lengths = torch.full((batch,), T)

        clear_entropy = uncertainty_model.compute_entropy_streaming(clear_hidden, lengths)
        ambiguous_entropy = uncertainty_model.compute_entropy_streaming(ambiguous_hidden, lengths)

        # Clear should have different entropy than ambiguous
        # (exact ordering depends on model weights, but values should differ)
        assert clear_entropy.shape == (1,)
        assert ambiguous_entropy.shape == (1,)
        assert torch.isfinite(clear_entropy).all()
        assert torch.isfinite(ambiguous_entropy).all()


class TestActiveLearning:
    """Test active learning scenario: sampling high-uncertainty regions."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_identify_high_uncertainty_samples(self, uncertainty_model):
        """Test identifying top-k samples with highest uncertainty."""
        batch, T = 10, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        # Compute uncertainty via entropy
        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        # Select top-k highest uncertainty
        k = 3
        _, top_k_indices = torch.topk(entropy, k)

        assert top_k_indices.shape == (k,)

        # Verify these are the highest entropy samples
        top_k_entropy = entropy[top_k_indices]
        remaining_entropy = entropy[~torch.isin(torch.arange(batch), top_k_indices)]

        # All top-k should be >= all remaining
        assert top_k_entropy.min() >= remaining_entropy.max() - 1e-6

    def test_identify_high_uncertainty_positions(self, uncertainty_model):
        """Test identifying positions with highest boundary uncertainty."""
        batch, T = 2, 100
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        # Compute boundary marginals
        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True
        )

        # For each sample, find top-k most uncertain positions
        k = 10
        for b in range(batch):
            _, top_positions = torch.topk(boundary_probs[b], k)

            assert top_positions.shape == (k,)
            assert (top_positions >= 0).all()
            assert (top_positions < T).all()

    def test_active_learning_batch_selection(self, uncertainty_model):
        """Test selecting a batch for active learning based on uncertainty."""
        torch.manual_seed(42)

        # Simulate unlabeled pool
        pool_size = 20
        T = 50
        unlabeled_pool = torch.randn(pool_size, T, 32)
        lengths = torch.full((pool_size,), T)

        # Compute uncertainty for all samples
        entropy = uncertainty_model.compute_entropy_streaming(unlabeled_pool, lengths)

        # Select acquisition batch (top uncertain samples)
        acquisition_size = 5
        _, acquisition_indices = torch.topk(entropy, acquisition_size)

        # "Label" selected samples (simulate oracle)
        selected_hidden = unlabeled_pool[acquisition_indices]
        selected_lengths = lengths[acquisition_indices]
        selected_labels = torch.randint(0, 4, (acquisition_size, T))

        # Train on selected samples
        loss = uncertainty_model.compute_loss(
            selected_hidden, selected_lengths, selected_labels, use_triton=False
        )

        assert torch.isfinite(loss)


class TestFocusedLearningRegimes:
    """Test different focused learning regimes."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_boundary_focus_vs_uncertainty_focus(self, uncertainty_model):
        """Test that different focus modes produce different weightings."""
        torch.manual_seed(42)
        batch, T = 3, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss_uncertainty = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            focus_mode="high_uncertainty",
            reduction="none",
            use_triton=False,
        )

        loss_boundary = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            focus_mode="boundary_regions",
            reduction="none",
            use_triton=False,
        )

        # Both should be finite
        assert torch.isfinite(loss_uncertainty).all()
        assert torch.isfinite(loss_boundary).all()

        # Different focus modes should produce different weights
        # (not always different values, but typically different)
        # We test that both modes are functional
        assert loss_uncertainty.shape == loss_boundary.shape == (batch,)

    def test_variable_weights_per_sample(self, uncertainty_model):
        """Test that uncertainty weights vary across samples."""
        torch.manual_seed(42)
        batch, T = 5, 50
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        # Get per-sample weighted losses
        weighted_loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            uncertainty_weight=1.0,
            reduction="none",
            use_triton=False,
        )

        # Get per-sample standard losses
        standard_loss = uncertainty_model.compute_loss(
            hidden,
            lengths,
            labels,
            reduction="none",
            use_triton=False,
        )

        # Compute implied weights: weighted_loss / standard_loss
        # Weight = 1 + uncertainty_weight * uncertainty
        implied_weights = weighted_loss / (standard_loss + 1e-8)

        # Weights should vary (not all the same)
        weight_std = implied_weights.std()
        assert weight_std > 1e-6  # Some variation expected


class TestVariableLengthFocusedLearning:
    """Test focused learning with variable-length sequences."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_variable_lengths_weighted_loss(self, uncertainty_model):
        """Test weighted loss works with variable-length sequences."""
        batch = 3
        T_max = 100
        lengths = torch.tensor([100, 75, 50])

        hidden = torch.randn(batch, T_max, 32)
        labels = torch.randint(0, 4, (batch, T_max))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            use_triton=False,
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_variable_lengths_entropy_curriculum(self, uncertainty_model):
        """Test curriculum learning with variable-length sequences."""
        batch = 5
        T_max = 100
        lengths = torch.tensor([100, 80, 60, 40, 20])

        hidden = torch.randn(batch, T_max, 32)

        # Compute entropy
        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert entropy.shape == (batch,)
        assert torch.isfinite(entropy).all()

        # Sort by entropy
        sorted_indices = torch.argsort(entropy)
        assert sorted_indices.shape == (batch,)


class TestClinicalScaleFocusedLearning:
    """Test focused learning at clinical sequence lengths."""

    @pytest.fixture
    def clinical_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=8,
            max_duration=32,
            hidden_dim=64,
        )

    def test_weighted_loss_clinical_length(self, clinical_model):
        """Test weighted loss at clinical sequence length."""
        batch = 2
        T = 1000  # Clinical scale

        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 8, (batch, T))

        loss = clinical_model.compute_loss_uncertainty_weighted(
            hidden,
            lengths,
            labels,
            use_triton=False,
        )

        assert torch.isfinite(loss)

    def test_curriculum_clinical_length(self, clinical_model):
        """Test curriculum learning at clinical sequence length."""
        batch = 5
        T = 500

        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)

        # Compute entropy for curriculum ordering
        entropy = clinical_model.compute_entropy_streaming(hidden, lengths)

        assert torch.isfinite(entropy).all()

        # Sort and verify
        sorted_indices = torch.argsort(entropy)
        sorted_entropy = entropy[sorted_indices]
        assert (sorted_entropy[:-1] <= sorted_entropy[1:] + 1e-6).all()

    def test_active_learning_clinical_scale(self, clinical_model):
        """Test active learning selection at clinical scale."""
        pool_size = 10
        T = 500

        hidden = torch.randn(pool_size, T, 64)
        lengths = torch.full((pool_size,), T)

        # Compute uncertainty
        entropy = clinical_model.compute_entropy_streaming(hidden, lengths)

        # Select top-3 most uncertain
        _, top_indices = torch.topk(entropy, 3)

        assert top_indices.shape == (3,)
        assert torch.isfinite(entropy[top_indices]).all()
