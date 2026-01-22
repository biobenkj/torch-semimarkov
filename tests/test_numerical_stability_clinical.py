"""Tests for numerical stability at clinical sequence lengths.

Tests verify:
- Float32 cumsum precision at long sequence lengths
- Variable-length batch handling
- Extreme score values don't cause overflow/underflow
- Gradient magnitude stability across different sequence lengths
- Zero-centering before cumsum prevents precision loss
"""

import pytest
import torch

from torch_semimarkov import SemiMarkovCRFHead, UncertaintySemiMarkovCRFHead


class TestCumsumPrecision:
    """Test cumulative sum precision at clinical sequence lengths."""

    @pytest.fixture
    def clinical_model(self):
        return SemiMarkovCRFHead(
            num_classes=8,
            max_duration=32,
            hidden_dim=64,
        )

    @pytest.mark.parametrize("T", [100, 500, 1000, 2000, 5000])
    def test_partition_finite_across_lengths(self, clinical_model, T):
        """Test partition function is finite across clinical lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)

        result = clinical_model(hidden, lengths, use_triton=False)
        partition = result["partition"]

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all()

    @pytest.mark.parametrize("T", [100, 500, 1000, 2000, 5000])
    def test_loss_finite_across_lengths(self, clinical_model, T):
        """Test loss is finite across clinical lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 8, (batch, T))

        loss = clinical_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_cumsum_precision_long_sequence(self, clinical_model):
        """Test that cumsum maintains precision for long sequences."""
        batch, T, C = 1, 5000, 8

        # Create scores with known sum to verify precision
        scores = torch.randn(batch, T, C) * 0.1

        # Compute cumulative scores in float32 (as done in the module)
        cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Verify the final cumsum matches direct sum
        direct_sum = scores.sum(dim=1)
        cumsum_final = cum_scores[:, -1]

        # Allow some numerical tolerance but should be reasonably close
        torch.testing.assert_close(direct_sum, cumsum_final, rtol=1e-3, atol=1e-3)

    def test_zero_centering_improves_precision(self):
        """Test that zero-centering before cumsum improves numerical stability."""
        batch, T, C = 1, 5000, 8

        # Create scores with large mean
        scores = torch.randn(batch, T, C) + 100.0  # Large offset

        # Without zero-centering
        cum_no_center = torch.cumsum(scores.float(), dim=1)

        # With zero-centering
        scores_centered = scores - scores.mean(dim=1, keepdim=True)
        cum_centered = torch.cumsum(scores_centered.float(), dim=1)

        # Both should be finite
        assert torch.isfinite(cum_no_center).all()
        assert torch.isfinite(cum_centered).all()

        # Centered version should have smaller values (more stable)
        assert cum_centered.abs().max() < cum_no_center.abs().max()


class TestVariableLengthBatches:
    """Test handling of variable-length batches."""

    @pytest.fixture
    def clinical_model(self):
        return SemiMarkovCRFHead(
            num_classes=8,
            max_duration=32,
            hidden_dim=64,
        )

    def test_variable_lengths_small_range(self, clinical_model):
        """Test with variable lengths in small range."""
        batch = 4
        T_max = 200
        lengths = torch.tensor([200, 180, 160, 140])

        hidden = torch.randn(batch, T_max, 64)
        labels = torch.randint(0, 8, (batch, T_max))

        loss = clinical_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    @pytest.mark.slow
    def test_variable_lengths_large_range(self, clinical_model):
        """Test with variable lengths spanning large range."""
        batch = 4
        T_max = 1000
        lengths = torch.tensor([1000, 500, 250, 100])

        hidden = torch.randn(batch, T_max, 64)
        labels = torch.randint(0, 8, (batch, T_max))

        loss = clinical_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_variable_lengths_partition_per_sample(self, clinical_model):
        """Test partition is computed correctly per sample with variable lengths."""
        batch = 3
        T_max = 500
        lengths = torch.tensor([500, 300, 100])

        hidden = torch.randn(batch, T_max, 64)

        result = clinical_model(hidden, lengths, use_triton=False)
        partition = result["partition"]

        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all()

        # Each sample's partition should be different
        # (extremely unlikely to be equal with random inputs)
        assert not torch.allclose(partition[0], partition[1])
        assert not torch.allclose(partition[1], partition[2])

    @pytest.mark.slow
    def test_extreme_length_ratio(self, clinical_model):
        """Test with extreme length ratios in batch."""
        batch = 3
        T_max = 2000
        lengths = torch.tensor([2000, 100, 50])

        hidden = torch.randn(batch, T_max, 64)
        labels = torch.randint(0, 8, (batch, T_max))

        loss = clinical_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)


class TestExtremeScoreValues:
    """Test numerical stability with extreme input values."""

    @pytest.fixture
    def model(self):
        return SemiMarkovCRFHead(
            num_classes=4,
            max_duration=16,
            hidden_dim=32,
        )

    def test_small_score_values(self, model):
        """Test with very small score values."""
        batch, T = 2, 100
        hidden = torch.randn(batch, T, 32) * 0.001  # Very small values
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_large_score_values(self, model):
        """Test with large score values."""
        batch, T = 2, 100
        hidden = torch.randn(batch, T, 32) * 10.0  # Larger values
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_mixed_scale_values(self, model):
        """Test with mixed scale values in batch."""
        batch, T = 4, 100
        hidden = torch.randn(batch, T, 32)
        # Different scales per batch element
        hidden[0] *= 0.001
        hidden[1] *= 0.1
        hidden[2] *= 1.0
        hidden[3] *= 10.0

        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_high_contrast_input(self, model):
        """Test with high-contrast input (some features very large/small)."""
        batch, T = 2, 100
        hidden = torch.zeros(batch, T, 32)
        hidden[:, :, 0] = 10.0  # One feature very large
        hidden[:, :, 1] = -10.0  # One feature very negative
        hidden[:, :, 2:] = torch.randn(batch, T, 30) * 0.1  # Rest small

        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_sparse_input(self, model):
        """Test with sparse-like input (mostly zeros)."""
        batch, T = 2, 100
        hidden = torch.zeros(batch, T, 32)
        # Set only ~10% of values
        mask = torch.rand(batch, T, 32) < 0.1
        hidden[mask] = torch.randn(mask.sum())

        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)


class TestGradientStability:
    """Test gradient stability across sequence lengths."""

    @pytest.fixture
    def model(self):
        return SemiMarkovCRFHead(
            num_classes=4,
            max_duration=16,
            hidden_dim=32,
        )

    @pytest.mark.parametrize("T", [50, 100, 500, 1000])
    def test_gradients_finite_across_lengths(self, model, T):
        """Test that gradients are finite across clinical lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 32, requires_grad=True)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        assert torch.isfinite(hidden.grad).all()

    @pytest.mark.parametrize("T", [50, 100, 500, 1000])
    def test_gradient_magnitude_reasonable(self, model, T):
        """Test that gradient magnitudes are reasonable (not exploding/vanishing)."""
        torch.manual_seed(42)
        batch = 2
        hidden = torch.randn(batch, T, 32, requires_grad=True)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        grad_norm = hidden.grad.norm()

        # Gradient should not be zero (vanishing)
        assert grad_norm > 1e-10

        # Gradient should not be extremely large (exploding)
        # Allow some scaling with T but should not be unreasonable
        assert grad_norm < 1e6

    def test_gradient_consistency_across_lengths(self, model):
        """Test that gradient statistics are consistent across different lengths."""
        torch.manual_seed(42)
        batch = 2

        grad_means = []
        grad_stds = []

        for T in [100, 200, 500]:
            hidden = torch.randn(batch, T, 32, requires_grad=True)
            lengths = torch.full((batch,), T)
            labels = torch.randint(0, 4, (batch, T))

            loss = model.compute_loss(hidden, lengths, labels, use_triton=False)
            loss.backward()

            # Normalize by sequence length for comparison
            grad_per_position = hidden.grad / T
            grad_means.append(grad_per_position.abs().mean().item())
            grad_stds.append(grad_per_position.std().item())

        # Per-position gradient statistics should be similar across lengths
        # (within an order of magnitude)
        max_mean = max(grad_means)
        min_mean = min(grad_means)
        assert max_mean < min_mean * 100  # Within 2 orders of magnitude

    def test_parameter_gradients_finite(self, model):
        """Test that model parameter gradients are finite."""
        batch, T = 2, 500
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        # Check all parameter gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


class TestLogsumexpStability:
    """Test logsumexp stability in edge cases."""

    @pytest.fixture
    def model(self):
        return SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=32,
        )

    def test_uniform_scores(self, model):
        """Test with uniform scores (high entropy case)."""
        batch, T = 2, 100
        # All scores equal → logsumexp over many equal values
        hidden = torch.ones(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_peaked_scores(self, model):
        """Test with peaked scores (low entropy case)."""
        batch, T = 2, 100
        hidden = torch.zeros(batch, T, 32)
        # Make one class much more likely
        hidden[:, :, 0] = 10.0

        lengths = torch.full((batch,), T)
        labels = torch.zeros(batch, T, dtype=torch.long)

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_alternating_extremes(self, model):
        """Test with alternating extreme values."""
        batch, T = 2, 100
        hidden = torch.zeros(batch, T, 32)
        # Alternate between very positive and very negative
        hidden[:, 0::2, :16] = 5.0
        hidden[:, 1::2, :16] = -5.0
        hidden[:, 0::2, 16:] = -5.0
        hidden[:, 1::2, 16:] = 5.0

        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)


class TestClinicalDomainScales:
    """Test numerical stability at realistic clinical data scales."""

    @pytest.fixture
    def ecg_model(self):
        """ECG arrhythmia model (5 classes, 250Hz, 10s-60s)."""
        return SemiMarkovCRFHead(
            num_classes=5,
            max_duration=100,  # ~400ms at 250Hz
            hidden_dim=64,
        )

    @pytest.fixture
    def eeg_model(self):
        """EEG sleep staging model (5 classes, low feature rate)."""
        return SemiMarkovCRFHead(
            num_classes=5,
            max_duration=300,  # 30s epochs at 10Hz
            hidden_dim=128,
        )

    @pytest.fixture
    def genomics_model(self):
        """Genomics segmentation model (many classes, moderate max_duration).

        Note: max_duration is reduced from typical genomics scale (3000+) to 100
        for testing feasibility. Full-scale genomics would use GPU with Triton.
        """
        return SemiMarkovCRFHead(
            num_classes=24,
            max_duration=100,  # Reduced for CPU testing
            hidden_dim=256,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("T", [2500, 7500])  # 10s, 30s at 250Hz
    def test_ecg_scale(self, ecg_model, T):
        """Test at ECG recording scales."""
        batch = 2
        hidden = torch.randn(batch, T, 64)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 5, (batch, T))

        loss = ecg_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    @pytest.mark.slow
    @pytest.mark.parametrize("T", [2400, 4800])  # 4h, 8h at 10Hz feature rate
    def test_eeg_scale(self, eeg_model, T):
        """Test at EEG sleep study scales."""
        batch = 2
        hidden = torch.randn(batch, T, 128)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 5, (batch, T))

        loss = eeg_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    @pytest.mark.slow
    def test_genomics_moderate_scale(self, genomics_model):
        """Test at moderate genomics scale."""
        batch = 2
        T = 5000  # Moderate genomics window
        hidden = torch.randn(batch, T, 256)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 24, (batch, T))

        loss = genomics_model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)


class TestUncertaintyNumericalStability:
    """Test numerical stability of uncertainty quantification methods."""

    @pytest.fixture
    def uncertainty_model(self):
        return UncertaintySemiMarkovCRFHead(
            num_classes=4,
            max_duration=16,
            hidden_dim=32,
        )

    @pytest.mark.parametrize("T", [100, 500, 1000])
    def test_marginals_finite_across_lengths(self, uncertainty_model, T):
        """Test boundary marginals are finite across clinical lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        boundary_probs = uncertainty_model.compute_boundary_marginals(
            hidden, lengths, use_streaming=True
        )

        assert torch.isfinite(boundary_probs).all()

    @pytest.mark.parametrize("T", [100, 500, 1000])
    def test_entropy_finite_across_lengths(self, uncertainty_model, T):
        """Test entropy is finite across clinical lengths."""
        batch = 2
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)

        entropy = uncertainty_model.compute_entropy_streaming(hidden, lengths)

        assert torch.isfinite(entropy).all()

    def test_weighted_loss_stable_clinical_length(self, uncertainty_model):
        """Test weighted loss is stable at clinical length."""
        batch, T = 2, 500
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = uncertainty_model.compute_loss_uncertainty_weighted(
            hidden, lengths, labels, use_triton=False
        )

        assert torch.isfinite(loss)


class TestEdgeCases:
    """Test numerical stability in edge cases."""

    @pytest.fixture
    def model(self):
        return SemiMarkovCRFHead(
            num_classes=4,
            max_duration=16,
            hidden_dim=32,
        )

    def test_minimum_sequence_length(self, model):
        """Test with minimum viable sequence length."""
        batch, T = 2, 17  # Just longer than max_duration
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_single_segment_sequence(self, model):
        """Test with sequence that could be single segment."""
        batch, T = 2, 16  # Exactly max_duration
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.zeros(batch, T, dtype=torch.long)  # Single class

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_many_short_segments(self, model):
        """Test with many short segments (high boundary count)."""
        batch, T = 2, 200
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        # Alternate labels every 2 positions → many boundaries
        labels = torch.zeros(batch, T, dtype=torch.long)
        for i in range(T):
            labels[:, i] = (i // 2) % 4

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_few_long_segments(self, model):
        """Test with few long segments."""
        batch, T = 2, 200
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        # Only 2 segments (change in middle)
        labels = torch.zeros(batch, T, dtype=torch.long)
        labels[:, T // 2 :] = 1

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_batch_size_one(self, model):
        """Test with batch size 1."""
        batch, T = 1, 500
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)

    def test_large_batch_size(self, model):
        """Test with larger batch size."""
        batch, T = 16, 100
        hidden = torch.randn(batch, T, 32)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, 4, (batch, T))

        loss = model.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)
