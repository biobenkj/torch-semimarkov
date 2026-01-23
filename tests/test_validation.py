"""Tests for input validation utilities."""

import pytest
import torch

from torch_semimarkov.validation import (
    validate_cum_scores,
    validate_device_consistency,
    validate_hidden_states,
    validate_labels,
    validate_lengths,
)


class TestValidateHiddenStates:
    """Tests for validate_hidden_states."""

    def test_valid_3d_tensor(self):
        """Valid 3D tensor should pass."""
        hidden = torch.randn(2, 100, 64)
        validate_hidden_states(hidden)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        hidden = torch.randn(100, 64)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_hidden_states(hidden)

    def test_4d_tensor_raises(self):
        """4D tensor should raise ValueError."""
        hidden = torch.randn(2, 100, 64, 32)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_hidden_states(hidden)

    def test_nan_raises(self):
        """Tensor with NaN should raise ValueError."""
        hidden = torch.randn(2, 100, 64)
        hidden[0, 50, 32] = float("nan")
        with pytest.raises(ValueError, match="contains NaN"):
            validate_hidden_states(hidden)

    def test_inf_raises(self):
        """Tensor with Inf should raise ValueError."""
        hidden = torch.randn(2, 100, 64)
        hidden[1, 25, 16] = float("inf")
        with pytest.raises(ValueError, match="contains Inf"):
            validate_hidden_states(hidden)

    def test_nan_check_disabled(self):
        """NaN check can be disabled."""
        hidden = torch.randn(2, 100, 64)
        hidden[0, 50, 32] = float("nan")
        validate_hidden_states(hidden, check_nan=False)  # Should not raise

    def test_inf_check_disabled(self):
        """Inf check can be disabled."""
        hidden = torch.randn(2, 100, 64)
        hidden[1, 25, 16] = float("inf")
        validate_hidden_states(hidden, check_inf=False)  # Should not raise

    def test_custom_name_in_error(self):
        """Custom name should appear in error message."""
        hidden = torch.randn(100, 64)
        with pytest.raises(ValueError, match="encoder_output"):
            validate_hidden_states(hidden, name="encoder_output")


class TestValidateLengths:
    """Tests for validate_lengths."""

    def test_valid_lengths(self):
        """Valid lengths should pass."""
        lengths = torch.tensor([100, 100])
        validate_lengths(lengths, max_length=100)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        lengths = torch.tensor([[100, 100]])
        with pytest.raises(ValueError, match="must be 1D"):
            validate_lengths(lengths, max_length=100)

    def test_batch_size_mismatch_raises(self):
        """Wrong batch size should raise ValueError."""
        lengths = torch.tensor([100, 100])
        with pytest.raises(ValueError, match="batch size"):
            validate_lengths(lengths, max_length=100, batch_size=3)

    def test_zero_length_raises(self):
        """Zero length should raise ValueError."""
        lengths = torch.tensor([100, 0])
        with pytest.raises(ValueError, match="must be positive"):
            validate_lengths(lengths, max_length=100)

    def test_negative_length_raises(self):
        """Negative length should raise ValueError."""
        lengths = torch.tensor([100, -5])
        with pytest.raises(ValueError, match="must be positive"):
            validate_lengths(lengths, max_length=100)

    def test_exceeds_max_raises(self):
        """Length exceeding max should raise ValueError."""
        lengths = torch.tensor([100, 200])
        with pytest.raises(ValueError, match="cannot exceed T=100"):
            validate_lengths(lengths, max_length=100)


class TestValidateLabels:
    """Tests for validate_labels."""

    def test_valid_labels(self):
        """Valid labels should pass."""
        labels = torch.randint(0, 4, (2, 100))
        validate_labels(labels, num_classes=4)  # Should not raise

    def test_1d_tensor_raises(self):
        """1D tensor should raise ValueError."""
        labels = torch.randint(0, 4, (100,))
        with pytest.raises(ValueError, match="must be 2D"):
            validate_labels(labels, num_classes=4)

    def test_batch_size_mismatch_raises(self):
        """Wrong batch size should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        with pytest.raises(ValueError, match="batch size"):
            validate_labels(labels, num_classes=4, batch_size=3)

    def test_seq_length_mismatch_raises(self):
        """Wrong sequence length should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        with pytest.raises(ValueError, match="sequence length"):
            validate_labels(labels, num_classes=4, seq_length=50)

    def test_negative_label_raises(self):
        """Negative label should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        labels[0, 50] = -1
        with pytest.raises(ValueError, match="must be in"):
            validate_labels(labels, num_classes=4)

    def test_label_out_of_range_raises(self):
        """Label >= num_classes should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        labels[1, 25] = 4  # num_classes = 4, so 4 is out of range
        with pytest.raises(ValueError, match="must be in"):
            validate_labels(labels, num_classes=4)


class TestValidateCumScores:
    """Tests for validate_cum_scores."""

    def test_valid_cum_scores(self):
        """Valid cum_scores should pass."""
        cum_scores = torch.zeros(2, 101, 4)
        validate_cum_scores(cum_scores)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        cum_scores = torch.zeros(101, 4)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_cum_scores(cum_scores)

    def test_t_plus_1_too_small_raises(self):
        """T+1 < 2 should raise ValueError."""
        cum_scores = torch.zeros(2, 1, 4)  # T=0
        with pytest.raises(ValueError, match="T\\+1 dimension must be >= 2"):
            validate_cum_scores(cum_scores)

    def test_dtype_warning(self):
        """Non-float32 dtype should warn."""
        cum_scores = torch.zeros(2, 101, 4, dtype=torch.float16)
        with pytest.warns(UserWarning, match="should be float32"):
            validate_cum_scores(cum_scores)

    def test_dtype_warning_disabled(self):
        """dtype warning can be disabled."""
        cum_scores = torch.zeros(2, 101, 4, dtype=torch.float16)
        validate_cum_scores(cum_scores, warn_dtype=False)  # Should not warn


class TestValidateDeviceConsistency:
    """Tests for validate_device_consistency."""

    def test_same_device_passes(self):
        """Tensors on same device should pass."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        validate_device_consistency(t1, t2)  # Should not raise

    def test_single_tensor_passes(self):
        """Single tensor should pass."""
        t1 = torch.randn(2, 3)
        validate_device_consistency(t1)  # Should not raise

    def test_none_values_skipped(self):
        """None values should be skipped."""
        t1 = torch.randn(2, 3)
        validate_device_consistency(t1, None, t1)  # Should not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_raises(self):
        """Different devices should raise ValueError."""
        t_cpu = torch.randn(2, 3)
        t_cuda = torch.randn(2, 3, device="cuda")
        with pytest.raises(ValueError, match="Device mismatch"):
            validate_device_consistency(t_cpu, t_cuda)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_with_names(self):
        """Device mismatch should include names in error."""
        t_cpu = torch.randn(2, 3)
        t_cuda = torch.randn(2, 3, device="cuda")
        with pytest.raises(ValueError, match="hidden_states"):
            validate_device_consistency(t_cpu, t_cuda, names=["hidden_states", "lengths"])


class TestCRFHeadValidation:
    """Integration tests for validation in CRF heads."""

    def test_forward_validates_hidden_states_shape(self):
        """forward() should reject wrong hidden_states shape."""
        from torch_semimarkov import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(100, 64)  # Missing batch dim
        lengths = torch.tensor([100])

        with pytest.raises(ValueError, match="must be 3D"):
            crf.forward(hidden, lengths)

    def test_forward_validates_lengths_bounds(self):
        """forward() should reject lengths > T."""
        from torch_semimarkov import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(2, 100, 64)
        lengths = torch.tensor([100, 200])  # 200 > T=100

        with pytest.raises(ValueError, match="cannot exceed"):
            crf.forward(hidden, lengths)

    def test_compute_loss_validates_labels(self):
        """compute_loss() should reject out-of-range labels."""
        from torch_semimarkov import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(2, 100, 64)
        lengths = torch.tensor([100, 100])
        labels = torch.randint(0, 10, (2, 100))  # Labels go up to 9, but num_classes=4

        with pytest.raises(ValueError, match="must be in"):
            crf.compute_loss(hidden, lengths, labels)

    def test_decode_validates_inputs(self):
        """decode() should validate inputs."""
        from torch_semimarkov import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(100, 64)  # Missing batch dim
        lengths = torch.tensor([100])

        with pytest.raises(ValueError, match="must be 3D"):
            crf.decode(hidden, lengths)


class TestStreamingAPIValidation:
    """Tests for validation in streaming API."""

    def test_streaming_forward_validates_cum_scores(self):
        """semi_crf_streaming_forward() should validate cum_scores."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 1, 4)  # T=0, invalid
        transition = torch.randn(4, 4)
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([1, 1])

        with pytest.raises(ValueError, match="T\\+1 dimension must be >= 2"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    def test_streaming_forward_validates_lengths(self):
        """semi_crf_streaming_forward() should validate lengths."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4)  # T=100
        transition = torch.randn(4, 4)
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([100, 200])  # 200 > T=100

        with pytest.raises(ValueError, match="cannot exceed"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)
