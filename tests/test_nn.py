"""Tests for SemiMarkovCRFHead nn.Module."""

import pytest
import torch

from torch_semimarkov import SemiMarkovCRFHead, semi_crf_streaming_forward


class TestSemiMarkovCRFHead:
    """Tests for SemiMarkovCRFHead module."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for fast tests."""
        return {
            "num_classes": 4,
            "max_duration": 8,
            "hidden_dim": 16,
            "batch": 2,
            "T": 20,
        }

    @pytest.fixture
    def crf_head(self, small_config):
        """Create a CRF head for testing."""
        return SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
            hidden_dim=small_config["hidden_dim"],
        )

    def test_init_with_hidden_dim(self, small_config):
        """Test initialization with projection layer."""
        crf = SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
            hidden_dim=small_config["hidden_dim"],
        )

        assert crf.projection is not None
        assert crf.projection.in_features == small_config["hidden_dim"]
        assert crf.projection.out_features == small_config["num_classes"]
        assert crf.transition.shape == (
            small_config["num_classes"],
            small_config["num_classes"],
        )
        assert crf.duration_bias.shape == (
            small_config["max_duration"],
            small_config["num_classes"],
        )

    def test_init_without_hidden_dim(self, small_config):
        """Test initialization without projection layer."""
        crf = SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
        )

        assert crf.projection is None

    def test_forward_shape(self, crf_head, small_config):
        """Test forward pass output shapes."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)

        result = crf_head(hidden_states, lengths, use_triton=False)

        assert "partition" in result
        assert "cum_scores" in result
        assert result["partition"].shape == (batch,)
        assert result["cum_scores"].shape == (batch, T + 1, num_classes)

    def test_forward_matches_streaming_api(self, small_config):
        """Test that forward matches direct call to semi_crf_streaming_forward."""
        num_classes = small_config["num_classes"]
        max_duration = small_config["max_duration"]
        batch = small_config["batch"]
        T = small_config["T"]

        # Create CRF without projection (input is already num_classes)
        crf = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
        )

        # Input scores (directly in label space)
        scores = torch.randn(batch, T, num_classes)
        lengths = torch.full((batch,), T)

        # Forward via CRFHead
        result = crf(scores, lengths, use_triton=False)

        # Manual computation
        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        partition_manual = semi_crf_streaming_forward(
            cum_scores,
            crf.transition,
            crf.duration_bias,
            lengths,
            max_duration,
            use_triton=False,
        )

        torch.testing.assert_close(result["partition"], partition_manual)

    def test_compute_loss_shape(self, crf_head, small_config):
        """Test compute_loss output shape."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        # Mean reduction (default)
        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert loss.shape == ()  # Scalar

        # Sum reduction
        loss_sum = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="sum"
        )
        assert loss_sum.shape == ()

        # No reduction
        loss_none = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="none"
        )
        assert loss_none.shape == (batch,)

    def test_compute_loss_positive(self, crf_head, small_config):
        """Test that NLL loss is positive (partition >= gold_score)."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        loss = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="none"
        )

        # NLL should be non-negative (partition >= any single path score)
        assert (loss >= -1e-5).all(), f"NLL should be non-negative, got {loss}"

    def test_gradients_flow(self, crf_head, small_config):
        """Test that gradients flow through compute_loss."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim, requires_grad=True)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert crf_head.transition.grad is not None
        assert crf_head.duration_bias.grad is not None
        assert crf_head.projection.weight.grad is not None

    def test_score_gold_single_segment(self, small_config):
        """Test gold scoring with single segment (all same label)."""
        num_classes = small_config["num_classes"]
        max_duration = small_config["max_duration"]
        batch = 1
        T = 5

        crf = SemiMarkovCRFHead(num_classes=num_classes, max_duration=max_duration)

        # All zeros label
        labels = torch.zeros(batch, T, dtype=torch.long)
        lengths = torch.tensor([T])

        # Simple scores: all 1.0 for label 0
        scores = torch.zeros(batch, T, num_classes)
        scores[:, :, 0] = 1.0

        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        gold_score = crf._score_gold(cum_scores, labels, lengths)

        # Expected: content = T * 1.0 = 5.0, duration_bias[5, 0], no transitions
        # (duration_bias[k] stores bias for segments of duration k)
        expected_content = 5.0
        expected_duration = crf.duration_bias[T, 0].item()  # duration=5 uses index 5
        expected = expected_content + expected_duration

        assert abs(gold_score.item() - expected) < 1e-5

    def test_score_gold_multiple_segments(self, small_config):
        """Test gold scoring with multiple segments."""
        num_classes = 4
        max_duration = 10
        batch = 1
        T = 6

        crf = SemiMarkovCRFHead(num_classes=num_classes, max_duration=max_duration)

        # Labels: [0, 0, 1, 1, 1, 2] -> segments: (0-1, label 0), (2-4, label 1), (5, label 2)
        labels = torch.tensor([[0, 0, 1, 1, 1, 2]])
        lengths = torch.tensor([T])

        # Simple scores
        scores = torch.ones(batch, T, num_classes)
        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        gold_score = crf._score_gold(cum_scores, labels, lengths)

        # Expected components (duration_bias[k] stores bias for segments of duration k):
        # Segment 1 (0-1, label 0, dur=2): content = 2*1 = 2, dur_bias[2, 0]
        # Segment 2 (2-4, label 1, dur=3): content = 3*1 = 3, dur_bias[3, 1], trans[0, 1]
        # Segment 3 (5-5, label 2, dur=1): content = 1*1 = 1, dur_bias[1, 2], trans[1, 2]
        expected_content = 2 + 3 + 1
        expected_duration = (
            crf.duration_bias[2, 0].item()  # dur=2 uses index 2
            + crf.duration_bias[3, 1].item()  # dur=3 uses index 3
            + crf.duration_bias[1, 2].item()  # dur=1 uses index 1
        )
        expected_transition = crf.transition[0, 1].item() + crf.transition[1, 2].item()
        expected = expected_content + expected_duration + expected_transition

        assert abs(gold_score.item() - expected) < 1e-5

    def test_decode_shape(self, crf_head, small_config):
        """Test decode output shape."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)

        max_score = crf_head.decode(hidden_states, lengths, use_triton=False)

        assert max_score.shape == (batch,)

    def test_extra_repr(self, crf_head, small_config):
        """Test string representation."""
        repr_str = crf_head.extra_repr()
        assert f"num_classes={small_config['num_classes']}" in repr_str
        assert f"max_duration={small_config['max_duration']}" in repr_str
        assert f"hidden_dim={small_config['hidden_dim']}" in repr_str

    def test_variable_lengths(self, crf_head, small_config):
        """Test with variable sequence lengths."""
        batch = 3
        T_max = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T_max, hidden_dim)
        lengths = torch.tensor([T_max, T_max // 2, T_max // 4])
        labels = torch.randint(0, num_classes, (batch, T_max))

        # Should not error
        result = crf_head(hidden_states, lengths, use_triton=False)
        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)

        assert result["partition"].shape == (batch,)
        assert loss.shape == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSemiMarkovCRFHeadGPU:
    """GPU-specific tests for SemiMarkovCRFHead."""

    @pytest.fixture
    def gpu_config(self):
        """GPU configuration."""
        return {
            "num_classes": 8,
            "max_duration": 16,
            "hidden_dim": 32,
            "batch": 4,
            "T": 100,
        }

    def test_forward_gpu(self, gpu_config):
        """Test forward pass on GPU."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()

        result = crf(hidden_states, lengths, use_triton=True)

        assert result["partition"].device.type == "cuda"
        assert result["partition"].shape == (gpu_config["batch"],)

    def test_compute_loss_gpu(self, gpu_config):
        """Test compute_loss on GPU with Triton kernels."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()
        labels = torch.randint(
            0, gpu_config["num_classes"], (gpu_config["batch"], gpu_config["T"])
        ).cuda()

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=True)

        assert loss.device.type == "cuda"
        assert loss >= 0  # NLL should be non-negative

    def test_gradients_gpu(self, gpu_config):
        """Test gradient computation on GPU."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"],
            gpu_config["T"],
            gpu_config["hidden_dim"],
            requires_grad=True,
            device="cuda",
        )
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()
        labels = torch.randint(
            0, gpu_config["num_classes"], (gpu_config["batch"], gpu_config["T"])
        ).cuda()

        # Retain grad since hidden_states becomes non-leaf after projection in compute_loss
        hidden_states.retain_grad()

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=True)
        loss.backward()

        assert hidden_states.grad is not None
        assert crf.transition.grad is not None
        assert crf.duration_bias.grad is not None

    def test_triton_vs_pytorch(self, gpu_config):
        """Test that Triton and PyTorch implementations match."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()

        # Forward pass
        result_triton = crf(hidden_states, lengths, use_triton=True)
        result_pytorch = crf(hidden_states, lengths, use_triton=False)

        torch.testing.assert_close(
            result_triton["partition"],
            result_pytorch["partition"],
            rtol=1e-4,
            atol=1e-4,
        )
