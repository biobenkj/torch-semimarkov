import pytest
import torch

from torch_semimarkov.triton_scan import (
    semi_crf_forward_pytorch,
    semi_crf_triton_forward,
)

# =============================================================================
# Log Semiring Tests (existing)
# =============================================================================


def test_triton_forward_cpu_fallback_matches_pytorch():
    torch.manual_seed(0)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, use_triton=True)
    ref = semi_crf_forward_pytorch(edge.detach(), lengths)

    assert torch.allclose(out, ref, atol=1e-6)

    out.sum().backward()
    assert edge.grad is not None
    assert torch.isfinite(edge.grad).all()


def test_triton_validate_mode_matches_double():
    torch.manual_seed(1)
    batch, T, K, C = 1, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, validate=True)
    ref = semi_crf_forward_pytorch(edge.double(), lengths).to(edge.dtype)

    assert torch.allclose(out, ref, atol=1e-6)


# =============================================================================
# Max Semiring Tests
# =============================================================================


def test_max_semiring_pytorch_matches_semimarkov():
    """Test that max semiring PyTorch implementation matches SemiMarkov with MaxSemiring."""
    from torch_semimarkov import SemiMarkov
    from torch_semimarkov.semirings import MaxSemiring

    torch.manual_seed(42)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference: SemiMarkov with MaxSemiring
    model = SemiMarkov(MaxSemiring)
    ref, _ = model.logpartition(edge, lengths=lengths)

    # Test: triton_scan with max semiring
    out = semi_crf_forward_pytorch(edge, lengths, semiring="max")

    assert torch.allclose(out, ref, atol=1e-5), f"max diff: {(out - ref).abs().max()}"


def test_max_semiring_triton_forward_matches_pytorch():
    """Test that max semiring triton forward matches PyTorch reference."""
    torch.manual_seed(0)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, use_triton=True, semiring="max")
    ref = semi_crf_forward_pytorch(edge.detach(), lengths, semiring="max")

    assert torch.allclose(out, ref, atol=1e-6), f"max diff: {(out - ref).abs().max()}"

    # Test gradients
    out.sum().backward()
    assert edge.grad is not None
    assert torch.isfinite(edge.grad).all()


def test_max_semiring_gradients_are_sparse():
    """Test that max semiring gradients are sparse (argmax indicator)."""
    torch.manual_seed(123)
    batch, T, K, C = 1, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, semiring="max")
    out.sum().backward()

    # Max semiring gradients should be sparse (0 or 1 for argmax)
    # Due to ties, values should be in {0, 1} or very close
    grad = edge.grad
    assert grad is not None
    # Check that gradients are mostly 0 or 1 (allowing for numerical precision)
    is_zero_or_one = (grad.abs() < 1e-6) | ((grad - 1).abs() < 1e-6)
    assert is_zero_or_one.float().mean() > 0.9, "Max semiring gradients should be mostly 0 or 1"


def test_max_semiring_validate_mode():
    """Test that max semiring validate mode uses float64."""
    torch.manual_seed(1)
    batch, T, K, C = 1, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, validate=True, semiring="max")
    ref = semi_crf_forward_pytorch(edge.double(), lengths, semiring="max").to(edge.dtype)

    assert torch.allclose(out, ref, atol=1e-6)


def test_max_vs_log_semiring_ordering():
    """Test that max semiring score <= log semiring partition (max is tighter bound)."""
    torch.manual_seed(999)
    batch, T, K, C = 3, 10, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    log_partition = semi_crf_triton_forward(edge, lengths, semiring="log")
    max_score = semi_crf_triton_forward(edge, lengths, semiring="max")

    # Max score should always be <= log partition (since max <= logsumexp)
    assert (max_score <= log_partition + 1e-5).all(), "Max score should be <= log partition"


def test_max_semiring_variable_lengths():
    """Test max semiring with variable sequence lengths."""
    torch.manual_seed(456)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, semiring="max")
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="max")

    assert torch.allclose(out, ref, atol=1e-6)


def test_invalid_semiring_raises():
    """Test that invalid semiring name raises ValueError."""
    edge = torch.randn(1, 4, 3, 2, 2)
    lengths = torch.tensor([5])

    with pytest.raises(ValueError, match="semiring must be"):
        semi_crf_triton_forward(edge, lengths, semiring="invalid")

    with pytest.raises(ValueError, match="semiring must be"):
        semi_crf_forward_pytorch(edge, lengths, semiring="tropical")
