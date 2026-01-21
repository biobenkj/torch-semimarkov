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
    ref, _, _ = model.logpartition(edge, lengths=lengths)

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


# =============================================================================
# Hybrid Approach Tests
# =============================================================================


def test_hybrid_inference_vs_training_paths():
    """Test that inference (no grad) and training (with grad) paths give same results."""
    torch.manual_seed(789)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Inference path (no requires_grad)
    inference_result = semi_crf_triton_forward(edge, lengths)

    # Training path (with requires_grad)
    edge_train = edge.clone().requires_grad_(True)
    training_result = semi_crf_triton_forward(edge_train, lengths)

    # Results should match
    assert torch.allclose(inference_result, training_result, atol=1e-6)


def test_use_compile_false_fallback():
    """Test that use_compile=False falls back to gradient checkpointing."""
    torch.manual_seed(321)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Test with use_compile=False (should use gradient checkpointing fallback)
    out = semi_crf_triton_forward(edge, lengths, use_compile=False)
    ref = semi_crf_forward_pytorch(edge.detach(), lengths)

    assert torch.allclose(out, ref, atol=1e-6)

    # Verify gradients work
    out.sum().backward()
    assert edge.grad is not None
    assert torch.isfinite(edge.grad).all()


def test_hybrid_gradients_match_pytorch():
    """Test that gradients from hybrid approach match pure PyTorch reference."""
    torch.manual_seed(555)
    batch, T, K, C = 2, 6, 3, 2
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    edge_test = edge_ref.detach().clone().requires_grad_(True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference gradients (pure PyTorch)
    out_ref = semi_crf_forward_pytorch(edge_ref, lengths)
    out_ref.sum().backward()

    # Test gradients (via hybrid triton forward, use_compile=False to avoid torch.compile overhead in test)
    out_test = semi_crf_triton_forward(edge_test, lengths, use_compile=False)
    out_test.sum().backward()

    # Gradients should match
    assert torch.allclose(
        edge_ref.grad, edge_test.grad, atol=1e-5
    ), f"max grad diff: {(edge_ref.grad - edge_test.grad).abs().max()}"


# =============================================================================
# CUDA-specific Tests (requires GPU)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_log_kernel_cuda():
    """Test that log semiring Triton kernel works on CUDA."""
    from torch_semimarkov.triton_scan import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton not available")

    torch.manual_seed(111)
    batch, T, K, C = 4, 20, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # This should use the Triton kernel (no grad = inference path)
    result = semi_crf_triton_forward(edge, lengths, semiring="log")

    # Compare with CPU reference
    ref = semi_crf_triton_forward(edge.cpu(), lengths.cpu(), semiring="log")

    assert torch.allclose(
        result.cpu(), ref, atol=1e-5
    ), f"CUDA/CPU mismatch: max diff {(result.cpu() - ref).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_max_kernel_cuda():
    """Test that max semiring Triton kernel works on CUDA."""
    from torch_semimarkov.triton_scan import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton not available")

    torch.manual_seed(222)
    batch, T, K, C = 4, 20, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # This should use the Triton max kernel (no grad = inference path)
    result = semi_crf_triton_forward(edge, lengths, semiring="max")

    # Compare with CPU reference
    ref = semi_crf_triton_forward(edge.cpu(), lengths.cpu(), semiring="max")

    assert torch.allclose(
        result.cpu(), ref, atol=1e-5
    ), f"CUDA/CPU mismatch: max diff {(result.cpu() - ref).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_both_semirings_cuda_sequential():
    """Test both semirings sequentially on CUDA to catch state issues."""
    from torch_semimarkov.triton_scan import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton not available")

    torch.manual_seed(333)
    batch, T, K, C = 4, 20, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Run log first, then max (mimics benchmark order)
    result_log = semi_crf_triton_forward(edge, lengths, semiring="log")
    result_max = semi_crf_triton_forward(edge, lengths, semiring="max")

    # Compare with CPU references
    ref_log = semi_crf_triton_forward(edge.cpu(), lengths.cpu(), semiring="log")
    ref_max = semi_crf_triton_forward(edge.cpu(), lengths.cpu(), semiring="max")

    assert torch.allclose(result_log.cpu(), ref_log, atol=1e-5), "Log mismatch"
    assert torch.allclose(result_max.cpu(), ref_max, atol=1e-5), "Max mismatch"

    # Max should be <= Log for same inputs
    assert (result_max <= result_log + 1e-5).all(), "Max should be <= Log"
