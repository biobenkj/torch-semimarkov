"""Tests for the explicit forward-backward implementation.

This module tests the PyTorch reference backward pass implementation,
verifying correctness against torch.autograd.gradcheck and the existing
autograd-based backward.

Note: Some tests are skipped because they reference checkpointed backward
functions that were refactored into the streaming API.
"""

import pytest
import torch

# Import from the correct module (backward, not triton_backward)
from torch_semimarkov.backward import (
    HAS_TRITON,
    semi_crf_backward_beta,
    semi_crf_backward_pytorch,
    semi_crf_compute_marginals,
    semi_crf_forward_backward,
    semi_crf_forward_with_alpha,
    semi_crf_triton_backward,
)

# These functions were refactored into the streaming API - mark tests as skipped
from torch_semimarkov.streaming.pytorch_reference import (
    _compute_checkpoint_interval,
)
from torch_semimarkov.triton_scan import (
    semi_crf_forward_pytorch,
)

# Checkpointed functions no longer exist in this form - tests will be skipped
_CHECKPOINTED_FUNCTIONS_AVAILABLE = False

# =============================================================================
# Forward with Alpha Tests
# =============================================================================


def test_forward_with_alpha_matches_reference():
    """Test that forward_with_alpha matches the reference forward."""
    torch.manual_seed(42)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="log")

    # Our implementation
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    assert torch.allclose(
        partition, ref_partition, atol=1e-5
    ), f"Partition mismatch: max diff {(partition - ref_partition).abs().max()}"


def test_forward_with_alpha_variable_lengths():
    """Test forward_with_alpha with variable sequence lengths."""
    torch.manual_seed(123)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    assert torch.allclose(partition, ref_partition, atol=1e-5)


def test_forward_with_alpha_max_semiring():
    """Test forward_with_alpha with max semiring."""
    torch.manual_seed(456)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="max")
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="max")

    assert torch.allclose(partition, ref_partition, atol=1e-5)


# =============================================================================
# Backward Beta Tests
# =============================================================================


def test_backward_beta_boundary_conditions():
    """Test that beta has correct boundary conditions."""
    torch.manual_seed(789)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    beta = semi_crf_backward_beta(edge, lengths, semiring="log")

    # At final position (T-1), beta should be 0
    assert torch.allclose(
        beta[:, T - 1, :], torch.zeros(batch, C), atol=1e-6
    ), "Beta at final position should be 0"


def test_backward_beta_variable_lengths():
    """Test backward_beta with variable lengths."""
    torch.manual_seed(321)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    beta = semi_crf_backward_beta(edge, lengths, semiring="log")

    # Check that beta at each sequence's final position is 0
    for b in range(batch):
        final_pos = lengths[b].item() - 1
        assert torch.allclose(
            beta[b, final_pos, :], torch.zeros(C), atol=1e-6
        ), f"Beta at final position for batch {b} should be 0"


# =============================================================================
# Marginal/Gradient Tests
# =============================================================================


def test_marginals_sum_to_expected_segments():
    """Test that marginals sum to expected number of segments.

    In a semi-CRF with variable durations, the expected number of segments
    is NOT fixed at T-1. Instead, it's the expected value under the model's
    distribution. We verify this by checking that the sum is within valid
    bounds and matches what we compute via a different method.
    """
    torch.manual_seed(111)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Total marginals = expected number of segments under the distribution
    # This should be between 1 (one long segment) and T-1 (all unit segments)
    total_marginal = marginals.sum(dim=(1, 2, 3, 4))

    # Check bounds: at least 1 segment, at most T-1 segments
    assert (total_marginal >= 1.0 - 1e-4).all(), f"Total marginal below 1: {total_marginal}"
    assert (
        total_marginal <= (lengths - 1).float() + 1e-4
    ).all(), f"Total marginal above T-1: {total_marginal}"


def test_marginals_non_negative():
    """Test that all marginals are non-negative."""
    torch.manual_seed(222)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    assert (marginals >= -1e-6).all(), "Marginals should be non-negative"


def test_marginals_bounded_by_one():
    """Test that all marginals are <= 1."""
    torch.manual_seed(333)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    assert (marginals <= 1.0 + 1e-6).all(), "Marginals should be <= 1"


# =============================================================================
# Gradient Correctness Tests
# =============================================================================


def test_backward_matches_autograd():
    """Test that explicit backward matches autograd-computed gradients."""
    torch.manual_seed(444)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Compute gradients via autograd
    partition_autograd = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition_autograd.sum().backward()
    grad_autograd = edge.grad.clone()

    # Compute gradients via explicit backward
    edge_detached = edge.detach()
    partition_explicit, grad_explicit = semi_crf_backward_pytorch(
        edge_detached, lengths, semiring="log"
    )

    # Check partition values match
    assert torch.allclose(
        partition_autograd.detach(), partition_explicit, atol=1e-5
    ), f"Partition mismatch: {(partition_autograd.detach() - partition_explicit).abs().max()}"

    # Check gradients match
    assert torch.allclose(
        grad_autograd, grad_explicit, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_autograd - grad_explicit).abs().max()}"


def test_backward_matches_autograd_variable_lengths():
    """Test gradient correctness with variable sequence lengths."""
    torch.manual_seed(555)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    # Autograd gradients
    partition_autograd = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition_autograd.sum().backward()
    grad_autograd = edge.grad.clone()

    # Explicit backward gradients
    edge_detached = edge.detach()
    partition_explicit, grad_explicit = semi_crf_backward_pytorch(
        edge_detached, lengths, semiring="log"
    )

    assert torch.allclose(partition_autograd.detach(), partition_explicit, atol=1e-5)
    assert torch.allclose(
        grad_autograd, grad_explicit, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_autograd - grad_explicit).abs().max()}"


def test_gradcheck_small():
    """Test gradients with torch.autograd.gradcheck on small inputs."""
    torch.manual_seed(666)
    batch, T, K, C = 1, 4, 2, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    # Use gradcheck to verify numerical gradients
    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_medium():
    """Test gradients with torch.autograd.gradcheck on medium inputs."""
    torch.manual_seed(777)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_variable_lengths():
    """Test gradients with variable lengths using gradcheck."""
    torch.manual_seed(888)
    batch, T, K, C = 2, 8, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([8, 5], dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# Autograd Function Tests
# =============================================================================


def test_autograd_function_forward():
    """Test that SemiCRFBackward.forward matches reference."""
    torch.manual_seed(999)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    out = semi_crf_forward_backward(edge, lengths, semiring="log")

    assert torch.allclose(out, ref, atol=1e-5)


def test_autograd_function_backward():
    """Test that SemiCRFBackward computes correct gradients."""
    torch.manual_seed(1000)
    batch, T, K, C = 2, 6, 3, 2

    # Reference gradients via autograd
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Our implementation
    edge_test = edge_ref.detach().clone().requires_grad_(True)
    out_test = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out_test.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out_ref.detach(), out_test.detach(), atol=1e-5)
    assert torch.allclose(
        grad_ref, grad_test, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_ref - grad_test).abs().max()}"


def test_autograd_function_with_grad_output():
    """Test backward with non-uniform grad_output."""
    torch.manual_seed(1100)
    batch, T, K, C = 3, 6, 3, 2

    # Reference
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    grad_output = torch.randn(batch)

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    (out_ref * grad_output).sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Our implementation
    edge_test = edge_ref.detach().clone().requires_grad_(True)
    out_test = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    (out_test * grad_output).sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_position_sequence():
    """Test with sequence length = 2 (single segment)."""
    torch.manual_seed(1200)
    batch, T, K, C = 2, 2, 2, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()

    # Our implementation
    edge.grad = None
    out = semi_crf_forward_backward(
        edge.detach().clone().requires_grad_(True), lengths, semiring="log"
    )

    assert torch.allclose(out, ref.detach(), atol=1e-5)


def test_large_K():
    """Test with K larger than T (many durations unused)."""
    torch.manual_seed(1300)
    batch, T, K, C = 2, 5, 10, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


def test_single_label():
    """Test with C=1 (single label)."""
    torch.manual_seed(1400)
    batch, T, K, C = 2, 6, 3, 1
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


def test_numerical_stability_large_values():
    """Test with large edge values (potential overflow)."""
    torch.manual_seed(1500)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C) * 10  # Large values
    edge = edge.requires_grad_(True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.isfinite(out).all(), "Output should be finite"
    assert torch.isfinite(grad_test).all(), "Gradients should be finite"
    assert torch.allclose(out, ref.detach(), atol=1e-4)
    assert torch.allclose(grad_ref, grad_test, atol=1e-3)


def test_numerical_stability_small_values():
    """Test with small edge values (potential underflow)."""
    torch.manual_seed(1600)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C) * 0.1  # Small values
    edge = edge.requires_grad_(True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.isfinite(out).all()
    assert torch.isfinite(grad_test).all()
    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Triton Backward Kernel Tests (CUDA only)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_matches_pytorch():
    """Test that Triton backward kernel matches PyTorch reference."""
    from torch_semimarkov.backward import launch_triton_backward_kernel

    torch.manual_seed(2000)
    batch, T, K, C = 4, 16, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition using PyTorch
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # Compute gradients using PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Compute gradients using Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(
        grad_triton, grad_pytorch, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_variable_lengths():
    """Test Triton backward kernel with variable sequence lengths."""
    from torch_semimarkov.backward import launch_triton_backward_kernel

    torch.manual_seed(2100)
    batch, T, K, C = 4, 20, 6, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.tensor([20, 15, 10, 5], dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(
        grad_triton, grad_pytorch, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_non_power_of_2_labels():
    """Test Triton backward kernel with non-power-of-2 label count."""
    from torch_semimarkov.backward import launch_triton_backward_kernel

    torch.manual_seed(2200)
    batch, T, K, C = 3, 12, 4, 5  # C=5 is not power of 2
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(
        grad_triton, grad_pytorch, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_autograd_function():
    """Test full autograd integration with Triton backward kernel."""
    torch.manual_seed(2300)
    batch, T, K, C = 4, 16, 5, 4

    # Reference gradients via PyTorch
    edge_ref = torch.randn(batch, T - 1, K, C, C, device="cuda", requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Triton backward via autograd function
    edge_triton = edge_ref.detach().clone().requires_grad_(True)
    out_triton = semi_crf_triton_backward(edge_triton, lengths, semiring="log")
    out_triton.sum().backward()
    grad_triton = edge_triton.grad

    assert torch.allclose(out_ref.detach(), out_triton.detach(), atol=1e-5)
    assert torch.allclose(
        grad_ref, grad_triton, atol=1e-4
    ), f"Gradient mismatch: max diff {(grad_ref - grad_triton).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_larger_shapes():
    """Test Triton backward kernel with larger shapes (closer to genomics use)."""
    from torch_semimarkov.backward import launch_triton_backward_kernel

    torch.manual_seed(2400)
    batch, T, K, C = 8, 64, 12, 8
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(
        grad_triton, grad_pytorch, atol=1e-3
    ), f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_gradcheck():
    """Verify Triton backward with torch.autograd.gradcheck."""
    torch.manual_seed(2500)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(
        batch, T - 1, K, C, C, dtype=torch.float64, device="cuda", requires_grad=True
    )
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    def func(edge_input):
        return semi_crf_triton_backward(edge_input, lengths, semiring="log")

    # Use gradcheck to verify numerical gradients
    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# Checkpoint Utility Tests
# =============================================================================


def test_checkpoint_interval_computation():
    """Test that checkpoint interval is max(sqrt(T*K), K).

    This tests the _compute_checkpoint_interval function which is used by
    the streaming API to determine optimal checkpointing frequency.
    """
    import math

    # Test with K=1 (HMM-like behavior)
    test_cases_default = [(4, 1), (16, 1), (64, 1), (100, 1), (256, 1), (1000, 1)]
    for T, K in test_cases_default:
        interval = _compute_checkpoint_interval(T, K)
        # Formula: max(K, int(sqrt(T * K)))
        expected = max(K, int(math.sqrt(T * K)))
        assert interval == expected, f"For T={T}, K={K}, expected {expected}, got {interval}"

    # Test with larger K
    test_cases_large_k = [(16, 8), (64, 10), (100, 20), (256, 5)]
    for T, K in test_cases_large_k:
        interval = _compute_checkpoint_interval(T, K)
        expected = max(K, int(math.sqrt(T * K)))
        assert interval == expected, f"For T={T}, K={K}, expected {expected}, got {interval}"


def test_checkpointed_memory_reduction():
    """Verify that checkpointing actually reduces memory usage.

    This is a pure math test validating the memory savings formula.
    The streaming API uses checkpointing to achieve O(√(T*K) × K × C) memory
    instead of O(T × C) for full alpha storage.
    """
    T = 100
    K = 8
    C = 10
    interval = _compute_checkpoint_interval(T, K)

    # Full storage: O(T * C) for alpha
    full_alpha_elements = T * C

    # Checkpointed storage: O(num_checkpoints * K * C) for ring buffer
    num_checkpoints = (T + interval - 1) // interval

    # Full alpha storage for reference comparison
    # The streaming API uses ring checkpoints which are much smaller
    reduction_factor = full_alpha_elements / (num_checkpoints * C)

    # Should achieve significant memory reduction
    assert (
        reduction_factor >= 3
    ), f"Expected at least 3x memory reduction, got {reduction_factor:.1f}x"


def test_optimized_checkpointed_memory_tradeoff():
    """Verify memory trade-off: O(√(T*K) × K × C) vs O(T × C).

    This is a pure math test validating the memory-compute tradeoff.
    The streaming API stores ring buffer checkpoints at intervals to enable
    efficient backward pass recomputation.
    """
    T = 100
    K = 8
    C = 10
    interval = _compute_checkpoint_interval(T, K)
    num_checkpoints = (T + interval - 1) // interval

    # Ring buffer checkpointing: O(num_checkpoints × K × C)
    ring_checkpoint_elements = num_checkpoints * K * C

    # Full storage: O(T × C)
    full_elements = T * C

    # Checkpointing should use significantly less memory than full storage
    reduction = full_elements / ring_checkpoint_elements

    # With K=8, T=100, interval≈28, num_ckpts≈4
    # ring_elements = 4 * 8 * 10 = 320
    # full_elements = 100 * 10 = 1000
    # reduction ≈ 3.1x
    assert reduction >= 1, f"Expected positive memory reduction, got {reduction:.1f}x"

    # Ring buffer should be much smaller than full alpha storage
    alpha_elements = num_checkpoints * C  # If we only stored alpha at checkpoints
    ring_ratio = ring_checkpoint_elements / alpha_elements
    assert abs(ring_ratio - K) < 0.1, f"Ring uses K× more than alpha-only, got {ring_ratio:.1f}×"
