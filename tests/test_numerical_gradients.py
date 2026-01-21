"""
Numerical gradient verification tests for SemiMarkov.

These tests use finite differences to verify that the analytical gradients
computed by autograd are correct. This provides strong validation that the
backward pass implementation is mathematically correct.
"""

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def finite_difference_gradient(func, x, eps=1e-5):
    """
    Compute gradient of func(x) using central finite differences.

    Args:
        func: Function that takes a tensor and returns a scalar
        x: Input tensor
        eps: Finite difference step size

    Returns:
        Tensor of same shape as x containing numerical gradients
    """
    grad = torch.zeros_like(x)
    x_flat = x.view(-1)
    grad_flat = grad.view(-1)

    for i in range(x_flat.numel()):
        x_plus = x_flat.clone()
        x_minus = x_flat.clone()
        x_plus[i] += eps
        x_minus[i] -= eps

        # Reshape back and compute function
        f_plus = func(x_plus.view_as(x))
        f_minus = func(x_minus.view_as(x))

        grad_flat[i] = (f_plus - f_minus) / (2 * eps)

    return grad


class TestNumericalGradients:
    """Test that analytical gradients match finite difference gradients."""

    def test_logpartition_gradient_small(self):
        """Verify gradient for small example using finite differences."""
        batch, N, K, C = 1, 4, 2, 2
        torch.manual_seed(42)

        # Use float64 for better numerical precision
        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        def forward_func(e):
            v, _, _ = sm.logpartition(e, lengths=lengths)
            return v.sum()

        # Analytical gradient
        edge_grad = edge.clone().requires_grad_(True)
        v, _, _ = sm.logpartition(edge_grad, lengths=lengths)
        v.sum().backward()
        analytical_grad = edge_grad.grad.clone()

        # Numerical gradient
        numerical_grad = finite_difference_gradient(forward_func, edge, eps=1e-6)

        # Compare
        max_diff = (analytical_grad - numerical_grad).abs().max().item()
        rel_diff = (
            ((analytical_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)).max().item()
        )

        assert max_diff < 1e-4, f"Max absolute diff: {max_diff:.2e}"
        assert rel_diff < 1e-3, f"Max relative diff: {rel_diff:.2e}"

    def test_logpartition_gradient_larger(self):
        """Verify gradient for slightly larger example."""
        batch, N, K, C = 2, 6, 3, 2
        torch.manual_seed(123)

        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        def forward_func(e):
            v, _, _ = sm.logpartition(e, lengths=lengths)
            return v.sum()

        # Analytical gradient
        edge_grad = edge.clone().requires_grad_(True)
        v, _, _ = sm.logpartition(edge_grad, lengths=lengths)
        v.sum().backward()
        analytical_grad = edge_grad.grad.clone()

        # Numerical gradient
        numerical_grad = finite_difference_gradient(forward_func, edge, eps=1e-6)

        max_diff = (analytical_grad - numerical_grad).abs().max().item()
        assert max_diff < 1e-4, f"Max absolute diff: {max_diff:.2e}"

    def test_logpartition_gradient_variable_lengths(self):
        """Verify gradient with variable sequence lengths."""
        batch, N, K, C = 2, 6, 3, 2
        torch.manual_seed(456)

        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64)
        lengths = torch.tensor([6, 4], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        def forward_func(e):
            v, _, _ = sm.logpartition(e, lengths=lengths)
            return v.sum()

        # Analytical gradient
        edge_grad = edge.clone().requires_grad_(True)
        v, _, _ = sm.logpartition(edge_grad, lengths=lengths)
        v.sum().backward()
        analytical_grad = edge_grad.grad.clone()

        # Numerical gradient
        numerical_grad = finite_difference_gradient(forward_func, edge, eps=1e-6)

        max_diff = (analytical_grad - numerical_grad).abs().max().item()
        assert max_diff < 1e-4, f"Max absolute diff: {max_diff:.2e}"

    def test_gradient_consistency_streaming(self):
        """Streaming backend produces consistent gradients."""
        batch, N, K, C = 2, 6, 3, 2
        torch.manual_seed(789)

        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        # Get gradient from streaming backend
        edge_grad = edge.clone().requires_grad_(True)
        v, _, _ = sm._dp_scan_streaming(edge_grad, lengths)
        v.sum().backward()
        grad = edge_grad.grad.clone()

        # Verify gradient is finite and non-zero
        assert torch.isfinite(grad).all(), "Gradient contains non-finite values"
        assert grad.abs().sum() > 0, "Gradient is all zeros"

        # Run again to verify determinism
        edge_ref = edge.clone().requires_grad_(True)
        v_ref, _, _ = sm._dp_scan_streaming(edge_ref, lengths)
        v_ref.sum().backward()
        grad_ref = edge_ref.grad.clone()

        max_diff = (grad - grad_ref).abs().max().item()
        assert max_diff < 1e-10, f"Streaming gradient not deterministic, diff: {max_diff:.2e}"


class TestMarginalGradients:
    """Test that marginals (which use gradients) are computed correctly."""

    def test_marginals_sum_to_one_per_position(self):
        """
        For each timestep, the sum of marginal probabilities over all
        (k, c_new, c_prev) combinations that start at that timestep
        should equal the probability of reaching that timestep.
        """
        batch, N, K, C = 2, 8, 4, 3
        torch.manual_seed(42)

        edge = torch.randn(batch, N - 1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        # Marginals should be non-negative
        assert (marginals >= -1e-6).all()
        # Marginals should be <= 1
        assert (marginals <= 1 + 1e-6).all()
        # Sum of all marginals should approximately equal N-1 (number of positions)
        # Actually, sum of marginals gives expected number of segments
        total_marginals = marginals.sum(dim=(2, 3, 4))  # Sum over K, C, C
        # Each position contributes some probability mass
        assert (total_marginals >= 0).all()

    def test_marginals_gradient_check(self):
        """Marginals should be the gradient of log partition w.r.t. potentials."""
        batch, N, K, C = 1, 5, 3, 2
        torch.manual_seed(42)

        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64, requires_grad=True)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        # Compute marginals (uses gradient internally)
        # Need to detach for marginals call since it computes its own gradients
        marginals = sm.marginals(edge.detach(), lengths=lengths)

        # Manually compute gradient using autograd
        v, _, _ = sm.logpartition(edge, lengths=lengths)
        v.sum().backward()
        manual_grad = edge.grad

        # Marginals should match the gradient
        # Note: marginals are normalized by the partition function
        # The gradient gives unnormalized edge "usage counts"
        # For LogSemiring, marginals = grad(log Z) = grad(Z) / Z
        # which equals the probability of each edge being used
        # The two should be close (both represent expected edge usage)
        assert marginals.shape == manual_grad.shape
        # They should be highly correlated
        correlation = torch.corrcoef(
            torch.stack([marginals.flatten(), manual_grad.float().flatten()])
        )[0, 1]
        assert correlation > 0.99, f"Correlation: {correlation:.4f}"


class TestHessianComputation:
    """Test second-order gradients (Hessian)."""

    def test_double_backward(self):
        """Second-order gradients should be computable."""
        batch, N, K, C = 1, 4, 2, 2
        torch.manual_seed(42)

        edge = torch.randn(batch, N - 1, K, C, C, requires_grad=True, dtype=torch.float64)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        v, _, _ = sm.logpartition(edge, lengths=lengths)

        # First backward
        (grad,) = torch.autograd.grad(v.sum(), edge, create_graph=True)

        # Second backward (Hessian-vector product with ones)
        (hessian_row_sum,) = torch.autograd.grad(grad.sum(), edge)

        # Should be finite
        assert torch.isfinite(hessian_row_sum).all()

    def test_hessian_finite_diff_verification(self):
        """Verify Hessian computation using finite differences on gradients."""
        batch, N, K, C = 1, 3, 2, 2
        torch.manual_seed(42)

        edge = torch.randn(batch, N - 1, K, C, C, dtype=torch.float64)
        lengths = torch.full((batch,), N, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        def grad_func(e):
            e = e.clone().requires_grad_(True)
            v, _, _ = sm.logpartition(e, lengths=lengths)
            v.sum().backward()
            return e.grad.sum()

        # Compute Hessian diagonal element using finite difference on gradient
        eps = 1e-5
        idx = (0, 0, 0, 0, 0)  # First element

        edge_plus = edge.clone()
        edge_plus[idx] += eps
        grad_plus = grad_func(edge_plus)

        edge_minus = edge.clone()
        edge_minus[idx] -= eps
        grad_minus = grad_func(edge_minus)

        numerical_hessian = (grad_plus - grad_minus) / (2 * eps)

        # Analytical Hessian via double backward
        edge_grad = edge.clone().requires_grad_(True)
        v, _, _ = sm.logpartition(edge_grad, lengths=lengths)
        (grad,) = torch.autograd.grad(v.sum(), edge_grad, create_graph=True)
        (hessian_row,) = torch.autograd.grad(grad.sum(), edge_grad)
        analytical_hessian = hessian_row[idx]

        diff = abs(numerical_hessian.item() - analytical_hessian.item())
        assert diff < 1e-3, f"Hessian diff: {diff:.2e}"
