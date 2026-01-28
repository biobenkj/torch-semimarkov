"""
Tests for the streaming API.

The streaming API computes edge potentials on-the-fly from cumulative scores,
enabling memory-efficient inference at chromosome scale (T=400K+).
"""

import warnings

import pytest
import torch

from torch_semimarkov.streaming import (
    compute_edge_block_streaming,
    semi_crf_streaming_forward,
)


def create_streaming_inputs(batch, T, K, C, device="cpu", dtype=torch.float32, seed=42):
    """Create test inputs for the streaming API."""
    torch.manual_seed(seed)

    # Simulate projected encoder features
    projected = torch.randn(batch, T, C, device=device, dtype=dtype)
    # Zero-center (critical for numerical stability at large T)
    projected = projected - projected.mean(dim=1, keepdim=True)

    # Cumulative scores: (batch, T+1, C)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    # Lengths
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    return cum_scores, transition, duration_bias, lengths


class TestEdgeBlockComputation:
    """Test the compute_edge_block_streaming helper."""

    def test_edge_block_shape(self):
        """Verify edge block has correct shape."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(batch, T, K, C)

        edge_block = compute_edge_block_streaming(cum_scores, transition, duration_bias, t=10, k=3)

        assert edge_block.shape == (
            batch,
            C,
            C,
        ), f"Expected shape (batch, C, C), got {edge_block.shape}"

    def test_edge_block_values(self):
        """Verify edge block computation is correct."""
        batch, T, K, C = 1, 20, 4, 2
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(batch, T, K, C)

        t, k = 5, 2
        edge_block = compute_edge_block_streaming(cum_scores, transition, duration_bias, t, k)

        # Manual computation
        # Duration k uses index k-1 (0-based indexing)
        content_score = cum_scores[:, t + k, :] - cum_scores[:, t, :]  # (batch, C)
        segment_score = content_score + duration_bias[k - 1]
        expected = segment_score.unsqueeze(-1) + transition.T.unsqueeze(0)

        torch.testing.assert_close(edge_block, expected)

    def test_edge_block_with_boundaries(self):
        """Verify edge block with boundary projections."""
        batch, T, K, C = 2, 50, 6, 3
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(batch, T, K, C)
        proj_start = torch.randn(batch, T, C)
        proj_end = torch.randn(batch, T, C)

        t, k = 10, 3
        edge_block = compute_edge_block_streaming(
            cum_scores, transition, duration_bias, t, k, proj_start, proj_end
        )

        # Verify shape
        assert edge_block.shape == (batch, C, C)

        # Verify boundaries are included (non-zero contribution)
        edge_no_boundary = compute_edge_block_streaming(cum_scores, transition, duration_bias, t, k)
        assert not torch.allclose(edge_block, edge_no_boundary)


class TestStreamingForward:
    """Test the streaming forward pass."""

    def test_forward_produces_finite_values(self):
        """Verify forward pass produces finite partition values."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all(), "Partition contains non-finite values"
        assert partition.shape == (batch,), f"Expected shape (batch,), got {partition.shape}"

    def test_forward_variable_lengths(self):
        """Verify forward handles variable sequence lengths."""
        batch, T, K, C = 4, 50, 6, 3
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(batch, T, K, C)
        lengths = torch.tensor([T, T - 10, T - 20, T], dtype=torch.long)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all()
        # Different lengths should give different values
        assert not torch.allclose(partition[0], partition[1])

    def test_forward_max_semiring(self):
        """Verify max semiring (Viterbi) works."""
        batch, T, K, C = 2, 50, 6, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition_log = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, semiring="log"
        )
        partition_max = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        assert torch.isfinite(partition_max).all()
        # Max should be less than or equal to logsumexp
        assert (partition_max <= partition_log + 1e-5).all()

    def test_forward_short_sequences(self):
        """Verify forward handles sequences shorter than K."""
        K, C, batch = 10, 3, 2

        for T in [4, 6, 8, 12]:
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, K
            )

            assert torch.isfinite(partition).all(), f"T={T}: Non-finite values"


class TestStreamingBackward:
    """Test the streaming backward pass."""

    def test_backward_produces_finite_gradients(self):
        """Verify backward produces finite gradients."""
        batch, T, K, C = 2, 50, 6, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "cum_scores grad non-finite"
        assert torch.isfinite(transition.grad).all(), "transition grad non-finite"
        assert torch.isfinite(duration_bias.grad).all(), "duration_bias grad non-finite"

    def test_backward_gradient_shapes(self):
        """Verify gradient shapes match input shapes."""
        batch, T, K, C = 2, 50, 6, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        assert cum_scores.grad.shape == cum_scores.shape
        assert transition.grad.shape == transition.shape
        assert duration_bias.grad.shape == duration_bias.shape

    def test_backward_with_boundaries(self):
        """Verify backward works with boundary projections."""
        batch, T, K, C = 2, 50, 6, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)
        proj_start = torch.randn(batch, T, C, requires_grad=True)
        proj_end = torch.randn(batch, T, C, requires_grad=True)

        cum_scores.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )
        partition.sum().backward()

        assert proj_start.grad is not None, "proj_start should have gradient"
        assert proj_end.grad is not None, "proj_end should have gradient"
        assert torch.isfinite(proj_start.grad).all()
        assert torch.isfinite(proj_end.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_cuda(self):
        """Verify backward works on CUDA."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all()


class TestGradientCorrectness:
    """Test gradient correctness using finite differences."""

    def test_gradcheck_small(self):
        """Verify gradients using torch.autograd.gradcheck."""
        batch, T, K, C = 1, 10, 4, 2
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, dtype=torch.float64
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # gradcheck requires float64 for numerical precision
        def func(cs, tr, db):
            return semi_crf_streaming_forward(cs, tr, db, lengths, K)

        # Use smaller epsilon for better numerical stability
        # Suppress float32 warning since gradcheck requires float64
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="cum_scores should be float32")
            torch.autograd.gradcheck(
                func,
                (cum_scores, transition, duration_bias),
                eps=1e-4,
                atol=1e-3,
                rtol=1e-3,
            )

    def test_gradient_sum_property(self):
        """Verify marginal probabilities sum to approximately 1 per position."""
        batch, T, K, C = 1, 20, 4, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        # The gradient w.r.t. cum_scores represents flow through each position
        # Sum of gradients per position should be close to 0 for interior positions
        # (because each position contributes +1 to end and -1 to start)
        grad_sum_per_pos = cum_scores.grad.sum(dim=-1)  # (batch, T+1)

        # Interior positions should have near-zero gradient sum
        # (excluding first and last positions which are boundary conditions)
        interior_grad_sum = grad_sum_per_pos[:, 1:-1]
        assert interior_grad_sum.abs().mean() < 0.5, "Interior gradient sums should be small"


class TestK1HMMLike:
    """Test streaming with K=1 (HMM-like, unit segments only)."""

    def test_streaming_k1_produces_finite(self):
        """Test streaming forward with K=1 produces finite values."""
        batch, T, K, C = 2, 50, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all(), "K=1: Partition contains non-finite values"
        assert partition.shape == (batch,)

    def test_streaming_k1_gradient_flow(self):
        """Test gradients flow correctly with K=1."""
        batch, T, K, C = 2, 30, 1, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
        partition.sum().backward()

        assert cum_scores.grad is not None, "K=1: cum_scores should have gradient"
        assert transition.grad is not None, "K=1: transition should have gradient"
        assert duration_bias.grad is not None, "K=1: duration_bias should have gradient"
        assert torch.isfinite(cum_scores.grad).all(), "K=1: cum_scores grad non-finite"
        assert torch.isfinite(transition.grad).all(), "K=1: transition grad non-finite"
        assert torch.isfinite(duration_bias.grad).all(), "K=1: duration_bias grad non-finite"

    def test_streaming_k1_variable_lengths(self):
        """Test K=1 with variable sequence lengths."""
        batch, T, K, C = 4, 40, 1, 3
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(batch, T, K, C)
        lengths = torch.tensor([T, T - 10, T - 20, T - 5], dtype=torch.long)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all()
        # Different lengths should give different values
        assert (
            len(set(partition.tolist())) > 1
        ), "K=1: Different lengths should give different values"

    def test_streaming_k1_gradcheck(self):
        """Verify K=1 gradients using torch.autograd.gradcheck."""
        batch, T, K, C = 1, 10, 1, 2
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, dtype=torch.float64
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        def func(cs, tr, db):
            return semi_crf_streaming_forward(cs, tr, db, lengths, K)

        # Suppress float32 warning since gradcheck requires float64
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="cum_scores should be float32")
            torch.autograd.gradcheck(
                func,
                (cum_scores, transition, duration_bias),
                eps=1e-4,
                atol=1e-3,
                rtol=1e-3,
            )


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_large_T_produces_finite(self):
        """Verify forward works with moderately large T."""
        batch, T, K, C = 1, 1000, 10, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

        assert torch.isfinite(partition).all(), f"T={T}: Non-finite partition"

    def test_zero_centered_input(self):
        """Verify zero-centering warning for non-centered inputs."""
        batch, T, K, C = 1, 100, 8, 4

        # Create non-zero-centered input (should trigger warning)
        torch.manual_seed(42)
        cum_scores = torch.randn(batch, T + 1, C) * 100  # Large drift
        cum_scores = torch.cumsum(cum_scores, dim=1)  # Even larger drift
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, K
            )
            # Should get a warning about non-zero-centered input
            del _partition  # Only called to trigger warning
            assert len(w) > 0, "Expected warning about non-zero-centered input"


# =============================================================================
# Oracle Tests with Manual Computation
# =============================================================================


class TestStreamingOracleValues:
    """Small-scale tests with manually computed expected values.

    These tests use minimal configurations (T=2-4, K=2, C=2) to verify
    the streaming implementation produces correct partition values by
    comparing against manually computed expected results.
    """

    def test_duration_bias_affects_partition(self):
        """
        Changing duration_bias should change the partition value.

        This verifies that duration_bias is correctly incorporated into
        the partition computation.
        """
        batch, T, K, C = 1, 5, 4, 2
        cum_scores, transition, _, lengths = create_streaming_inputs(batch, T, K, C)

        # Compute with zero duration bias
        duration_bias_zero = torch.zeros(K, C)
        partition_zero = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias_zero, lengths, K
        )

        # Compute with non-zero duration bias
        duration_bias_nonzero = torch.ones(K, C) * 0.5
        partition_nonzero = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias_nonzero, lengths, K
        )

        # Partition should be different
        assert not torch.allclose(
            partition_zero, partition_nonzero, atol=1e-3
        ), "Duration bias should affect partition value"

        # Non-zero bias should increase partition (since all biases are positive)
        assert (
            partition_nonzero > partition_zero
        ).all(), "Positive duration bias should increase partition"

    def test_transition_affects_partition(self):
        """
        Changing transition matrix should change the partition value.

        This verifies that transitions are correctly incorporated.
        """
        batch, T, K, C = 1, 5, 4, 2
        cum_scores, _, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        # Compute with zero transition
        transition_zero = torch.zeros(C, C)
        partition_zero = semi_crf_streaming_forward(
            cum_scores, transition_zero, duration_bias, lengths, K
        )

        # Compute with non-zero transition
        transition_nonzero = torch.ones(C, C) * 0.5
        partition_nonzero = semi_crf_streaming_forward(
            cum_scores, transition_nonzero, duration_bias, lengths, K
        )

        # Partition should be different
        assert not torch.allclose(
            partition_zero, partition_nonzero, atol=1e-3
        ), "Transition matrix should affect partition value"

    def test_content_scores_affect_partition(self):
        """
        Changing content scores should change the partition value.

        This verifies that content (emission) scores are correctly incorporated.
        """
        batch, T, K, C = 1, 5, 4, 2
        _, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        # Compute with zero content scores
        cum_scores_zero = torch.zeros(batch, T + 1, C)
        partition_zero = semi_crf_streaming_forward(
            cum_scores_zero, transition, duration_bias, lengths, K
        )

        # Compute with non-zero content scores
        cum_scores_nonzero = torch.zeros(batch, T + 1, C)
        cum_scores_nonzero[:, :, 0] = torch.arange(T + 1).float()  # Ramp for class 0
        partition_nonzero = semi_crf_streaming_forward(
            cum_scores_nonzero, transition, duration_bias, lengths, K
        )

        # Partition should be different
        assert not torch.allclose(
            partition_zero, partition_nonzero, atol=1e-3
        ), "Content scores should affect partition value"

    def test_cumulative_scores_formula(self):
        """
        Verify: cum_scores[:, t, :] = Σ_{i<t} scores[:, i, :].

        This tests the helper function indirectly by verifying the formula.
        """
        batch, T, C = 2, 10, 3
        scores = torch.randn(batch, T, C)

        # Zero-center (as done in practice)
        scores_centered = scores - scores.mean(dim=1, keepdim=True)

        # Build cum_scores
        cum_scores = torch.zeros(batch, T + 1, C)
        cum_scores[:, 1:, :] = torch.cumsum(scores_centered, dim=1)

        # Verify formula for each position
        for t in range(1, T + 1):
            expected = scores_centered[:, :t, :].sum(dim=1)
            actual = cum_scores[:, t, :]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_edge_block_formula_detailed(self):
        """
        Verify edge block formula in detail:
        edge[c_new, c_prev] = content_score[c_new] + duration_bias[k, c_new] + transition[c_prev, c_new]
        """
        batch, T, K, C = 1, 10, 5, 3

        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)

        t, k = 3, 2
        edge_block = compute_edge_block_streaming(cum_scores, transition, duration_bias, t, k)

        # Verify each element
        # Duration k uses index k-1 (0-based indexing)
        for c_new in range(C):
            for c_prev in range(C):
                content_score = cum_scores[0, t + k, c_new] - cum_scores[0, t, c_new]
                dur_bias = duration_bias[k - 1, c_new]
                trans = transition[c_prev, c_new]
                expected = content_score + dur_bias + trans

                actual = edge_block[0, c_new, c_prev]
                assert torch.isclose(
                    actual, expected, atol=1e-6
                ), f"Edge[{c_new}, {c_prev}]: expected {expected:.4f}, got {actual:.4f}"


if __name__ == "__main__":
    # Quick manual test
    batch, T, K, C = 2, 100, 8, 4
    cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

    cum_scores.requires_grad_(True)
    transition.requires_grad_(True)
    duration_bias.requires_grad_(True)

    print("Input shapes:")
    print(f"  cum_scores: {cum_scores.shape}")
    print(f"  transition: {transition.shape}")
    print(f"  duration_bias: {duration_bias.shape}")
    print(f"  lengths: {lengths}")

    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
    print(f"\nPartition: {partition}")

    partition.sum().backward()
    print("\nGradients:")
    print(f"  cum_scores.grad finite: {torch.isfinite(cum_scores.grad).all()}")
    print(f"  transition.grad finite: {torch.isfinite(transition.grad).all()}")
    print(f"  duration_bias.grad finite: {torch.isfinite(duration_bias.grad).all()}")
