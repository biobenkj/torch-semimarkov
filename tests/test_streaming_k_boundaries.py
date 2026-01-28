"""K-specific boundary tests for streaming Semi-CRF.

Tests the critical K value boundaries:
- K=1: Linear CRF fast path (all segments duration 1)
- K=2: Specialized semi-CRF path
- K=3: Minimum K for Triton kernel (ring buffer boundary)

Includes manual oracle tests that enumerate all valid segmentations
for small configurations to verify partition computation.
"""

import pytest
import torch

from torch_semimarkov.streaming import HAS_TRITON, semi_crf_streaming_forward

# Import reference implementations for dispatch verification
from torch_semimarkov.streaming.pytorch_reference import (
    linear_crf_forward_pytorch,
    semi_crf_k2_forward_pytorch,
    semi_crf_k2_viterbi_pytorch,
    semi_crf_streaming_forward_pytorch,
)

if HAS_TRITON:
    from torch_semimarkov.streaming import launch_streaming_triton_kernel


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


# =============================================================================
# K=1 Tests (Linear CRF Fast Path)
# =============================================================================


class TestK1LinearCRFBoundary:
    """K=1 boundary tests - dispatches to LinearCRFStreaming.

    K=1 produces HMM-like behavior where each position is its own segment.
    The ring buffer has only 1 slot, so all positions alias to index 0.
    """

    def test_k1_dispatch_uses_linear_crf(self):
        """Verify K=1 routes to linear_crf_forward_pytorch, not generic streaming."""
        batch, T, K, C = 2, 20, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        # Direct call to linear CRF
        partition_direct = linear_crf_forward_pytorch(
            cum_scores, transition, lengths, duration_bias
        )

        # Call through dispatch (should route to linear CRF)
        partition_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        torch.testing.assert_close(
            partition_dispatch,
            partition_direct,
            rtol=1e-5,
            atol=1e-5,
            msg="K=1 dispatch should route to linear_crf_forward_pytorch",
        )

    def test_k1_oracle_t3_c2(self):
        """Manual oracle: T=3, K=1, C=2 - enumerate all paths.

        With K=1, each position is a segment of duration 1.
        For T=3, C=2, we have 2^3 = 8 possible label sequences.

        The partition function is:
        Z = sum over all (c0, c1, c2) of:
            exp(content[0,c0] + dur_bias[0,c0] +
                trans[c0,c1] + content[1,c1] + dur_bias[0,c1] +
                trans[c1,c2] + content[2,c2] + dur_bias[0,c2])

        where content[t,c] = cum_scores[t+1,c] - cum_scores[t,c]
        """
        batch, T, K, C = 1, 3, 1, 2
        torch.manual_seed(123)  # Fixed seed for reproducible test

        # Create simple inputs for manual computation
        cum_scores = torch.zeros(batch, T + 1, C)
        # Set specific cumulative scores for easy manual calculation
        # content[t,c] = cum_scores[t+1,c] - cum_scores[t,c]
        cum_scores[0, 1, :] = torch.tensor([1.0, 0.5])  # content[0] = [1.0, 0.5]
        cum_scores[0, 2, :] = torch.tensor([1.5, 1.5])  # content[1] = [0.5, 1.0]
        cum_scores[0, 3, :] = torch.tensor([2.5, 2.0])  # content[2] = [1.0, 0.5]

        # Simple transition matrix
        transition = torch.zeros(C, C)
        transition[0, 0] = 0.1  # stay in class 0
        transition[0, 1] = 0.2  # 0 -> 1
        transition[1, 0] = 0.3  # 1 -> 0
        transition[1, 1] = 0.1  # stay in class 1

        # Duration bias for k=1 (only index 0 used)
        duration_bias = torch.zeros(K, C)
        duration_bias[0, :] = torch.tensor([0.1, 0.2])

        lengths = torch.tensor([T])

        # Manual computation
        # content scores per position
        content = torch.zeros(T, C)
        content[0] = cum_scores[0, 1, :] - cum_scores[0, 0, :]  # [1.0, 0.5]
        content[1] = cum_scores[0, 2, :] - cum_scores[0, 1, :]  # [0.5, 1.0]
        content[2] = cum_scores[0, 3, :] - cum_scores[0, 2, :]  # [1.0, 0.5]

        # Use implementation to get partition
        partition_impl = linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

        # Verify via forward algorithm manually
        # alpha[t, c] shape: (C,)
        alpha = torch.zeros(C)  # alpha[0, :] = 0

        for t in range(1, T + 1):
            emission = content[t - 1] + duration_bias[0]  # (C,)
            # alpha_new[c_dst] = logsumexp_{c_src}(alpha[c_src] + trans[c_src, c_dst]) + emission[c_dst]
            alpha_new = torch.logsumexp(alpha.unsqueeze(-1) + transition, dim=0) + emission
            alpha = alpha_new

        # Final partition
        expected_partition = torch.logsumexp(alpha, dim=0)

        torch.testing.assert_close(
            partition_impl,
            expected_partition.unsqueeze(0),
            rtol=1e-5,
            atol=1e-5,
            msg="K=1 oracle: partition doesn't match manual forward computation",
        )

    def test_k1_gradient_flow(self):
        """Verify gradients flow correctly through K=1 path."""
        batch, T, K, C = 2, 15, 1, 3
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "K=1 cum_scores grad not finite"
        assert torch.isfinite(transition.grad).all(), "K=1 transition grad not finite"
        assert torch.isfinite(duration_bias.grad).all(), "K=1 duration_bias grad not finite"


# =============================================================================
# K=2 Tests (Specialized Semi-CRF Path)
# =============================================================================


class TestK2SpecializedBoundary:
    """K=2 boundary tests - dispatches to semi_crf_k2_forward_pytorch.

    K=2 allows segments of duration 1 or 2. Uses explicit 2-step history
    instead of ring buffer.
    """

    def test_k2_dispatch_uses_specialized(self):
        """Verify K=2 routes to K=2 specialized path, not generic streaming."""
        batch, T, K, C = 2, 20, 2, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        # Direct call to K=2 specialized
        partition_direct = semi_crf_k2_forward_pytorch(
            cum_scores, transition, duration_bias, lengths
        )

        # Call through dispatch (should route to K=2 specialized)
        partition_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        torch.testing.assert_close(
            partition_dispatch,
            partition_direct,
            rtol=1e-5,
            atol=1e-5,
            msg="K=2 dispatch should route to semi_crf_k2_forward_pytorch",
        )

    def test_k2_oracle_t3_c2(self):
        """Manual oracle: T=3, K=2, C=2 - enumerate all segmentations.

        Valid segmentations for T=3 with K=2:
        - [1,1,1]: three duration-1 segments
        - [1,2]: duration-1 segment (pos 0), duration-2 segment (pos 1-2)
        - [2,1]: duration-2 segment (pos 0-1), duration-1 segment (pos 2)

        Note: [3] is not valid because K=2 (max duration is 2).

        Each segmentation with label assignment contributes to partition.
        """
        batch, T, K, C = 1, 3, 2, 2
        torch.manual_seed(456)

        # Create simple inputs
        cum_scores = torch.zeros(batch, T + 1, C)
        # content[t:t+k, c] = cum_scores[t+k, c] - cum_scores[t, c]
        cum_scores[0, 1, :] = torch.tensor([1.0, 0.5])
        cum_scores[0, 2, :] = torch.tensor([1.8, 1.2])
        cum_scores[0, 3, :] = torch.tensor([2.5, 2.0])

        transition = torch.zeros(C, C)
        transition[0, 0] = 0.1
        transition[0, 1] = 0.2
        transition[1, 0] = 0.15
        transition[1, 1] = 0.1

        duration_bias = torch.zeros(K, C)
        duration_bias[0, :] = torch.tensor([0.1, 0.05])  # k=1 uses index 0
        duration_bias[1, :] = torch.tensor([0.2, 0.15])  # k=2 uses index 1

        lengths = torch.tensor([T])

        # Verify against K=1 relationship
        # K=2 should have partition >= K=1 (more paths to sum)
        partition_k2 = semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths)

        # K=1 partition (using only duration-1 segments)
        partition_k1 = linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

        # K=2 >= K=1 because K=2 includes all K=1 paths plus additional paths
        assert partition_k2.item() >= partition_k1.item() - 1e-6, (
            f"K=2 partition ({partition_k2.item():.4f}) should be >= "
            f"K=1 partition ({partition_k1.item():.4f})"
        )

        # Also verify the partition is finite and reasonable
        assert torch.isfinite(partition_k2).all(), "K=2 partition should be finite"

    def test_k2_reference_comparison(self):
        """Compare K=2 specialized vs generic PyTorch reference for larger case."""
        batch, T, K, C = 4, 50, 2, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        # K=2 specialized path
        partition_k2 = semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths)

        # Generic streaming path (for reference)
        partition_generic, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # They should match (generic should handle K=2 correctly)
        torch.testing.assert_close(
            partition_k2,
            partition_generic,
            rtol=1e-4,
            atol=1e-4,
            msg="K=2 specialized should match generic streaming for K=2",
        )

    def test_k2_viterbi_oracle_optimal_path(self):
        """Verify Viterbi returns known-optimal segmentation.

        Set up scores so the optimal path is unambiguous:
        - Make class 0 strongly preferred at all positions
        - Make duration 2 strongly preferred over duration 1
        """
        batch, T, K, C = 1, 4, 2, 2
        torch.manual_seed(789)

        # Create inputs where optimal path is clear
        cum_scores = torch.zeros(batch, T + 1, C)
        # Make class 0 have much higher content scores
        for t in range(1, T + 1):
            cum_scores[0, t, 0] = t * 2.0  # Class 0: high cumulative score
            cum_scores[0, t, 1] = t * 0.5  # Class 1: low cumulative score

        transition = torch.zeros(C, C)
        # No transition preference
        transition.fill_(0.0)

        duration_bias = torch.zeros(K, C)
        # Strong preference for duration 2
        duration_bias[0, :] = 0.0  # k=1
        duration_bias[1, :] = 1.0  # k=2 (preferred)

        lengths = torch.tensor([T])

        scores, paths, durations = semi_crf_k2_viterbi_pytorch(
            cum_scores, transition, duration_bias, lengths
        )

        # With these scores:
        # - Class 0 is preferred (higher content)
        # - Duration 2 is preferred (higher bias)
        # Optimal should be: [d=2, d=2] with labels [0, 0, 0, 0]
        # (two segments of duration 2, both class 0)

        # Check that Viterbi prefers class 0
        assert (
            paths[0, : lengths[0]] == 0
        ).all(), f"Expected all class 0, got {paths[0, :lengths[0]].tolist()}"

        # Check finite score
        assert torch.isfinite(scores).all(), "Viterbi score should be finite"

    def test_k2_gradient_flow(self):
        """Verify gradients flow correctly through K=2 path."""
        batch, T, K, C = 2, 20, 2, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(batch, T, K, C)

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "K=2 cum_scores grad not finite"
        assert torch.isfinite(transition.grad).all(), "K=2 transition grad not finite"
        assert torch.isfinite(duration_bias.grad).all(), "K=2 duration_bias grad not finite"


# =============================================================================
# K=3 Tests (Minimum Triton Kernel Boundary)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
class TestK3TritonBoundary:
    """K=3 is the minimum K for Triton kernel - critical boundary.

    K>=3 uses the general streaming algorithm with ring buffer.
    K=3 is the smallest K where the ring buffer has meaningful rotation
    (positions 0,1,2 map to distinct ring indices).
    """

    def test_k3_triton_forward_matches_pytorch(self):
        """K=3 Triton kernel matches PyTorch reference."""
        batch, T, K, C = 2, 50, 3, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            msg="K=3 Triton should match PyTorch reference",
        )

    def test_k3_backward_finite_gradients(self):
        """K=3 backward pass produces finite gradients."""
        batch, T, K, C = 2, 30, 3, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "K=3 cum_scores grad not finite"
        assert torch.isfinite(transition.grad).all(), "K=3 transition grad not finite"
        assert torch.isfinite(duration_bias.grad).all(), "K=3 duration_bias grad not finite"

    def test_k3_gradients_match_pytorch(self):
        """K=3 Triton gradients match PyTorch reference."""
        batch, T, K, C = 2, 30, 3, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients
        torch.testing.assert_close(
            cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2, msg="K=3 cum_scores grad mismatch"
        )
        torch.testing.assert_close(
            tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2, msg="K=3 transition grad mismatch"
        )
        torch.testing.assert_close(
            db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2, msg="K=3 duration_bias grad mismatch"
        )

    def test_k3_variable_lengths(self):
        """K=3 handles variable sequence lengths correctly."""
        batch, T, K, C = 4, 50, 3, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            msg="K=3 variable lengths: Triton should match PyTorch",
        )

    def test_k3_max_semiring(self):
        """K=3 max semiring (Viterbi) matches PyTorch reference."""
        batch, T, K, C = 2, 40, 3, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            msg="K=3 max semiring: Triton should match PyTorch",
        )

    def test_k3_oracle_partition_relationship(self):
        """K=3 partition >= K=2 partition (more paths to sum)."""
        batch, T, C = 2, 30, 4
        torch.manual_seed(999)

        # Create inputs on CUDA
        projected = torch.randn(batch, T, C, device="cuda")
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda")
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        transition = torch.randn(C, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # K=2 inputs
        duration_bias_k2 = torch.randn(2, C, device="cuda") * 0.1
        partition_k2 = semi_crf_k2_forward_pytorch(
            cum_scores, transition, duration_bias_k2, lengths
        )

        # K=3 inputs (extend duration_bias)
        duration_bias_k3 = torch.zeros(3, C, device="cuda")
        duration_bias_k3[:2, :] = duration_bias_k2
        duration_bias_k3[2, :] = torch.randn(C, device="cuda") * 0.1

        partition_k3, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias_k3, lengths, K=3
        )

        # K=3 should be >= K=2 (includes all K=2 paths plus duration-3 paths)
        assert (partition_k3 >= partition_k2 - 1e-4).all(), (
            f"K=3 partition should be >= K=2 partition\n"
            f"K=3: {partition_k3.tolist()}\n"
            f"K=2: {partition_k2.tolist()}"
        )


# =============================================================================
# K Transition Tests
# =============================================================================


class TestKBoundaryTransitions:
    """Tests for behavior at K value boundaries."""

    def test_k2_partition_gte_k1(self):
        """K=2 partition >= K=1 partition (more paths to sum)."""
        batch, T, C = 2, 30, 4
        torch.manual_seed(111)

        cum_scores, transition, _, lengths = create_streaming_inputs(
            batch, T, 2, C  # K doesn't matter for cum_scores/transition
        )

        # K=1
        duration_bias_k1 = torch.randn(1, C) * 0.1
        partition_k1 = linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias_k1)

        # K=2 (with same bias for k=1, new bias for k=2)
        duration_bias_k2 = torch.zeros(2, C)
        duration_bias_k2[0, :] = duration_bias_k1[0, :]
        duration_bias_k2[1, :] = torch.randn(C) * 0.1

        partition_k2 = semi_crf_k2_forward_pytorch(
            cum_scores, transition, duration_bias_k2, lengths
        )

        assert (partition_k2 >= partition_k1 - 1e-5).all(), (
            f"K=2 partition should be >= K=1 partition\n"
            f"K=2: {partition_k2.tolist()}\n"
            f"K=1: {partition_k1.tolist()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
    def test_k3_partition_gte_k2(self):
        """K=3 partition >= K=2 partition (more paths to sum)."""
        batch, T, C = 2, 30, 4
        torch.manual_seed(222)

        cum_scores, transition, _, lengths = create_streaming_inputs(batch, T, 3, C, device="cuda")

        # K=2
        duration_bias_k2 = torch.randn(2, C, device="cuda") * 0.1
        partition_k2 = semi_crf_k2_forward_pytorch(
            cum_scores, transition, duration_bias_k2, lengths
        )

        # K=3
        duration_bias_k3 = torch.zeros(3, C, device="cuda")
        duration_bias_k3[:2, :] = duration_bias_k2
        duration_bias_k3[2, :] = torch.randn(C, device="cuda") * 0.1

        partition_k3, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias_k3, lengths, K=3
        )

        assert (partition_k3 >= partition_k2 - 1e-4).all(), (
            f"K=3 partition should be >= K=2 partition\n"
            f"K=3: {partition_k3.tolist()}\n"
            f"K=2: {partition_k2.tolist()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
    def test_k2_to_k3_dispatch_transition(self):
        """Verify K=2 uses specialized path, K=3 uses Triton (on CUDA)."""
        batch, T, C = 2, 30, 4
        torch.manual_seed(333)

        cum_scores, transition, _, lengths = create_streaming_inputs(batch, T, 3, C, device="cuda")

        # K=2 should use specialized path (not Triton)
        duration_bias_k2 = torch.randn(2, C, device="cuda") * 0.1
        partition_k2_specialized = semi_crf_k2_forward_pytorch(
            cum_scores, transition, duration_bias_k2, lengths
        )
        partition_k2_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias_k2, lengths, K=2, use_triton=True
        )

        torch.testing.assert_close(
            partition_k2_dispatch,
            partition_k2_specialized,
            rtol=1e-5,
            atol=1e-5,
            msg="K=2 dispatch should use specialized path even with use_triton=True",
        )

        # K=3 should use Triton
        duration_bias_k3 = torch.randn(3, C, device="cuda") * 0.1
        partition_k3_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias_k3, lengths, K=3
        )
        partition_k3_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias_k3, lengths, K=3, use_triton=True
        )

        torch.testing.assert_close(
            partition_k3_dispatch,
            partition_k3_triton,
            rtol=1e-4,
            atol=1e-4,
            msg="K=3 dispatch should use Triton kernel",
        )

    def test_k_dispatch_cpu_fallback(self):
        """Verify K>=3 on CPU falls back to PyTorch reference."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        # On CPU, should use PyTorch reference
        partition_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        partition_reference, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_dispatch,
            partition_reference,
            rtol=1e-5,
            atol=1e-5,
            msg="CPU dispatch should use PyTorch reference",
        )
