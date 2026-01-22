"""
Tests for the Triton streaming forward kernel.

Validates the Triton implementation against the PyTorch reference.
"""

import pytest
import torch

from torch_semimarkov.streaming import (
    HAS_TRITON,
    semi_crf_streaming_forward_pytorch,
)

# Conditionally import Triton functions
if HAS_TRITON:
    from torch_semimarkov.streaming import (
        launch_streaming_triton_backward,
        launch_streaming_triton_kernel,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingKernel:
    """Test the Triton streaming forward kernel against PyTorch reference."""

    def test_triton_matches_pytorch_small(self):
        """Verify Triton kernel matches PyTorch for small inputs."""
        batch, T, K, C = 2, 20, 4, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_matches_pytorch_medium(self):
        """Verify Triton kernel matches PyTorch for medium inputs."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_matches_pytorch_large_C(self):
        """Verify Triton kernel works with larger C (requires padding)."""
        batch, T, K, C = 2, 50, 6, 24  # C=24 -> C_PAD=32
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_matches_pytorch_non_power_of_2_C(self):
        """Verify Triton kernel handles non-power-of-2 C values."""
        for C in [3, 5, 7, 11, 13, 17]:
            batch, T, K = 2, 30, 5
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, msg=f"C={C} failed"
            )

    def test_triton_variable_lengths(self):
        """Verify Triton kernel handles variable sequence lengths."""
        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_short_sequences(self):
        """Verify Triton kernel handles sequences shorter than K."""
        K, C, batch = 10, 4, 2

        for T in [4, 6, 8, 12]:
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, msg=f"T={T} failed"
            )

    def test_triton_max_semiring(self):
        """Verify Triton max semiring matches PyTorch."""
        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_produces_finite_values(self):
        """Verify Triton kernel produces finite values."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        assert torch.isfinite(partition_triton).all(), "Partition contains non-finite values"

    def test_triton_checkpoints_saved(self):
        """Verify Triton kernel saves checkpoints correctly."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        _, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Verify checkpoint shape
        expected_num_ckpts = (T + ckpt_interval - 1) // ckpt_interval + 1
        assert (
            ring_ckpts.shape[1] == expected_num_ckpts
        ), f"Expected {expected_num_ckpts} checkpoints"
        assert ring_ckpts.shape[2] == K
        assert ring_ckpts.shape[3] == C

        # Verify checkpoint 0 is initialized correctly (alpha[0] = 0, rest = -inf)
        ckpt_0 = ring_ckpts[:, 0, :, :]  # (batch, K, C)
        assert torch.allclose(ckpt_0[:, 0, :], torch.zeros_like(ckpt_0[:, 0, :]))

    def test_triton_larger_sequence(self):
        """Verify Triton kernel works with larger sequences."""
        batch, T, K, C = 2, 500, 16, 8
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Use slightly looser tolerance for longer sequences
        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingTraining:
    """Test the Triton streaming kernel in training mode (with gradients)."""

    def test_triton_forward_with_gradients(self):
        """Verify Triton forward works when gradients are needed."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # Use Triton path (full Triton forward + backward)
        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "cum_scores grad non-finite"
        assert torch.isfinite(transition.grad).all(), "transition grad non-finite"
        assert torch.isfinite(duration_bias.grad).all(), "duration_bias grad non-finite"

    def test_triton_gradients_match_pytorch(self):
        """Verify Triton path gradients match PyTorch reference."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path (full Triton forward + backward kernels)
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients (use looser tolerance for Triton backward)
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)

    def test_dispatch_inference_vs_training(self):
        """Verify correct dispatch based on requires_grad."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Inference (no grad) - should use Triton kernel
        partition_inf = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K
        )
        assert torch.isfinite(partition_inf).all()

        # Training (with grad)
        cum_scores.requires_grad_(True)
        partition_train = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K
        )
        assert torch.isfinite(partition_train).all()
        partition_train.sum().backward()
        assert cum_scores.grad is not None

    def test_triton_backward_kernel_raw(self):
        """Test the raw Triton backward kernel launcher."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Run forward
        partition, ring_checkpoints, checkpoint_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Run backward
        grad_output = torch.ones(batch, device="cuda")
        grad_cum_scores, grad_transition, grad_duration_bias, _, _ = (
            launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                checkpoint_interval,
                grad_output,
            )
        )

        # Verify shapes
        assert grad_cum_scores.shape == cum_scores.shape
        assert grad_transition.shape == transition.shape
        assert grad_duration_bias.shape == duration_bias.shape

        # Verify finite values
        assert torch.isfinite(grad_cum_scores).all(), "grad_cum_scores non-finite"
        assert torch.isfinite(grad_transition).all(), "grad_transition non-finite"
        assert torch.isfinite(grad_duration_bias).all(), "grad_duration_bias non-finite"

    def test_triton_gradients_variable_lengths(self):
        """Verify Triton gradients handle variable sequence lengths."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

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
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)

        # Compare gradients (looser tolerance)
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingBoundaries:
    """Test the Triton streaming kernel with boundary projections (Phase 4B)."""

    def create_boundary_inputs(self, batch, T, K, C, device="cuda", dtype=torch.float32, seed=42):
        """Create test inputs including boundary projections."""
        torch.manual_seed(seed)

        # Standard inputs
        projected = torch.randn(batch, T, C, device=device, dtype=dtype)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1
        duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        # Boundary projections
        proj_start = torch.randn(batch, T, C, device=device, dtype=dtype) * 0.1
        proj_end = torch.randn(batch, T, C, device=device, dtype=dtype) * 0.1

        return cum_scores, transition, duration_bias, lengths, proj_start, proj_end

    def test_triton_with_boundaries_forward(self):
        """Verify Triton forward with boundaries matches PyTorch."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_with_boundaries_forward_max(self):
        """Verify Triton max semiring with boundaries matches PyTorch."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="max",
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="max",
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_with_boundaries_gradients(self):
        """Verify Triton backward with boundaries matches PyTorch."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py, tr_py, db_py, lengths, K, proj_start=ps_py, proj_end=pe_py, use_triton=False
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr, tr_tr, db_tr, lengths, K, proj_start=ps_tr, proj_end=pe_tr, use_triton=True
        )
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=1e-2, atol=1e-2)

    def test_triton_boundaries_backward_kernel_raw(self):
        """Test the raw Triton backward kernel with boundaries."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # Run forward
        partition, ring_checkpoints, checkpoint_interval = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Run backward
        grad_output = torch.ones(batch, device="cuda")
        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end = (
            launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                checkpoint_interval,
                grad_output,
                proj_start=proj_start,
                proj_end=proj_end,
            )
        )

        # Verify shapes
        assert grad_cum_scores.shape == cum_scores.shape
        assert grad_transition.shape == transition.shape
        assert grad_duration_bias.shape == duration_bias.shape
        assert grad_proj_start.shape == proj_start.shape
        assert grad_proj_end.shape == proj_end.shape

        # Verify finite values
        assert torch.isfinite(grad_cum_scores).all(), "grad_cum_scores non-finite"
        assert torch.isfinite(grad_transition).all(), "grad_transition non-finite"
        assert torch.isfinite(grad_duration_bias).all(), "grad_duration_bias non-finite"
        assert torch.isfinite(grad_proj_start).all(), "grad_proj_start non-finite"
        assert torch.isfinite(grad_proj_end).all(), "grad_proj_end non-finite"

    def test_triton_boundaries_larger_C(self):
        """Verify boundaries work with C=24 (genomics scale)."""
        batch, T, K, C = 2, 50, 8, 24
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_boundaries_variable_lengths(self):
        """Verify boundaries handle variable sequence lengths."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=ps_py,
            proj_end=pe_py,
            use_triton=False,
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=ps_tr,
            proj_end=pe_tr,
            use_triton=True,
        )
        partition_tr.sum().backward()

        # Compare
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingBenchmark:
    """Benchmark tests for Triton streaming kernel."""

    @pytest.mark.parametrize("T", [100, 500, 1000])
    @pytest.mark.parametrize("K", [8, 16, 32])
    def test_benchmark_correctness(self, T, K):
        """Verify correctness across different T and K values."""
        batch, C = 4, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3, msg=f"T={T}, K={K} failed"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestDurationDependentTransitions:
    """Test duration-dependent transitions (Phase 4A)."""

    def create_duration_transition_inputs(
        self, batch, T, K, C, device="cuda", dtype=torch.float32, seed=42
    ):
        """Create test inputs with duration-dependent transitions (K, C, C)."""
        torch.manual_seed(seed)

        # Standard inputs
        projected = torch.randn(batch, T, C, device=device, dtype=dtype)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        # Duration-dependent transitions: (K, C, C)
        transition = torch.randn(K, C, C, device=device, dtype=dtype) * 0.1

        duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        return cum_scores, transition, duration_bias, lengths

    def test_pytorch_duration_transitions_forward(self):
        """Verify PyTorch forward pass works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Run forward pass
        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Basic sanity checks
        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition contains non-finite values"

    def test_triton_duration_transitions_forward(self):
        """Verify Triton forward kernel works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_duration_transitions_forward_max(self):
        """Verify Triton max semiring works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        # Triton kernel
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)

    def test_triton_duration_transitions_gradients(self):
        """Verify Triton backward gradients match PyTorch for (K, C, C) transitions."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
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
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)

    def test_duration_transitions_gradient_shape(self):
        """Verify gradient shape is (K, C, C) for duration-dependent transitions."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Triton path
        transition = transition.clone().requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )
        partition.sum().backward()

        # Verify gradient shape matches input shape
        assert transition.grad.shape == (
            K,
            C,
            C,
        ), f"Expected gradient shape (K, C, C) = ({K}, {C}, {C}), got {transition.grad.shape}"

    def test_duration_transitions_with_boundaries(self):
        """Verify (K, C, C) transitions work together with boundary projections."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Add boundary projections
        proj_start = torch.randn(batch, T, C, device="cuda") * 0.1
        proj_end = torch.randn(batch, T, C, device="cuda") * 0.1

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py,
            tr_py,
            duration_bias,
            lengths,
            K,
            proj_start=ps_py,
            proj_end=pe_py,
            use_triton=False,
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr,
            tr_tr,
            duration_bias,
            lengths,
            K,
            proj_start=ps_tr,
            proj_end=pe_tr,
            use_triton=True,
        )
        partition_tr.sum().backward()

        # Compare
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=1e-2, atol=1e-2)

    def test_static_vs_duration_transitions_shape(self):
        """Verify static (C, C) transitions still work after Phase 4A changes."""
        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)

        # Create inputs
        projected = torch.randn(batch, T, C, device="cuda")
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda")
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        # Static transition (C, C)
        transition_static = torch.randn(C, C, device="cuda") * 0.1
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # Forward pass with static transitions should still work
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition_static, duration_bias, lengths, K
        )
        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition_static, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestGradientScalingBugFix:
    """Test that shared parameter gradients are correctly weighted by grad_output.

    These tests specifically target the bug where:
    - BUGGY: grad = Σ_b(marginal[b]) × Σ_b(grad_output[b])
    - CORRECT: grad = Σ_b(marginal[b] × grad_output[b])

    The bug was masked because all tests used .sum().backward() which creates
    uniform grad_output = [1, 1, ..., 1], making both formulas equivalent.
    """

    def test_heterogeneous_grad_output_shared_params(self):
        """Verify shared parameter gradients with non-uniform grad_output."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Create heterogeneous grad_output
        grad_output = torch.tensor([0.5, 2.0], device="cuda")

        # Test Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.backward(grad_output)

        # Test PyTorch reference path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.backward(grad_output)

        # Both should produce identical gradients for shared parameters
        torch.testing.assert_close(
            tr_tr.grad,
            tr_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Transition gradient mismatch with heterogeneous grad_output",
        )
        torch.testing.assert_close(
            db_tr.grad,
            db_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Duration bias gradient mismatch with heterogeneous grad_output",
        )

    def test_gradient_linearity_in_grad_output(self):
        """Verify that gradients scale linearly with grad_output."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # First run with grad_output = [1.0, 1.0]
        tr_1 = transition.clone().requires_grad_(True)
        partition_1 = semi_crf_streaming_forward(
            cum_scores, tr_1, duration_bias, lengths, K, use_triton=True
        )
        partition_1.backward(torch.tensor([1.0, 1.0], device="cuda"))
        grad_1 = tr_1.grad.clone()

        # Second run with grad_output = [2.0, 2.0]
        tr_2 = transition.clone().requires_grad_(True)
        partition_2 = semi_crf_streaming_forward(
            cum_scores, tr_2, duration_bias, lengths, K, use_triton=True
        )
        partition_2.backward(torch.tensor([2.0, 2.0], device="cuda"))
        grad_2 = tr_2.grad.clone()

        # grad_2 should be exactly 2 * grad_1
        torch.testing.assert_close(
            grad_2,
            2.0 * grad_1,
            rtol=1e-5,
            atol=1e-5,
            msg="Gradient does not scale linearly with grad_output",
        )

    def test_zero_mask_gradient_isolation(self):
        """Verify that grad_output=0 completely masks a batch element's contribution.

        This is the critical test that directly catches the gradient scaling bug.
        When grad_output[1] = 0, the second batch element should contribute
        NOTHING to the shared parameter gradients.
        """
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        T, K, C = 30, 5, 4

        # Create different inputs for each batch element
        torch.manual_seed(42)
        cum_scores_0, transition, duration_bias, _ = create_streaming_inputs(
            1, T, K, C, device="cuda"
        )
        torch.manual_seed(123)  # Different seed for different data
        cum_scores_1, _, _, _ = create_streaming_inputs(1, T, K, C, device="cuda")

        # Combine into batch of 2
        cum_scores_batch = torch.cat([cum_scores_0, cum_scores_1], dim=0)
        lengths = torch.tensor([T, T], dtype=torch.long, device="cuda")

        # Run batch of 2 with mask [1.0, 0.0] - second element masked
        tr_batch = transition.clone().requires_grad_(True)
        partition_batch = semi_crf_streaming_forward(
            cum_scores_batch, tr_batch, duration_bias, lengths, K, use_triton=True
        )
        partition_batch.backward(torch.tensor([1.0, 0.0], device="cuda"))
        grad_tr_batch = tr_batch.grad.clone()

        # Run single-batch on first sequence only
        tr_single = transition.clone().requires_grad_(True)
        lengths_single = torch.tensor([T], dtype=torch.long, device="cuda")
        partition_single = semi_crf_streaming_forward(
            cum_scores_0, tr_single, duration_bias, lengths_single, K, use_triton=True
        )
        partition_single.backward(torch.ones(1, device="cuda"))
        grad_tr_single = tr_single.grad.clone()

        # MUST be exactly equal - masked sequence contributes nothing
        torch.testing.assert_close(
            grad_tr_batch,
            grad_tr_single,
            rtol=1e-4,
            atol=1e-4,
            msg="Masked sequence (grad_output=0) still contributed to gradient!",
        )

    def test_duration_dependent_gradient_scaling(self):
        """Verify gradient scaling works correctly with (K, C, C) transitions."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)

        # Create duration-dependent transitions
        projected = torch.randn(batch, T, C, device="cuda")
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda")
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        transition = torch.randn(K, C, C, device="cuda") * 0.1  # (K, C, C)
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # Test with heterogeneous grad_output
        grad_output = torch.tensor([0.3, 1.7], device="cuda")

        # Triton path
        tr_tr = transition.clone().requires_grad_(True)
        partition_tr = semi_crf_streaming_forward(
            cum_scores, tr_tr, duration_bias, lengths, K, use_triton=True
        )
        partition_tr.backward(grad_output)

        # PyTorch path
        tr_py = transition.clone().requires_grad_(True)
        partition_py = semi_crf_streaming_forward(
            cum_scores, tr_py, duration_bias, lengths, K, use_triton=False
        )
        partition_py.backward(grad_output)

        # Verify gradient shape is preserved
        assert tr_tr.grad.shape == (
            K,
            C,
            C,
        ), f"Expected gradient shape (K, C, C), got {tr_tr.grad.shape}"

        # Verify Triton matches PyTorch
        torch.testing.assert_close(
            tr_tr.grad,
            tr_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Duration-dependent transition gradients don't match",
        )


class TestTritonBackwardDebug:
    """Debug tests to diagnose Triton vs PyTorch gradient mismatches."""

    def test_triton_backward_debug_minimal(self):
        """Minimal debug test with tiny config to compare gradient values."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        # Tiny config for easy debugging
        batch, T, K, C = 1, 5, 2, 2
        torch.manual_seed(42)

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

        # Print detailed comparison
        print("\n" + "=" * 60)
        print("TRITON BACKWARD GRADIENT DEBUG")
        print("=" * 60)
        print(f"Config: batch={batch}, T={T}, K={K}, C={C}")
        print(f"Forward values match: {torch.allclose(partition_py, partition_tr)}")
        print(f"  PyTorch partition: {partition_py.item():.6f}")
        print(f"  Triton partition:  {partition_tr.item():.6f}")

        print("\n--- grad_cum_scores comparison ---")
        print(f"PyTorch:\n{cs_py.grad.squeeze()}")
        print(f"\nTriton:\n{cs_tr.grad.squeeze()}")
        print(f"\nDiff (Triton - PyTorch):\n{(cs_tr.grad - cs_py.grad).squeeze()}")
        print(f"\nMax abs diff: {(cs_tr.grad - cs_py.grad).abs().max().item():.6f}")

        print("\n--- grad_transition comparison ---")
        print(f"PyTorch:\n{tr_py.grad}")
        print(f"\nTriton:\n{tr_tr.grad}")
        print(f"\nDiff (Triton - PyTorch):\n{tr_tr.grad - tr_py.grad}")
        print(f"\nMax abs diff: {(tr_tr.grad - tr_py.grad).abs().max().item():.6f}")

        print("\n--- grad_duration_bias comparison ---")
        print(f"PyTorch:\n{db_py.grad}")
        print(f"\nTriton:\n{db_tr.grad}")
        print(f"\nDiff (Triton - PyTorch):\n{db_tr.grad - db_py.grad}")
        print(f"\nMax abs diff: {(db_tr.grad - db_py.grad).abs().max().item():.6f}")

        # Check if gradients match (this will fail, but shows the actual values)
        cs_match = torch.allclose(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        tr_match = torch.allclose(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        db_match = torch.allclose(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)

        print("\n--- Summary ---")
        print(f"cum_scores grad match: {cs_match}")
        print(f"transition grad match: {tr_match}")
        print(f"duration_bias grad match: {db_match}")
        print("=" * 60)

        # Don't assert - this is for debugging only
        if not (cs_match and tr_match and db_match):
            print("\nWARNING: Gradients don't match! Use output above to diagnose.")

    def test_triton_backward_debug_larger(self):
        """Larger debug test matching the actual failing test config."""
        from torch_semimarkov.streaming import semi_crf_streaming_forward

        # Same config as test_triton_gradients_match_pytorch
        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)

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

        # Statistical summary
        cs_diff = (cs_tr.grad - cs_py.grad).abs()
        tr_diff = (tr_tr.grad - tr_py.grad).abs()
        db_diff = (db_tr.grad - db_py.grad).abs()

        print("\n" + "=" * 60)
        print("TRITON BACKWARD GRADIENT DEBUG (LARGER)")
        print("=" * 60)
        print(f"Config: batch={batch}, T={T}, K={K}, C={C}")
        print(f"Forward values match: {torch.allclose(partition_py, partition_tr)}")

        print("\n--- grad_cum_scores statistics ---")
        print(f"Shape: {cs_py.grad.shape}")
        print(f"Max abs diff: {cs_diff.max().item():.6f}")
        print(f"Mean abs diff: {cs_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(cs_diff > 0.01).sum().item()} / {cs_diff.numel()}")

        # Show first few mismatched positions
        mismatch_mask = cs_diff > 0.01
        if mismatch_mask.any():
            idxs = torch.nonzero(mismatch_mask)[:5]
            print("First 5 mismatched positions:")
            for idx in idxs:
                b, t, c = idx.tolist()
                print(
                    f"  [{b}, {t}, {c}]: PyTorch={cs_py.grad[b, t, c]:.4f}, Triton={cs_tr.grad[b, t, c]:.4f}, diff={cs_diff[b, t, c]:.4f}"
                )

        print("\n--- grad_transition statistics ---")
        print(f"Shape: {tr_py.grad.shape}")
        print(f"Max abs diff: {tr_diff.max().item():.6f}")
        print(f"Mean abs diff: {tr_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(tr_diff > 0.01).sum().item()} / {tr_diff.numel()}")

        print("\n--- grad_duration_bias statistics ---")
        print(f"Shape: {db_py.grad.shape}")
        print(f"Max abs diff: {db_diff.max().item():.6f}")
        print(f"Mean abs diff: {db_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(db_diff > 0.01).sum().item()} / {db_diff.numel()}")
        print("=" * 60)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
    elif not HAS_TRITON:
        print("Triton not available, skipping tests")
    else:
        print("Running Triton streaming kernel validation...")

        # Quick validation
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        print("\nInput shapes:")
        print(f"  cum_scores: {cum_scores.shape}")
        print(f"  transition: {transition.shape}")
        print(f"  duration_bias: {duration_bias.shape}")
        print(f"  lengths: {lengths}")

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"\nPyTorch partition: {partition_pytorch}")

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"Triton partition:  {partition_triton}")
        print(f"Checkpoint interval: {ckpt_interval}")
        print(f"Ring checkpoints shape: {ring_ckpts.shape}")

        # Compare
        diff = (partition_triton - partition_pytorch).abs().max().item()
        print(f"\nMax difference: {diff:.6e}")

        if diff < 1e-4:
            print("PASSED: Triton matches PyTorch reference")
        else:
            print("FAILED: Triton does not match PyTorch reference")
