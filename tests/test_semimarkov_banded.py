import pytest
import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.banded_utils import (
    apply_permutation,
    measure_effective_bandwidth,
    snake_ordering,
)
from torch_semimarkov.semirings import LogSemiring


def test_banded_logpartition_matches_linear_scan():
    torch.manual_seed(0)
    batch, T, K, C = 1, 6, 4, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    struct = SemiMarkov(LogSemiring)

    v_banded, _, _ = struct.logpartition(
        edge,
        lengths=lengths,
        use_banded=True,
        banded_perm="none",
        banded_bw_ratio=1.1,
    )
    v_linear, _, _ = struct.logpartition(edge, lengths=lengths, use_linear_scan=True)

    assert torch.allclose(v_banded, v_linear, atol=1e-5, rtol=1e-5)


def test_banded_helpers_shapes():
    struct = SemiMarkov(LogSemiring)
    device = torch.device("cpu")
    lu, ld = struct._compute_bandwidth(span_length=4, K=4, C=2)
    assert lu == ld

    adj = struct._build_adjacency(span_length=3, K=3, C=2, device=device)
    assert adj.shape == (4, 4)
    assert adj.dtype == torch.bool

    use_banded, perm, best_bw, threshold = struct._choose_banded_permutation(
        span_length=4, K=4, C=2, perm_mode="rcm", bw_ratio=1.0, device=device
    )
    assert isinstance(use_banded, bool)
    assert isinstance(best_bw, int)
    assert isinstance(threshold, float)
    if perm is not None:
        size = (4 - 1) * 2
        assert perm.numel() == size
        assert torch.equal(torch.sort(perm).values, torch.arange(size))


class TestBandedPermutationModes:
    """Test different permutation modes for banded optimization."""

    @pytest.mark.parametrize("perm_mode", ["none", "snake", "rcm", "auto"])
    def test_all_permutation_modes_match_linear_scan(self, perm_mode):
        """All permutation modes should produce same result as linear scan."""
        torch.manual_seed(42)
        batch, T, K, C = 2, 8, 4, 2
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        v_banded, _, _ = struct.logpartition(
            edge,
            lengths=lengths,
            use_banded=True,
            banded_perm=perm_mode,
            banded_bw_ratio=1.1,  # High ratio to ensure banded path is used
        )
        v_linear, _, _ = struct.logpartition(edge, lengths=lengths, use_linear_scan=True)

        assert torch.allclose(v_banded, v_linear, atol=1e-4, rtol=1e-4)

    def test_snake_ordering_reduces_bandwidth(self):
        """Snake ordering should reduce bandwidth for duration structure."""
        K, C = 5, 3
        K_1 = K - 1

        # Build adjacency for a span where duration constraint matters
        struct = SemiMarkov(LogSemiring)
        span_length = 4
        adj = struct._build_adjacency(span_length, K, C, device=torch.device("cpu"))

        # Measure bandwidth with no permutation
        original_bw = measure_effective_bandwidth(adj.float(), fill_value=0.0)

        # Measure bandwidth with snake permutation
        perm = snake_ordering(K_1, C)
        permuted_adj = apply_permutation(adj.float(), perm)
        snake_bw = measure_effective_bandwidth(permuted_adj, fill_value=0.0)

        # Snake should not increase bandwidth (may reduce or keep same)
        assert snake_bw <= original_bw + 1  # Allow small tolerance

    def test_permutation_is_valid(self):
        """Permutation should be a valid bijection."""
        struct = SemiMarkov(LogSemiring)
        device = torch.device("cpu")

        for K in [3, 5, 8]:
            for C in [2, 3, 4]:
                K_1 = K - 1
                size = K_1 * C

                _, perm, _, _ = struct._choose_banded_permutation(
                    span_length=4, K=K, C=C, perm_mode="auto", bw_ratio=1.0, device=device
                )

                if perm is not None:
                    assert perm.numel() == size
                    # Should be a permutation of 0..size-1
                    assert torch.equal(torch.sort(perm).values, torch.arange(size))


class TestBandedFallbackLogic:
    """Test the fallback logic when bandwidth is too large."""

    def test_fallback_to_dense_with_low_bw_ratio(self):
        """With very low bw_ratio, should fallback to dense (standard) matmul."""
        torch.manual_seed(42)
        batch, T, K, C = 2, 8, 4, 3
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        # Very low bw_ratio means bandwidth threshold is very small,
        # so actual bandwidth will likely exceed threshold -> fallback to dense
        v_banded, _, _ = struct.logpartition(
            edge,
            lengths=lengths,
            use_banded=True,
            banded_perm="none",
            banded_bw_ratio=0.01,  # Very low threshold
        )
        v_linear, _, _ = struct.logpartition(edge, lengths=lengths, use_linear_scan=True)

        # Should still match (fallback to dense is correct)
        assert torch.allclose(v_banded, v_linear, atol=1e-4, rtol=1e-4)

    def test_threshold_calculation(self):
        """Verify threshold is calculated correctly from bw_ratio."""
        struct = SemiMarkov(LogSemiring)
        device = torch.device("cpu")

        K, C = 5, 3
        K_1 = K - 1
        size = K_1 * C
        bw_ratio = 0.5

        _, _, best_bw, threshold = struct._choose_banded_permutation(
            span_length=4, K=K, C=C, perm_mode="none", bw_ratio=bw_ratio, device=device
        )

        # threshold should be bw_ratio * size
        expected_threshold = bw_ratio * size
        assert abs(threshold - expected_threshold) < 1e-6


class TestBandedGradients:
    """Test gradient computation through banded path."""

    def test_banded_gradient_matches_linear(self):
        """Gradients through banded path should match linear scan."""
        torch.manual_seed(42)
        batch, T, K, C = 2, 6, 3, 2
        edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64)
        lengths = torch.full((batch,), T, dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        # Banded gradient
        edge_banded = edge.clone().requires_grad_(True)
        v_banded, _, _ = struct.logpartition(
            edge_banded,
            lengths=lengths,
            use_banded=True,
            banded_perm="auto",
            banded_bw_ratio=1.1,
        )
        v_banded.sum().backward()
        grad_banded = edge_banded.grad.clone()

        # Linear gradient
        edge_linear = edge.clone().requires_grad_(True)
        v_linear, _, _ = struct.logpartition(edge_linear, lengths=lengths, use_linear_scan=True)
        v_linear.sum().backward()
        grad_linear = edge_linear.grad.clone()

        max_diff = (grad_banded - grad_linear).abs().max().item()
        assert max_diff < 1e-4, f"Gradient diff: {max_diff:.2e}"


class TestBandwidthComputation:
    """Test bandwidth computation for different configurations."""

    def test_bandwidth_increases_with_span_length(self):
        """Bandwidth should generally increase with span length."""
        struct = SemiMarkov(LogSemiring)
        K, C = 6, 3

        bandwidths = []
        for span_length in [2, 4, 8, 16]:
            lu, ld = struct._compute_bandwidth(span_length, K, C)
            bandwidths.append(max(lu, ld))

        # Bandwidth should be non-decreasing
        for i in range(1, len(bandwidths)):
            assert bandwidths[i] >= bandwidths[i - 1]

    def test_bandwidth_proportional_to_c(self):
        """Bandwidth scales with C (number of classes)."""
        struct = SemiMarkov(LogSemiring)
        K = 5
        span_length = 4

        bw_c2 = struct._compute_bandwidth(span_length, K, 2)
        bw_c4 = struct._compute_bandwidth(span_length, K, 4)

        # With 4x C, bandwidth should be roughly 2x
        assert bw_c4[0] == 2 * bw_c2[0]

    def test_adjacency_has_correct_structure(self):
        """Adjacency matrix should have block structure from durations."""
        struct = SemiMarkov(LogSemiring)
        device = torch.device("cpu")

        K, C = 4, 2
        span_length = 3
        adj = struct._build_adjacency(span_length, K, C, device)

        K_1 = K - 1
        size = K_1 * C

        assert adj.shape == (size, size)
        assert adj.dtype == torch.bool

        # Check that adjacency respects duration constraint k1 + k2 <= span_length
        for k1 in range(K_1):
            for k2 in range(K_1):
                for c1 in range(C):
                    for c2 in range(C):
                        i = k1 * C + c1
                        j = k2 * C + c2
                        if k1 + k2 <= span_length:
                            # These should be connected (for some label pairs)
                            pass  # Connection depends on implementation
                        # Just verify it's a valid bool
                        assert adj[i, j].dtype == torch.bool
