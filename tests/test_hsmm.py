"""
Tests for the SemiMarkov.hsmm() static method.

The hsmm() method converts HSMM (Hidden Semi-Markov Model) parameters to edge scores
that can be used with the SemiMarkov inference algorithms.
"""

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


class TestHSMMBasic:
    """Basic functionality tests for hsmm()."""

    def test_hsmm_output_shape(self):
        """Output shape should be (batch, N, K, C, C)."""
        batch, N, K, C = 2, 10, 4, 3

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        assert edges.shape == (batch, N, K, C, C)

    def test_hsmm_output_dtype_matches_emission(self):
        """Output dtype should match emission dtype."""
        batch, N, K, C = 2, 5, 3, 2

        for dtype in [torch.float32, torch.float64]:
            init_z = torch.randn(C, dtype=dtype)
            trans_z_z = torch.randn(C, C, dtype=dtype)
            trans_z_l = torch.randn(C, K, dtype=dtype)
            emission = torch.randn(batch, N, K, C, dtype=dtype)

            edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

            assert edges.dtype == dtype

    def test_hsmm_batched_init(self):
        """init_z can be batched (batch, C) or unbatched (C,)."""
        batch, N, K, C = 2, 5, 3, 2

        # Unbatched init
        init_z_1d = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges_1d = SemiMarkov.hsmm(init_z_1d, trans_z_z, trans_z_l, emission)

        # Batched init (same value for all batches)
        init_z_2d = init_z_1d.unsqueeze(0).expand(batch, -1)
        edges_2d = SemiMarkov.hsmm(init_z_2d, trans_z_z, trans_z_l, emission)

        assert edges_1d.shape == edges_2d.shape
        assert torch.allclose(edges_1d, edges_2d)

    def test_hsmm_different_batch_inits(self):
        """Batched init with different values per batch item."""
        batch, N, K, C = 2, 5, 3, 2

        # Different init for each batch
        init_z = torch.randn(batch, C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        # init_z only affects position 0
        # Check that batch items differ at n=0 due to different inits
        assert edges.shape == (batch, N, K, C, C)
        # The difference should be in the first timestep only
        diff_at_0 = (edges[0, 0] - edges[1, 0]).abs().sum()
        diff_at_1 = (edges[0, 1] - edges[1, 1]).abs().sum()
        # At n=0, init contributes differently; at n>0, emission contributes differently
        # Both can differ, but init difference is isolated to n=0
        assert diff_at_0 > 0 or diff_at_1 > 0


class TestHSMMEdgeCases:
    """Edge case tests for hsmm()."""

    def test_hsmm_single_class(self):
        """Works with C=1 (single class/state)."""
        batch, N, K, C = 2, 5, 3, 1

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        assert edges.shape == (batch, N, K, C, C)
        assert torch.isfinite(edges).all()

    def test_hsmm_k_equals_2(self):
        """Works with K=2 (minimum duration span)."""
        batch, N, K, C = 2, 5, 2, 3

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        assert edges.shape == (batch, N, K, C, C)
        assert torch.isfinite(edges).all()

    def test_hsmm_batch_size_1(self):
        """Works with batch size 1."""
        batch, N, K, C = 1, 5, 3, 2

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        assert edges.shape == (batch, N, K, C, C)
        assert torch.isfinite(edges).all()

    def test_hsmm_short_sequence(self):
        """Works with N=2 (minimum sequence length for edge computation)."""
        batch, N, K, C = 2, 2, 3, 2

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        assert edges.shape == (batch, N, K, C, C)
        assert torch.isfinite(edges).all()


class TestHSMMCorrectness:
    """Correctness tests for hsmm() - verify the mathematical relationships."""

    def test_hsmm_init_only_affects_first_timestep(self):
        """Changing init_z should only change edges[:, 0, ...]."""
        batch, N, K, C = 2, 5, 3, 2

        init_z_1 = torch.randn(C)
        init_z_2 = torch.randn(C)  # Different init
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges_1 = SemiMarkov.hsmm(init_z_1, trans_z_z, trans_z_l, emission)
        edges_2 = SemiMarkov.hsmm(init_z_2, trans_z_z, trans_z_l, emission)

        # n=0 should differ
        assert not torch.allclose(edges_1[:, 0], edges_2[:, 0])
        # n>0 should be identical
        assert torch.allclose(edges_1[:, 1:], edges_2[:, 1:])

    def test_hsmm_transition_affects_all_timesteps(self):
        """Changing trans_z_z should affect all timesteps."""
        batch, N, K, C = 2, 5, 3, 2

        init_z = torch.randn(C)
        trans_z_z_1 = torch.randn(C, C)
        trans_z_z_2 = torch.randn(C, C)  # Different transitions
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges_1 = SemiMarkov.hsmm(init_z, trans_z_z_1, trans_z_l, emission)
        edges_2 = SemiMarkov.hsmm(init_z, trans_z_z_2, trans_z_l, emission)

        # All timesteps should differ
        for n in range(N):
            assert not torch.allclose(edges_1[:, n], edges_2[:, n])

    def test_hsmm_duration_affects_k_dimension(self):
        """Changing trans_z_l should affect the K dimension."""
        batch, N, K, C = 2, 5, 3, 2

        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l_1 = torch.randn(C, K)
        trans_z_l_2 = torch.randn(C, K)  # Different duration probs
        emission = torch.randn(batch, N, K, C)

        edges_1 = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l_1, emission)
        edges_2 = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l_2, emission)

        # All edges should differ (duration affects all)
        assert not torch.allclose(edges_1, edges_2)

    def test_hsmm_manual_computation(self):
        """Verify against manual computation for a simple case."""
        batch, N, K, C = 1, 2, 2, 2

        # Use specific values for easier verification
        init_z = torch.tensor([0.1, 0.2])  # log P(z_{-1})
        trans_z_z = torch.tensor([[0.3, 0.4], [0.5, 0.6]])  # log P(z_n | z_{n-1})
        trans_z_l = torch.tensor([[0.7, 0.8], [0.9, 1.0]])  # log P(l_n | z_n)
        emission = torch.zeros(batch, N, K, C)
        emission[0, 0, 0, 0] = 1.1
        emission[0, 0, 0, 1] = 1.2
        emission[0, 0, 1, 0] = 1.3
        emission[0, 0, 1, 1] = 1.4
        emission[0, 1, 0, 0] = 1.5
        emission[0, 1, 0, 1] = 1.6
        emission[0, 1, 1, 0] = 1.7
        emission[0, 1, 1, 1] = 1.8

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        # At n=0, c2=0, c1=0, k=0:
        # init_z[c1] + trans_z_z[c1, c2] + trans_z_l[c2, k] + emission[b, n, k, c2]
        # = init_z[0] + trans_z_z.T[0, 0] + trans_z_l.T[0, 0] + emission[0, 0, 0, 0]
        # = 0.1 + 0.3 + 0.7 + 1.1 = 2.2
        expected_0_0_0_0 = 0.1 + 0.3 + 0.7 + 1.1
        assert torch.isclose(edges[0, 0, 0, 0, 0], torch.tensor(expected_0_0_0_0), atol=1e-5)

        # At n=1, c2=1, c1=0, k=1:
        # No init contribution (n > 0)
        # trans_z_z.T[c2, c1] + trans_z_l.T[k, c2] + emission[b, n, k, c2]
        # = trans_z_z[c1, c2] + trans_z_l[c2, k] + emission[0, 1, 1, 1]
        # = 0.4 + 1.0 + 1.8 = 3.2
        expected_1_1_1_0 = 0.4 + 1.0 + 1.8
        assert torch.isclose(edges[0, 1, 1, 1, 0], torch.tensor(expected_1_1_1_0), atol=1e-5)


class TestHSMMIntegration:
    """Integration tests - verify hsmm() output works with SemiMarkov inference."""

    def test_hsmm_with_logpartition(self):
        """hsmm() output should work with logpartition()."""
        batch, N, K, C = 2, 10, 4, 3

        torch.manual_seed(42)
        init_z = torch.randn(C)
        trans_z_z = torch.randn(C, C)
        trans_z_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)

        # Use N-1 for edge potentials (standard SemiMarkov convention)
        # hsmm returns (batch, N, K, C, C), logpartition expects (batch, N-1, K, C, C)
        edge_potentials = edges[:, :-1]  # Remove last timestep

        sm = SemiMarkov(LogSemiring)
        lengths = torch.full((batch,), N, dtype=torch.long)
        v, potentials, _ = sm.logpartition(edge_potentials, lengths=lengths)

        # v has shape (ssize, batch) where ssize is semiring size (1 for LogSemiring)
        assert v.shape[-1] == batch
        assert torch.isfinite(v).all()

    def test_hsmm_gradient_flow(self):
        """Gradients should flow through hsmm() to all inputs."""
        batch, N, K, C = 2, 6, 3, 2

        init_z = torch.randn(C, requires_grad=True)
        trans_z_z = torch.randn(C, C, requires_grad=True)
        trans_z_l = torch.randn(C, K, requires_grad=True)
        emission = torch.randn(batch, N, K, C, requires_grad=True)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)
        edge_potentials = edges[:, :-1]

        sm = SemiMarkov(LogSemiring)
        lengths = torch.full((batch,), N, dtype=torch.long)
        v, _, _ = sm.logpartition(edge_potentials, lengths=lengths)

        loss = v.sum()
        loss.backward()

        # All inputs should have gradients
        assert init_z.grad is not None
        assert trans_z_z.grad is not None
        assert trans_z_l.grad is not None
        assert emission.grad is not None

        # Gradients should be finite
        assert torch.isfinite(init_z.grad).all()
        assert torch.isfinite(trans_z_z.grad).all()
        assert torch.isfinite(trans_z_l.grad).all()
        assert torch.isfinite(emission.grad).all()

    def test_hsmm_normalized_probabilities(self):
        """Test with properly normalized log-probabilities."""
        batch, N, K, C = 2, 8, 4, 3

        torch.manual_seed(123)
        # Normalize to proper log-probabilities
        init_z = torch.log_softmax(torch.randn(C), dim=0)
        trans_z_z = torch.log_softmax(torch.randn(C, C), dim=1)  # P(z_n | z_{n-1})
        trans_z_l = torch.log_softmax(torch.randn(C, K), dim=1)  # P(l | z)
        emission = torch.randn(batch, N, K, C)  # Log-likelihoods (not normalized)

        edges = SemiMarkov.hsmm(init_z, trans_z_z, trans_z_l, emission)
        edge_potentials = edges[:, :-1]

        sm = SemiMarkov(LogSemiring)
        lengths = torch.full((batch,), N, dtype=torch.long)
        v, _, _ = sm.logpartition(edge_potentials, lengths=lengths)

        assert torch.isfinite(v).all()
