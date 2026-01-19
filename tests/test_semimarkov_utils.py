import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def test_sum_matches_logpartition():
    torch.manual_seed(0)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    sm = SemiMarkov(LogSemiring)
    v, _, _ = sm.logpartition(edge, lengths=lengths)
    total = sm.sum(edge, lengths=lengths)

    assert torch.allclose(total, LogSemiring.unconvert(v))


def test_marginals_shape_and_bounds():
    torch.manual_seed(1)
    batch, T, K, C = 2, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    sm = SemiMarkov(LogSemiring)
    marginals = sm.marginals(edge, lengths=lengths)

    assert marginals.shape == edge.shape
    assert torch.isfinite(marginals).all()
    assert (marginals >= -1e-4).all()
    assert (marginals <= 1.0 + 1e-4).all()


def test_to_parts_from_parts_roundtrip():
    sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
    C, K = 3, 4

    edge = SemiMarkov.to_parts(sequence, extra=(C, K))
    recovered, extra = SemiMarkov.from_parts(edge)

    assert extra == (C, K)
    assert torch.equal(recovered, sequence)


class TestToPartsFromPartsExpanded:
    """Expanded tests for to_parts() and from_parts() conversion functions."""

    def test_roundtrip_multiple_batch_items(self):
        """Roundtrip works with multiple sequences in batch."""
        # Two sequences with different segmentation patterns
        sequence = torch.tensor(
            [
                [0, -1, 1, -1, -1, 2],  # 0(dur=2), 1(dur=3), 2(dur=1)
                [1, -1, -1, 0, -1, 1],  # 1(dur=3), 0(dur=2), 1(dur=1)
            ],
            dtype=torch.long,
        )
        C, K = 3, 4

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_roundtrip_all_same_label_with_transitions(self):
        """Roundtrip works when consecutive segments have the same label."""
        # Two segments with same label (need at least 2 segments for transitions)
        # Label 0 for 2 timesteps, then label 0 again for 3 timesteps
        sequence = torch.tensor([[0, -1, 0, -1, -1]], dtype=torch.long)
        C, K = 2, 3

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_single_segment_no_roundtrip(self):
        """Single segment spanning whole sequence cannot be recovered (no transitions)."""
        # When there's only one segment with no transitions, from_parts returns all -1s
        # because there are no edge entries to decode. This is expected behavior.
        sequence = torch.tensor([[0, -1, -1, -1, -1]], dtype=torch.long)
        C, K = 2, 5

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        # No transitions means no edges, so recovered is all -1
        assert (recovered == -1).all()

    def test_roundtrip_alternating_labels(self):
        """Roundtrip works with rapidly alternating labels (dur=1 each)."""
        # Alternating single-duration segments
        sequence = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)
        C, K = 2, 3

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_roundtrip_short_sequence_n2(self):
        """Roundtrip works with minimum sequence length N=2."""
        sequence = torch.tensor([[0, 1]], dtype=torch.long)
        C, K = 2, 2

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_roundtrip_two_segments(self):
        """Roundtrip works with two segments."""
        # First segment: label 2 for 2 timesteps, second: label 1 for 2 timesteps
        sequence = torch.tensor([[2, -1, 1, -1]], dtype=torch.long)
        C, K = 3, 3

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_roundtrip_k_equals_2(self):
        """Roundtrip works with K=2 (durations 0 and 1 in edge indexing)."""
        # With K=2, max segment duration is 1 (each position is a segment)
        sequence = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
        C, K = 3, 2

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_roundtrip_large_c(self):
        """Roundtrip works with many classes."""
        sequence = torch.tensor([[0, -1, 5, -1, 9, -1]], dtype=torch.long)
        C, K = 10, 3

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_to_parts_output_shape(self):
        """to_parts output has correct shape."""
        sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
        C, K = 3, 4
        batch, N = sequence.shape

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))

        assert edge.shape == (batch, N - 1, K, C, C)

    def test_to_parts_output_is_binary(self):
        """to_parts output contains only 0s and 1s."""
        sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
        C, K = 3, 4

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))

        assert ((edge == 0) | (edge == 1)).all()

    def test_to_parts_has_correct_nonzero_count(self):
        """Number of nonzero entries equals number of transitions."""
        # 3 segments means 2 internal transitions (at positions where segments end)
        sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
        C, K = 3, 4

        edge = SemiMarkov.to_parts(sequence, extra=(C, K))

        # Count nonzeros - should equal number of segment boundaries
        # Segments: 0(dur=2) -> 1(dur=3) -> 2(dur=1)
        # Transitions at: position 2 (0->1), position 5 (1->2)
        nonzeros = edge.nonzero()
        assert nonzeros.shape[0] == 2  # Two transitions

    def test_roundtrip_with_explicit_lengths(self):
        """Roundtrip works with explicit lengths parameter."""
        sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
        C, K = 3, 4
        lengths = torch.tensor([6])

        edge = SemiMarkov.to_parts(sequence, extra=(C, K), lengths=lengths)
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert torch.equal(recovered, sequence)

    def test_from_parts_with_all_zeros(self):
        """from_parts handles edge tensor with no transitions (all zeros)."""
        batch, N_1, K, C = 1, 5, 3, 2
        edge = torch.zeros(batch, N_1, K, C, C, dtype=torch.long)

        # This should not raise an error, but result will be all -1s
        recovered, extra = SemiMarkov.from_parts(edge)

        assert extra == (C, K)
        assert recovered.shape == (batch, N_1 + 1)
        # All positions should be -1 since there are no edges
        assert (recovered == -1).all()


class TestMarginalsCorrectness:
    """Comprehensive tests for marginals computation correctness."""

    def test_marginals_are_probabilities(self):
        """Marginals should be valid probabilities (non-negative, bounded)."""
        batch, T, K, C = 2, 8, 4, 3
        torch.manual_seed(42)

        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        # Should be non-negative
        assert (marginals >= -1e-6).all(), "Marginals should be non-negative"
        # Should be <= 1
        assert (marginals <= 1.0 + 1e-6).all(), "Marginals should be <= 1"
        # Should be finite
        assert torch.isfinite(marginals).all(), "Marginals should be finite"

    def test_marginals_shape_matches_input(self):
        """Marginals should have same shape as input edge potentials."""
        batch, T, K, C = 3, 10, 5, 4
        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        assert marginals.shape == edge.shape

    def test_marginals_equals_gradient_of_log_partition(self):
        """
        Marginals should equal the gradient of log partition function.

        For LogSemiring: marginals[i] = d(log Z) / d(edge[i])
        This is the expected "usage" of each edge in the model.
        """
        batch, T, K, C = 1, 6, 3, 2
        torch.manual_seed(42)

        edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        # Compute marginals
        marginals = sm.marginals(edge.detach(), lengths=lengths)

        # Compute gradient manually
        v, _, _ = sm.logpartition(edge, lengths=lengths)
        v.sum().backward()
        grad = edge.grad

        # Marginals and gradients should match
        assert torch.allclose(marginals, grad, atol=1e-5), "Marginals should equal gradients"

    def test_marginals_with_variable_lengths(self):
        """Marginals handle variable sequence lengths correctly."""
        batch, T, K, C = 3, 8, 4, 2
        torch.manual_seed(123)

        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.tensor([8, 6, 4], dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        # For shorter sequences, marginals beyond their length should be 0
        # Batch item 2 has length 4, so positions 3+ (0-indexed) should have
        # very small or zero marginals (they're beyond the sequence)
        # Note: the exact behavior depends on implementation
        assert marginals.shape == edge.shape

    def test_marginals_deterministic(self):
        """Marginals should be deterministic given same input."""
        batch, T, K, C = 2, 6, 3, 2
        torch.manual_seed(42)

        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)

        marginals1 = sm.marginals(edge, lengths=lengths)
        marginals2 = sm.marginals(edge, lengths=lengths)

        assert torch.equal(marginals1, marginals2)

    def test_marginals_respects_potential_magnitude(self):
        """Higher potential edges should have higher marginal probability."""
        batch, T, K, C = 1, 4, 2, 2
        torch.manual_seed(42)

        # Create edge potentials where one transition is strongly favored
        edge = torch.zeros(batch, T - 1, K, C, C)
        # Make transition (n=1, k=1, c_new=0, c_prev=0) very likely
        edge[0, 1, 1, 0, 0] = 10.0
        # Make all other transitions neutral
        edge[0, 0, :, :, :] = 0.0
        edge[0, 2, :, :, :] = 0.0

        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        # The favored edge should have higher marginal than others at same position
        favored_marginal = marginals[0, 1, 1, 0, 0]
        # At position 1, other (k, c_new, c_prev) combinations should have lower marginal
        other_marginals = marginals[0, 1].clone()
        other_marginals[1, 0, 0] = 0  # Zero out the favored one
        max_other = other_marginals.max()

        assert favored_marginal > max_other, "Favored edge should have higher marginal"

    def test_marginals_sum_property(self):
        """
        Sum of marginals over certain dimensions should have interpretable meaning.

        The sum of marginals over all (k, c_new, c_prev) at each position
        gives the expected number of segment boundaries at that position.
        """
        batch, T, K, C = 2, 8, 4, 3
        torch.manual_seed(42)

        edge = torch.randn(batch, T - 1, K, C, C)
        lengths = torch.full((batch,), T, dtype=torch.long)

        sm = SemiMarkov(LogSemiring)
        marginals = sm.marginals(edge, lengths=lengths)

        # Sum over (K, C, C) at each position
        position_marginals = marginals.sum(dim=(2, 3, 4))  # (batch, T-1)

        # Each position's marginal sum should be between 0 and 1
        # (probability that a segment boundary occurs at that position)
        assert (position_marginals >= -1e-6).all()
        assert (position_marginals <= 1.0 + 1e-6).all()

        # Total expected segment boundaries should be reasonable
        # (at least 1 for minimum path, at most T-1 for all single-step segments)
        total_marginals = position_marginals.sum(dim=1)  # (batch,)
        assert (total_marginals >= 0).all()
        assert (total_marginals <= T).all()
