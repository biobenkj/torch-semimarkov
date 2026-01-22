"""Extended edge case tests for Semi-Markov CRF.

Tests identified gaps in coverage from code review audit, focusing on:
- Numerical stability at boundary conditions
- Untested code paths (boundary projections, 3D transitions)
- Edge cases in gold scoring and traceback

These tests complement test_edge_cases.py which covers K=1, T=1, C=1 configurations.
"""

import pytest
import torch

from torch_semimarkov import SemiMarkov, SemiMarkovCRFHead
from torch_semimarkov.duration import (
    CallableDuration,
    GeometricDuration,
    NegativeBinomialDuration,
    PoissonDuration,
    UniformDuration,
    create_duration_distribution,
)
from torch_semimarkov.semirings import LogSemiring, MaxSemiring
from torch_semimarkov.streaming import semi_crf_streaming_forward_pytorch


class TestDurationDistributionEdgeCases:
    """Critical: Numerical stability in duration distributions."""

    def test_geometric_extreme_logit_positive(self):
        """p->1 when logit is very large should produce finite values."""
        dur = GeometricDuration(max_duration=10, num_classes=2, init_logit=50.0)
        bias = dur()
        assert torch.isfinite(bias).all(), f"Got non-finite values: {bias}"

    def test_geometric_extreme_logit_negative(self):
        """p->0 when logit is very negative should produce finite values."""
        dur = GeometricDuration(max_duration=10, num_classes=2, init_logit=-50.0)
        bias = dur()
        assert torch.isfinite(bias).all(), f"Got non-finite values: {bias}"

    def test_geometric_gradient_flow(self):
        """Gradients should flow through GeometricDuration."""
        dur = GeometricDuration(max_duration=10, num_classes=2, learn_rate=True)
        bias = dur()
        loss = bias.sum()
        loss.backward()
        assert dur.logit_p.grad is not None
        assert torch.isfinite(dur.logit_p.grad).all()

    def test_negative_binomial_small_r(self):
        """Small r values should produce finite results (except possibly k=1).

        Note: Very small r (e.g., log_r=-20) can cause lgamma overflow at k=1.
        This test uses a moderately small r that still stresses the computation.
        """
        dur = NegativeBinomialDuration(max_duration=10, num_classes=2, init_log_r=-5.0)
        bias = dur()
        # Most values should be finite; k=1 with very small r may overflow
        assert torch.isfinite(bias[1:]).all(), f"Got non-finite values with small r: {bias}"

    def test_negative_binomial_very_small_r_warns(self):
        """Very small r should emit a warning about numerical instability."""
        import warnings

        dur = NegativeBinomialDuration(max_duration=10, num_classes=2, init_log_r=-20.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dur()  # Call to trigger warning
            # Should have emitted a warning
            assert len(w) == 1
            assert "non-finite values" in str(w[0].message)
            assert "NegativeBinomialDuration" in str(w[0].message)

    def test_negative_binomial_large_r(self):
        """Large r should produce finite values."""
        dur = NegativeBinomialDuration(max_duration=10, num_classes=2, init_log_r=10.0)
        bias = dur()
        assert torch.isfinite(bias).all(), f"Got non-finite values with large r: {bias}"

    def test_poisson_zero_lambda(self):
        """lambda->0 should not cause log(0)."""
        dur = PoissonDuration(max_duration=10, num_classes=2, init_log_lambda=-50.0)
        bias = dur()
        assert torch.isfinite(bias).all(), f"Got non-finite values with zero lambda: {bias}"

    def test_poisson_large_lambda(self):
        """Large lambda could cause overflow in k * log(lambda)."""
        dur = PoissonDuration(max_duration=10, num_classes=2, init_log_lambda=20.0)
        bias = dur()
        assert torch.isfinite(bias).all(), f"Got non-finite values with large lambda: {bias}"

    def test_callable_duration_custom_function(self):
        """CallableDuration should work with custom function."""

        def my_duration(K, C, device):
            # Custom: prefer duration K//2
            k = torch.arange(K, device=device, dtype=torch.float32)
            bias = -((k - K // 2) ** 2) / 10.0
            return bias.unsqueeze(-1).expand(K, C)

        dur = CallableDuration(max_duration=10, num_classes=4, func=my_duration)
        bias = dur()
        assert bias.shape == (10, 4)
        assert torch.isfinite(bias).all()

    def test_uniform_duration_all_zeros(self):
        """UniformDuration should return all zeros."""
        dur = UniformDuration(max_duration=10, num_classes=4)
        bias = dur()
        assert torch.all(bias == 0)

    def test_create_duration_distribution_unknown_raises(self):
        """Unknown distribution name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown duration distribution"):
            create_duration_distribution("unknown_type", 10, 4)

    def test_create_duration_distribution_passthrough(self):
        """Existing DurationDistribution instance should pass through."""
        existing = GeometricDuration(10, 4)
        result = create_duration_distribution(existing, 10, 4)
        assert result is existing


class TestGoldScoringEdgeCases:
    """Critical: Edge cases in score_gold_vectorized."""

    def test_segment_exceeds_max_duration(self):
        """Segment longer than K should clamp duration index."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=5)  # K=5
        # All same label = 1 segment of duration 10 > K
        labels = torch.zeros(1, 10, dtype=torch.long)
        hidden = torch.randn(1, 10, 4)
        lengths = torch.tensor([10])

        # Should not error, duration clamped to K-1=4
        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_alternating_labels_max_segments(self):
        """Alternating labels creates T segments (stress test)."""
        crf = SemiMarkovCRFHead(num_classes=2, max_duration=10)
        T = 100
        labels = (torch.arange(T) % 2).unsqueeze(0)  # [0,1,0,1,...]
        hidden = torch.randn(1, T, 2)
        lengths = torch.tensor([T])

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Loss should be finite with max segments, got {loss}"

    def test_single_segment_no_transitions(self):
        """All-same-label sequence has no transition scores."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=10)
        # Zero out transitions to verify they don't contribute
        crf.transition.data.fill_(100.0)  # Large value

        labels = torch.zeros(1, 5, dtype=torch.long)  # All label 0
        hidden = torch.zeros(1, 5, 4)
        hidden[:, :, 0] = 1.0  # Score for label 0
        lengths = torch.tensor([5])

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False, reduction="none")
        # Loss should not include the large transition values (only 1 segment)
        assert loss.item() < 1000, f"Loss unexpectedly high: {loss.item()}"

    def test_duration_index_boundary(self):
        """Duration k should use duration_bias[k], not duration_bias[k-1]."""
        crf = SemiMarkovCRFHead(num_classes=2, max_duration=10)
        # Set distinctive values for duration_bias
        crf.duration_dist.duration_bias.data.fill_(0.0)
        crf.duration_dist.duration_bias.data[3, 0] = 10.0  # Duration 3, label 0
        crf.transition.data.fill_(0.0)

        # Create segment of duration 3 with label 0
        labels = torch.tensor([[0, 0, 0]])  # Duration 3
        hidden = torch.zeros(1, 3, 2)
        lengths = torch.tensor([3])

        result = crf(hidden, lengths, use_triton=False)
        gold_score = crf._score_gold(result["cum_scores"], labels, lengths)

        # Gold score should include duration_bias[3, 0] = 10.0
        assert gold_score.item() >= 9.9, f"Expected gold_score >= 10, got {gold_score.item()}"

    def test_batch_with_different_segment_counts(self):
        """Batch with varying numbers of segments per sequence."""
        crf = SemiMarkovCRFHead(num_classes=3, max_duration=10)
        batch = 3
        T = 12

        # Sequence 0: 1 segment (all same label)
        # Sequence 1: 3 segments
        # Sequence 2: 6 segments
        labels = torch.zeros(batch, T, dtype=torch.long)
        labels[1, 4:8] = 1
        labels[1, 8:] = 2
        labels[2, :2] = 0
        labels[2, 2:4] = 1
        labels[2, 4:6] = 2
        labels[2, 6:8] = 0
        labels[2, 8:10] = 1
        labels[2, 10:] = 2

        hidden = torch.randn(batch, T, 3)
        lengths = torch.tensor([T, T, T])

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"


class TestBoundaryProjections:
    """High: Untested proj_start/proj_end API."""

    def test_proj_start_affects_partition(self):
        """Start boundary projection should change partition."""
        batch, T, C, K = 2, 20, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        # Without boundaries
        part1, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # With proj_start
        proj_start = torch.randn(batch, T, C)
        part2, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, proj_start=proj_start
        )

        assert not torch.allclose(part1, part2), "proj_start should affect partition"

    def test_proj_end_affects_partition(self):
        """End boundary projection should change partition."""
        batch, T, C, K = 2, 20, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        # Without boundaries
        part1, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # With proj_end
        proj_end = torch.randn(batch, T, C)
        part2, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, proj_end=proj_end
        )

        assert not torch.allclose(part1, part2), "proj_end should affect partition"

    def test_combined_boundary_projections(self):
        """Both proj_start and proj_end together."""
        batch, T, C, K = 2, 20, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)
        proj_start = torch.randn(batch, T, C)
        proj_end = torch.randn(batch, T, C)

        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        assert torch.isfinite(partition).all(), f"Partition should be finite: {partition}"

    def test_boundary_projections_variable_lengths(self):
        """Boundary projections with variable sequence lengths."""
        batch, T, C, K = 3, 20, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.tensor([20, 15, 10])
        proj_start = torch.randn(batch, T, C)
        proj_end = torch.randn(batch, T, C)

        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        assert torch.isfinite(partition).all(), f"Partition should be finite: {partition}"


class TestDurationDependentTransitions:
    """High: Untested 3D transition tensor code path."""

    def test_3d_transition_forward(self):
        """Duration-dependent transitions (K, C, C) should work."""
        batch, T, C, K = 2, 20, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(K, C, C)  # 3D!
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        assert torch.isfinite(partition).all(), f"Partition should be finite: {partition}"

    def test_3d_transition_different_per_duration(self):
        """Different transition matrices for different durations."""
        batch, T, C, K = 1, 10, 3, 5
        cum_scores = torch.randn(batch, T + 1, C)
        # Make transitions very different per duration
        transition = torch.zeros(K, C, C)
        for k in range(K):
            transition[k] = torch.eye(C) * (k + 1)  # Diagonal scaling
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        assert torch.isfinite(partition).all()

    def test_3d_transition_with_boundary_projections(self):
        """3D transitions combined with boundary projections."""
        batch, T, C, K = 2, 15, 4, 6
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(K, C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)
        proj_start = torch.randn(batch, T, C)
        proj_end = torch.randn(batch, T, C)

        partition, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        assert torch.isfinite(partition).all()


class TestTracebackEdgeCases:
    """High: Viterbi traceback boundary conditions."""

    def test_sequence_exceeds_max_traceback_length(self):
        """Very long sequences should skip traceback but compute score."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(1, 100, 4)
        lengths = torch.tensor([100])

        result = crf.decode_with_traceback(hidden, lengths, max_traceback_length=50)

        # Score should still be computed
        assert torch.isfinite(result.scores).all()
        # But segments list should be empty
        assert len(result.segments[0]) == 0

    def test_mixed_traceback_lengths_in_batch(self):
        """Batch with some sequences exceeding max_traceback_length."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(3, 100, 4)
        lengths = torch.tensor([100, 40, 80])  # 100 and 80 exceed limit of 50

        result = crf.decode_with_traceback(hidden, lengths, max_traceback_length=50)

        # All scores should be computed
        assert torch.isfinite(result.scores).all()
        # Sequences 0 and 2 should have empty segments
        assert len(result.segments[0]) == 0
        assert len(result.segments[2]) == 0
        # Sequence 1 (length 40 < 50) should have segments
        assert len(result.segments[1]) > 0

    def test_traceback_single_position_sequence(self):
        """T=1 traceback should produce exactly one segment."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(2, 1, 4)
        lengths = torch.tensor([1, 1])

        result = crf.decode_with_traceback(hidden, lengths)

        for b in range(2):
            assert len(result.segments[b]) == 1
            seg = result.segments[b][0]
            assert seg.start == 0
            assert seg.end == 0
            assert seg.duration == 1


class TestHSMMConversion:
    """Medium: HSMM parameter conversion."""

    def test_hsmm_batch_init(self):
        """init_z_1 can be (batch, C) instead of (C,)."""
        C, K, N, batch = 4, 8, 50, 2
        init = torch.randn(batch, C)  # Per-batch init
        trans_z = torch.randn(C, C)
        trans_l = torch.randn(C, K)
        emission = torch.randn(batch, N, K, C)

        edge = SemiMarkov.hsmm(init, trans_z, trans_l, emission)
        assert edge.shape == (batch, N, K, C, C)

    def test_hsmm_logpartition_integration(self):
        """HSMM-converted edges should produce valid partition.

        Note: SemiMarkov.hsmm returns edge tensor of shape (batch, N, K, C, C)
        but logpartition expects (batch, N-1, K, C, C) where N is sequence length.
        So edge[:, :N-1, :, :, :] corresponds to a sequence of length N.
        """
        C, K, N, batch = 4, 8, 20, 2
        init = torch.log_softmax(torch.randn(C), dim=-1)
        trans_z = torch.log_softmax(torch.randn(C, C), dim=-1)
        trans_l = torch.log_softmax(torch.randn(C, K), dim=-1)
        emission = torch.randn(batch, N, K, C)

        edge = SemiMarkov.hsmm(init, trans_z, trans_l, emission)
        # edge shape is (batch, N, K, C, C)
        # logpartition expects (batch, N-1, K, C, C) for sequence length N
        # So we use edge directly and set lengths = N+1
        lengths = torch.full((batch,), N + 1)

        sm = SemiMarkov(LogSemiring)
        log_Z, _, _ = sm.logpartition(edge, lengths=lengths)

        assert torch.isfinite(log_Z).all()


class TestPartsConversion:
    """Medium: to_parts/from_parts round-trip."""

    def test_to_parts_from_parts_roundtrip(self):
        """to_parts and from_parts should be inverses."""
        # Sequence: label 0 for 2 positions, label 1 for 3, label 2 for 1
        seq = torch.tensor([[0, -1, 1, -1, -1, 2]])
        C, K = 3, 4

        edge = SemiMarkov.to_parts(seq, (C, K))
        seq_recovered, (C_out, K_out) = SemiMarkov.from_parts(edge)

        assert C_out == C
        assert K_out == K
        assert torch.equal(seq, seq_recovered)

    def test_to_parts_multiple_batches(self):
        """to_parts should handle batched sequences."""
        batch = 3
        N = 8
        C, K = 4, 6

        # Create sequences with different segmentations
        seq = torch.full((batch, N), -1, dtype=torch.long)
        seq[0, 0] = 0
        seq[0, 3] = 1
        seq[0, 6] = 2
        seq[1, 0] = 1
        seq[1, 4] = 2
        seq[2, 0] = 3

        edge = SemiMarkov.to_parts(seq, (C, K))
        assert edge.shape == (batch, N - 1, K, C, C)


class TestSemiringEdgeCases:
    """Medium: Semiring operations at boundaries."""

    def test_logsemiring_all_neginf(self):
        """logsumexp of all -inf should return -inf."""
        x = torch.full((2, 3), float("-inf"))
        result = LogSemiring.sum(x, dim=-1)
        assert torch.all(result == float("-inf"))

    def test_maxsemiring_with_ties(self):
        """MaxSemiring should handle ties deterministically."""
        x = torch.tensor([[1.0, 1.0, 1.0]])
        result = MaxSemiring.sum(x, dim=-1)
        assert result.item() == 1.0

    def test_logsemiring_mixed_finite_neginf(self):
        """LogSemiring should handle mix of finite and -inf."""
        x = torch.tensor([[1.0, float("-inf"), 2.0]])
        result = LogSemiring.sum(x, dim=-1)
        # logsumexp([1, -inf, 2]) = log(e^1 + 0 + e^2) = log(e + e^2)
        expected = torch.logsumexp(x, dim=-1)
        assert torch.allclose(result, expected)


class TestNonContiguousTensors:
    """Medium: Non-contiguous tensor handling."""

    def test_forward_with_non_contiguous_input(self):
        """Forward should work with non-contiguous tensors."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        # Create non-contiguous tensor by selecting every other element from a larger tensor
        hidden_large = torch.randn(4, 40, 4)
        hidden = hidden_large[:, ::2, :]  # Select every other time step -> (4, 20, 4)
        assert not hidden.is_contiguous(), "Expected non-contiguous tensor"
        lengths = torch.full((4,), 20)

        result = crf(hidden, lengths, use_triton=False)
        assert torch.isfinite(result["partition"]).all()

    def test_loss_with_non_contiguous_labels(self):
        """Loss computation with non-contiguous label tensor."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(2, 20, 4)
        lengths = torch.full((2,), 20)
        # Create non-contiguous labels
        labels_full = torch.randint(0, 4, (20, 2))
        labels = labels_full.T  # Now (2, 20) but non-contiguous
        assert not labels.is_contiguous()

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)
        assert torch.isfinite(loss)


class TestCheckpointInterval:
    """Low: Checkpoint interval edge cases."""

    def test_checkpoint_interval_less_than_k(self):
        """Checkpoint interval should be at least K."""
        batch, T, C, K = 2, 100, 4, 20
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        # Request interval < K, should be clamped to K
        partition, _, actual_interval = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, checkpoint_interval=5
        )

        assert actual_interval >= K, f"Interval {actual_interval} should be >= K={K}"
        assert torch.isfinite(partition).all()

    def test_checkpoint_interval_very_large(self):
        """Very large checkpoint interval should still work."""
        batch, T, C, K = 2, 50, 4, 8
        cum_scores = torch.randn(batch, T + 1, C)
        transition = torch.randn(C, C)
        duration_bias = torch.randn(K, C)
        lengths = torch.full((batch,), T)

        # Request interval larger than T
        partition, ring_checkpoints, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, checkpoint_interval=1000
        )

        assert torch.isfinite(partition).all()
        # Should have minimal checkpoints
        assert ring_checkpoints.shape[1] == 1
