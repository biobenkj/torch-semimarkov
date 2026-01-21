"""Edge case tests for Semi-Markov CRF.

Tests extreme parameter configurations and boundary conditions:
- K=1 (HMM-like, all segments have duration 1)
- K=T (segments can span entire sequence)
- C=1 (single class)
- T=1 (single position)
- T=2 (minimal multi-position)

Also includes regression tests for bugs discovered during development.
"""

import pytest
import torch

from torch_semimarkov import SemiMarkovCRFHead


class TestEdgeCaseConfigurations:
    """Test extreme parameter configurations."""

    @pytest.mark.parametrize(
        "K,T,C,batch",
        [
            (1, 20, 4, 2),  # K=1: HMM-like (all segments duration 1)
            (2, 20, 4, 2),  # K=2: Minimum multi-duration support
            (20, 20, 4, 2),  # K=T: segments can span entire sequence
            (8, 20, 1, 2),  # C=1: single class (trivial)
            (8, 1, 4, 2),  # T=1: single position
            (8, 2, 4, 2),  # T=2: minimal multi-position
        ],
    )
    def test_edge_case_forward(self, K, T, C, batch):
        """Forward pass works for edge case configurations."""
        crf = SemiMarkovCRFHead(num_classes=C, max_duration=K)
        hidden = torch.randn(batch, T, C)
        lengths = torch.full((batch,), T)

        result = crf(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (batch,)
        assert torch.isfinite(result["partition"]).all()

    @pytest.mark.parametrize(
        "K,T,C,batch",
        [
            (1, 20, 4, 2),  # K=1: HMM-like (all segments duration 1)
            (2, 20, 4, 2),  # K=2: Minimum multi-duration support
            (20, 20, 4, 2),
            (8, 20, 1, 2),
            (8, 1, 4, 2),
            (8, 2, 4, 2),
        ],
    )
    def test_edge_case_decode(self, K, T, C, batch):
        """Viterbi decoding works for edge case configurations."""
        crf = SemiMarkovCRFHead(num_classes=C, max_duration=K)
        hidden = torch.randn(batch, T, C)
        lengths = torch.full((batch,), T)

        scores = crf.decode(hidden, lengths, use_triton=False)

        assert scores.shape == (batch,)
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize(
        "K,T,C,batch",
        [
            (1, 20, 4, 2),  # K=1: HMM-like (all segments duration 1)
            (2, 20, 4, 2),  # K=2: Minimum multi-duration support
            (20, 20, 4, 2),
            (8, 20, 1, 2),
            (8, 1, 4, 2),
            (8, 2, 4, 2),
        ],
    )
    def test_edge_case_loss(self, K, T, C, batch):
        """Loss computation works for edge case configurations."""
        crf = SemiMarkovCRFHead(num_classes=C, max_duration=K)
        hidden = torch.randn(batch, T, C)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, C, (batch, T))

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)

        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestK1Behavior:
    """Test K=1 (HMM-like) behavior where all segments must be duration 1.

    K=1 produces HMM-like behavior where the only valid segment duration is 1.
    This means each position is its own segment.
    """

    def test_k1_forces_unit_segments(self):
        """K=1 forces all segments to have duration 1 (HMM behavior)."""
        crf = SemiMarkovCRFHead(num_classes=3, max_duration=1)
        hidden = torch.randn(1, 10, 3)
        lengths = torch.tensor([10])

        result = crf.decode_with_traceback(hidden, lengths)

        for seg in result.segments[0]:
            assert seg.duration == 1, f"K=1 should force unit segments, got {seg.duration}"
        assert len(result.segments[0]) == 10  # One segment per position

    def test_k1_partition_finite(self):
        """K=1 produces finite partition function."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=1)
        hidden = torch.randn(2, 50, 4)
        lengths = torch.tensor([50, 30])

        result = crf(hidden, lengths, use_triton=False)

        assert torch.isfinite(result["partition"]).all()


class TestSingleClass:
    """Test C=1 (single class) behavior."""

    def test_single_class_all_same_label(self):
        """C=1 should produce segments all with label 0."""
        crf = SemiMarkovCRFHead(num_classes=1, max_duration=100)
        hidden = torch.randn(1, 50, 1)
        lengths = torch.tensor([50])

        result = crf.decode_with_traceback(hidden, lengths)

        # With C=1, all segments must have label 0
        for seg in result.segments[0]:
            assert seg.label == 0

        # Segments should cover entire sequence
        assert result.segments[0][0].start == 0
        assert result.segments[0][-1].end == 49

    def test_single_class_partition(self):
        """C=1 produces valid partition function."""
        crf = SemiMarkovCRFHead(num_classes=1, max_duration=10)
        hidden = torch.randn(2, 20, 1)
        lengths = torch.tensor([20, 15])

        result = crf(hidden, lengths, use_triton=False)

        assert torch.isfinite(result["partition"]).all()


class TestMinimalSequences:
    """Test very short sequences (T=1, T=2)."""

    def test_t1_single_segment(self):
        """T=1 sequence has exactly one segment."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(2, 1, 4)
        lengths = torch.tensor([1, 1])

        result = crf.decode_with_traceback(hidden, lengths)

        for b in range(2):
            assert len(result.segments[b]) == 1
            assert result.segments[b][0].start == 0
            assert result.segments[b][0].end == 0
            assert result.segments[b][0].duration == 1

    def test_t1_partition_includes_duration_bias(self):
        """T=1 partition includes duration_bias[1], not duration_bias[0]."""
        # This is a regression test for the duration indexing bug
        crf = SemiMarkovCRFHead(num_classes=2, max_duration=5)

        # Set known values
        crf.transition.data.fill_(0.0)
        crf.duration_dist.duration_bias.data.fill_(0.0)
        crf.duration_dist.duration_bias.data[1, :] = 1.0  # duration=1 gets +1

        hidden = torch.zeros(1, 1, 2)
        hidden[0, 0, 0] = 2.0  # Class 0 gets score 2
        hidden[0, 0, 1] = 1.0  # Class 1 gets score 1
        lengths = torch.tensor([1])

        result = crf(hidden, lengths, use_triton=False)

        # Forward recurrence at t=1:
        # alpha[c] = logsumexp_{c_src} [0 + content[c] + duration_bias[1,c] + transition[c_src,c]]
        # With transition=0 and C=2: alpha[c] = content[c] + dur_bias[1,c] + log(2)
        # alpha[0] = 2 + 1 + log(2) ≈ 3.69
        # alpha[1] = 1 + 1 + log(2) ≈ 2.69
        # partition = logsumexp([3.69, 2.69]) ≈ 4.00
        C = 2
        log_C = torch.tensor(C).float().log()
        expected = torch.logsumexp(torch.tensor([2.0 + 1.0 + log_C, 1.0 + 1.0 + log_C]), dim=0)
        assert torch.allclose(result["partition"], expected.unsqueeze(0), atol=1e-3)

    def test_t2_all_valid_segmentations(self):
        """T=2 sequence works with various segmentation possibilities."""
        crf = SemiMarkovCRFHead(num_classes=2, max_duration=3)
        hidden = torch.randn(1, 2, 2)
        lengths = torch.tensor([2])

        # Forward and decode should both work
        result = crf(hidden, lengths, use_triton=False)
        decode_result = crf.decode_with_traceback(hidden, lengths)

        assert torch.isfinite(result["partition"])
        assert len(decode_result.segments[0]) >= 1  # At least one segment


class TestViterbiTracebackConsistency:
    """Regression tests for Viterbi traceback score consistency."""

    def test_traceback_score_matches_viterbi_score(self):
        """Viterbi traceback segment scores sum to reported Viterbi score."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(2, 20, 4)
        lengths = torch.tensor([20, 15])

        result = crf.decode_with_traceback(hidden, lengths)

        for b in range(2):
            seg_sum = sum(seg.score for seg in result.segments[b])
            assert (
                abs(seg_sum - result.scores[b].item()) < 1e-4
            ), f"Batch {b}: segment sum {seg_sum} != Viterbi score {result.scores[b].item()}"

    def test_traceback_covers_full_sequence(self):
        """Traceback segments cover entire sequence without gaps or overlaps."""
        crf = SemiMarkovCRFHead(num_classes=3, max_duration=10)
        hidden = torch.randn(3, 25, 3)
        lengths = torch.tensor([25, 20, 15])

        result = crf.decode_with_traceback(hidden, lengths)

        for b in range(3):
            seq_len = lengths[b].item()
            segments = result.segments[b]

            # Check first segment starts at 0
            assert segments[0].start == 0, f"Batch {b}: first segment doesn't start at 0"

            # Check last segment ends at seq_len - 1
            assert (
                segments[-1].end == seq_len - 1
            ), f"Batch {b}: last segment ends at {segments[-1].end}, expected {seq_len - 1}"

            # Check segments are contiguous (no gaps or overlaps)
            for i in range(1, len(segments)):
                assert (
                    segments[i].start == segments[i - 1].end + 1
                ), f"Batch {b}: gap/overlap between segments {i - 1} and {i}"


class TestVariableLengths:
    """Test batches with variable sequence lengths."""

    def test_variable_lengths_forward(self):
        """Forward pass handles variable lengths in batch."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(4, 30, 4)
        lengths = torch.tensor([30, 25, 15, 10])

        result = crf(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (4,)
        assert torch.isfinite(result["partition"]).all()

    def test_variable_lengths_decode(self):
        """Viterbi decoding handles variable lengths in batch."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(4, 30, 4)
        lengths = torch.tensor([30, 25, 15, 10])

        result = crf.decode_with_traceback(hidden, lengths)

        for b in range(4):
            seq_len = lengths[b].item()
            # Last segment should end at seq_len - 1
            assert result.segments[b][-1].end == seq_len - 1

    def test_variable_lengths_loss(self):
        """Loss computation handles variable lengths in batch."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        hidden = torch.randn(4, 30, 4)
        lengths = torch.tensor([30, 25, 15, 10])
        labels = torch.randint(0, 4, (4, 30))

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)

        assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonConsistency:
    """Test PyTorch and Triton implementations produce consistent results."""

    def test_forward_consistency(self):
        """PyTorch and Triton forward pass produce same partition."""
        pytest.importorskip("triton")

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8).cuda()
        hidden = torch.randn(2, 100, 4).cuda()
        lengths = torch.tensor([100, 80]).cuda()

        result_pytorch = crf(hidden, lengths, use_triton=False)
        result_triton = crf(hidden, lengths, use_triton=True)

        assert torch.allclose(
            result_pytorch["partition"],
            result_triton["partition"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_decode_consistency(self):
        """PyTorch and Triton decode produce same Viterbi scores."""
        pytest.importorskip("triton")

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8).cuda()
        hidden = torch.randn(2, 100, 4).cuda()
        lengths = torch.tensor([100, 80]).cuda()

        scores_pytorch = crf.decode(hidden, lengths, use_triton=False)
        scores_triton = crf.decode(hidden, lengths, use_triton=True)

        assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4)
