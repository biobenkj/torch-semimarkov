"""Boundary precision tests for clinical applications.

Critical tests for ensuring no off-by-1 errors in boundary detection.
In clinical applications (ECG, EEG, genomics), precise boundary detection
is essential for accurate diagnosis and treatment.

Tests verify:
- Boundary at position 0 (segment start)
- Boundary at position T-1 (segment end)
- Consecutive boundaries (duration=1 segments)
- Boundary indices match label changes exactly
- Clinical regression tests (ECG beats, sleep stages, gene boundaries)
"""

import pytest
import torch

from torch_semimarkov import SemiMarkovCRFHead


class TestBoundaryExtraction:
    """Test boundary extraction from labels matches expected positions."""

    def test_single_segment_no_boundaries(self):
        """Single segment (all same label) should have no internal boundaries."""
        T = 20
        labels = torch.zeros(1, T, dtype=torch.long)  # All label 0

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        assert len(changes) == 0, "Single segment should have no boundaries"

    def test_two_segments_one_boundary(self):
        """Two segments should have exactly one boundary."""
        T = 20
        labels = torch.zeros(1, T, dtype=torch.long)
        labels[0, 10:] = 1  # Boundary at position 10

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        assert len(changes) == 1
        assert changes[0].item() == 9, "Boundary should be at position 9 (label[9] != label[10])"

    def test_boundary_at_position_0(self):
        """First position is always start of first segment (implicit boundary)."""
        labels = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Segment extraction
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])

        # First segment starts at 0
        assert seg_starts[0].item() == 0

    def test_boundary_at_position_T_minus_1(self):
        """Test boundary at the very last position (T-1)."""
        T = 10
        # Boundary at position 9: label changes from 0 to 1 at last position
        labels = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        assert changes[-1].item() == 8, "Boundary should be at position 8"

        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        # Last segment: [9, 9] with duration 1
        assert seg_starts[-1].item() == 9
        assert seg_ends[-1].item() == 9
        assert seg_ends[-1] - seg_starts[-1] + 1 == 1

    def test_consecutive_boundaries_duration_1(self):
        """Test consecutive boundaries creating duration=1 segments."""
        T = 5
        # Each position has a different label: [0, 1, 2, 3, 0]
        labels = torch.tensor([[0, 1, 2, 3, 0]])

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Boundaries at positions 0, 1, 2, 3 (4 boundaries for 5 segments)
        assert len(changes) == 4
        expected_boundaries = torch.tensor([0, 1, 2, 3])
        torch.testing.assert_close(changes, expected_boundaries)

        # All segments have duration 1
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        for i in range(len(seg_starts)):
            duration = seg_ends[i] - seg_starts[i] + 1
            assert duration == 1, f"Segment {i} should have duration 1, got {duration}"

    def test_boundary_indices_match_label_changes_exactly(self):
        """Verify boundary positions exactly match where labels change."""
        T = 20
        # Known boundaries at positions 5, 10, 15
        labels = torch.zeros(1, T, dtype=torch.long)
        labels[0, 5:10] = 1
        labels[0, 10:15] = 2
        labels[0, 15:] = 3

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Boundaries at 4, 9, 14 (positions where label[t] != label[t+1])
        expected_boundaries = torch.tensor([4, 9, 14])
        torch.testing.assert_close(changes, expected_boundaries)

        # Verify segment spans
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        assert seg_starts.tolist() == [0, 5, 10, 15]
        assert seg_ends.tolist() == [4, 9, 14, 19]


class TestScoreGoldBoundaryPrecision:
    """Test _score_gold boundary handling in SemiMarkovCRFHead."""

    @pytest.fixture
    def crf_head(self):
        return SemiMarkovCRFHead(num_classes=4, max_duration=16)

    def test_single_segment_content_score(self, crf_head):
        """Test content score for single segment spans all positions."""
        T = 10
        # Create scores where label 0 has value 1.0 at all positions
        scores = torch.zeros(1, T, 4)
        scores[:, :, 0] = 1.0

        # Cumulative scores
        cum_scores = torch.zeros(1, T + 1, 4, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Labels: all label 0
        labels = torch.zeros(1, T, dtype=torch.long)
        lengths = torch.tensor([T])

        gold_score = crf_head._score_gold(cum_scores, labels, lengths)

        # Content score should be sum of all 10 positions = 10.0
        expected_content = 10.0
        # Plus duration bias for duration=10 (clamped to max_duration-1 if needed)
        # Convention: duration_bias[k] stores bias for duration k
        dur_idx = min(T, crf_head.max_duration - 1)
        expected_duration = crf_head.duration_bias[dur_idx, 0].item()
        expected = expected_content + expected_duration

        assert abs(gold_score.item() - expected) < 1e-5

    def test_two_segment_boundary_position(self, crf_head):
        """Test boundary position is correctly handled in scoring."""
        T = 10
        # Scores: label 0 for first 5, label 1 for last 5
        scores = torch.zeros(1, T, 4)
        scores[:, :5, 0] = 1.0
        scores[:, 5:, 1] = 1.0

        cum_scores = torch.zeros(1, T + 1, 4, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Labels matching the scores
        labels = torch.zeros(1, T, dtype=torch.long)
        labels[:, 5:] = 1
        lengths = torch.tensor([T])

        gold_score = crf_head._score_gold(cum_scores, labels, lengths)

        # Segment 1: [0, 4], label=0, duration=5, content=5.0
        # Segment 2: [5, 9], label=1, duration=5, content=5.0
        # Convention: duration_bias[k] stores bias for duration k
        expected_content = 5.0 + 5.0
        expected_duration = (
            crf_head.duration_bias[5, 0].item() + crf_head.duration_bias[5, 1].item()
        )
        expected_transition = crf_head.transition[0, 1].item()
        expected = expected_content + expected_duration + expected_transition

        assert abs(gold_score.item() - expected) < 1e-5

    def test_boundary_at_last_position_scoring(self, crf_head):
        """Test scoring when boundary is at T-1."""
        T = 10
        # All label 0 except last position is label 1
        labels = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        scores = torch.ones(1, T, 4)

        cum_scores = torch.zeros(1, T + 1, 4, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)
        lengths = torch.tensor([T])

        gold_score = crf_head._score_gold(cum_scores, labels, lengths)

        # Segment 1: [0, 8], label=0, duration=9
        # Segment 2: [9, 9], label=1, duration=1
        # Convention: duration_bias[k] stores bias for duration k
        expected_content = 9.0 + 1.0  # All ones
        dur_idx_1 = min(9, crf_head.max_duration - 1)
        expected_duration = (
            crf_head.duration_bias[dur_idx_1, 0].item()
            + crf_head.duration_bias[1, 1].item()  # duration=1 -> index 1
        )
        expected_transition = crf_head.transition[0, 1].item()
        expected = expected_content + expected_duration + expected_transition

        assert abs(gold_score.item() - expected) < 1e-5


class TestClinicalBoundaryRegression:
    """Clinical regression tests for boundary detection."""

    def test_ecg_beat_boundaries(self, clinical_ecg_config, create_synthetic_clinical_data):
        """Test ECG beat detection boundaries are precise."""
        T = 50

        # Create synthetic ECG with known beat boundaries
        # Format: [N-beat][gap][V-beat][gap][N-beat]
        labels = torch.zeros(1, T, dtype=torch.long)

        # Beat 1 (N-type): positions 0-9
        labels[0, 0:10] = 0  # Normal beat
        # Gap: positions 10-14
        labels[0, 10:15] = 2  # Background/other
        # Beat 2 (V-type): positions 15-24
        labels[0, 15:25] = 1  # Ventricular
        # Gap: positions 25-29
        labels[0, 25:30] = 2  # Background
        # Beat 3 (N-type): positions 30-39
        labels[0, 30:40] = 0  # Normal
        # Trailing: positions 40-49
        labels[0, 40:] = 2

        # Extract boundaries
        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Boundaries should be exactly at: 9, 14, 24, 29, 39
        expected_boundaries = torch.tensor([9, 14, 24, 29, 39])
        torch.testing.assert_close(changes, expected_boundaries)

        # Verify first N-beat boundary
        # First N-beat ends at position 9 (not 8 or 10)
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        assert seg_starts[0] == 0 and seg_ends[0] == 9, "First beat should span [0, 9]"
        assert seg_ends[0] - seg_starts[0] + 1 == 10, "First beat duration should be 10"

    def test_eeg_sleep_stage_boundaries(self, clinical_eeg_config):
        """Test EEG sleep stage transition boundaries are precise."""
        # Simplified sleep staging sequence
        # Each "epoch" is represented by multiple time points
        T = 100  # 100 time points representing ~10 epochs

        labels = torch.zeros(1, T, dtype=torch.long)
        # Wake (0): 0-19
        labels[0, 0:20] = 0
        # N1 (1): 20-29
        labels[0, 20:30] = 1
        # N2 (2): 30-59
        labels[0, 30:60] = 2
        # N3 (3): 60-79
        labels[0, 60:80] = 3
        # REM (4): 80-99
        labels[0, 80:] = 4

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Boundaries at 19, 29, 59, 79
        expected_boundaries = torch.tensor([19, 29, 59, 79])
        torch.testing.assert_close(changes, expected_boundaries)

        # Verify transition from Wake to N1 is exactly at position 19->20
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        assert seg_starts[1] == 20, "N1 should start exactly at position 20"

    def test_gene_boundary_precision(self, clinical_genomics_config):
        """Test gene structure boundary detection precision."""
        T = 100

        # Simplified gene structure: 5'UTR - exon - intron - exon - 3'UTR
        labels = torch.zeros(1, T, dtype=torch.long)
        # 5'UTR (0): 0-9
        labels[0, 0:10] = 0
        # Exon 1 (1): 10-29
        labels[0, 10:30] = 1
        # Intron (2): 30-59
        labels[0, 30:60] = 2
        # Exon 2 (1): 60-79
        labels[0, 60:80] = 1
        # 3'UTR (3): 80-99
        labels[0, 80:] = 3

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # Boundaries at 9, 29, 59, 79
        expected_boundaries = torch.tensor([9, 29, 59, 79])
        torch.testing.assert_close(changes, expected_boundaries)

        # Verify exon-intron junction is precise (critical for splicing analysis)
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        # Exon 1 ends at 29, intron starts at 30
        assert seg_ends[1] == 29, "Exon 1 should end at position 29"
        assert seg_starts[2] == 30, "Intron should start at position 30"


class TestOffByOneRegression:
    """Specific regression tests for off-by-1 errors."""

    def test_segment_duration_calculation(self):
        """Verify duration = end - start + 1 (not end - start)."""
        T = 10
        labels = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]])

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        expected_durations = [3, 4, 3]  # Segments: [0,2], [3,6], [7,9]

        for i, expected_dur in enumerate(expected_durations):
            actual_dur = (seg_ends[i] - seg_starts[i] + 1).item()
            assert (
                actual_dur == expected_dur
            ), f"Segment {i}: expected duration {expected_dur}, got {actual_dur}"

    def test_cumsum_indexing_precision(self):
        """Verify cumsum indexing: content = cum[end+1] - cum[start]."""
        T = 5
        scores = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]])

        cum_scores = torch.zeros(1, T + 1, 2, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Segment [0, 4] (all 5 positions) for label 0
        # Content should be 1+2+3+4+5 = 15
        start, end = 0, 4
        content = cum_scores[0, end + 1, 0] - cum_scores[0, start, 0]
        assert content.item() == 15.0

        # Segment [2, 3] (positions 2 and 3) for label 0
        # Content should be 3+4 = 7
        start, end = 2, 3
        content = cum_scores[0, end + 1, 0] - cum_scores[0, start, 0]
        assert content.item() == 7.0

    def test_zero_length_edge_case(self):
        """Verify handling of edge case with minimal sequence."""
        T = 1
        labels = torch.tensor([[0]])

        seq_labels = labels[0]
        changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

        # No boundaries in single-element sequence
        assert len(changes) == 0

        # Single segment spanning [0, 0]
        seg_starts = torch.cat([torch.tensor([0]), changes + 1])
        seg_ends = torch.cat([changes, torch.tensor([T - 1])])

        assert seg_starts[0] == 0
        assert seg_ends[0] == 0
        assert (seg_ends[0] - seg_starts[0] + 1).item() == 1


class TestBoundaryWithCRFForward:
    """Test boundary handling in full CRF forward pass."""

    @pytest.fixture
    def model_config(self):
        return {
            "num_classes": 4,
            "max_duration": 16,
            "hidden_dim": 32,
        }

    def test_loss_consistent_with_boundary_positions(self, model_config):
        """Verify loss computation respects exact boundary positions."""
        cfg = model_config
        crf = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["hidden_dim"],
        )

        T = 20
        batch = 1

        # Create hidden states and labels with known boundary
        hidden = torch.randn(batch, T, cfg["hidden_dim"])
        labels = torch.zeros(batch, T, dtype=torch.long)
        labels[:, 10:] = 1  # Boundary at position 10
        lengths = torch.tensor([T])

        # Compute loss
        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)

        # Loss should be finite
        assert torch.isfinite(loss)

        # Different boundary position should give different loss
        labels_shifted = labels.clone()
        labels_shifted[:, 10] = 0
        labels_shifted[:, 11:] = 1  # Boundary shifted by 1

        loss_shifted = crf.compute_loss(hidden, lengths, labels_shifted, use_triton=False)

        # Losses should be different (different segmentations)
        assert not torch.allclose(loss, loss_shifted)

    def test_gradient_flow_through_boundary_regions(self, model_config):
        """Verify gradients flow correctly around boundary regions."""
        cfg = model_config
        crf = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["hidden_dim"],
        )

        T = 20
        batch = 1

        hidden = torch.randn(batch, T, cfg["hidden_dim"], requires_grad=True)
        labels = torch.zeros(batch, T, dtype=torch.long)
        labels[:, 10:] = 1  # Boundary at position 10
        lengths = torch.tensor([T])

        loss = crf.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        # Gradients should exist at all positions, including boundary
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()

        # Gradient at boundary region should be non-zero (boundary is informative)
        boundary_grad = hidden.grad[:, 9:11, :].abs().mean()
        assert boundary_grad > 0
