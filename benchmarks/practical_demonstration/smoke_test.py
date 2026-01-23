#!/usr/bin/env python3
"""
Smoke Test for Practical Demonstration Benchmarks

This script validates that the benchmark code runs correctly end-to-end
without requiring the full GENCODE or TIMIT datasets. It uses synthetic
data to exercise the same code paths as the real benchmarks.

Usage:
    python smoke_test.py           # Run all tests
    python smoke_test.py --verbose # Detailed output
    python smoke_test.py gencode   # Run only GENCODE tests
    python smoke_test.py timit     # Run only TIMIT tests

Exit codes:
    0: All tests passed
    1: One or more tests failed
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Test Infrastructure
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str = ""


class TestRunner:
    """Simple test runner with formatted output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []

    def run_test(self, name: str, test_fn: Callable[[], None]) -> bool:
        """Run a single test and record result."""
        start = time.time()
        try:
            test_fn()
            duration = time.time() - start
            result = TestResult(name, True, duration)
            self.results.append(result)
            self._print_result(result)
            return True
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name, False, duration, str(e))
            self.results.append(result)
            self._print_result(result)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _print_result(self, result: TestResult):
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.name} ({result.duration:.2f}s)")
        if not result.passed and result.message:
            print(f"    Error: {result.message}")

    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} tests passed")

        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        return passed == total


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_genomic_data(
    num_samples: int = 20,
    seq_length: int = 200,
    num_classes: int = 5,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """
    Generate synthetic genomic data with realistic segment structure.

    Creates sequences where:
    - Labels change every ~20-50 positions (mimicking exon/intron structure)
    - Each segment has a consistent label
    - DNA is one-hot encoded (A, C, G, T, N)
    """
    sequences = []
    labels = []
    lengths = []

    for _ in range(num_samples):
        # Variable length sequences
        length = np.random.randint(seq_length // 2, seq_length)
        lengths.append(length)

        # Generate segments with varying durations
        label_seq = []
        pos = 0
        current_label = np.random.randint(0, num_classes)

        while pos < length:
            # Duration varies by class (mimics exon/intron length differences)
            if current_label == 4:  # "intron" - longer
                duration = np.random.randint(30, 80)
            else:
                duration = np.random.randint(10, 40)

            duration = min(duration, length - pos)
            label_seq.extend([current_label] * duration)
            pos += duration
            current_label = np.random.randint(0, num_classes)

        label_seq = label_seq[:length]

        # Generate random DNA sequence (one-hot)
        dna_indices = np.random.randint(0, 5, size=length)
        dna_onehot = np.eye(5)[dna_indices]

        sequences.append(torch.tensor(dna_onehot, dtype=torch.float32))
        labels.append(torch.tensor(label_seq, dtype=torch.long))

    return sequences, labels, lengths


def generate_synthetic_acoustic_data(
    num_samples: int = 20,
    seq_length: int = 150,
    num_classes: int = 39,
    feature_dim: int = 39,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """
    Generate synthetic acoustic data with realistic phoneme structure.

    Creates sequences where:
    - Labels change every ~5-20 frames (mimicking phoneme durations)
    - Features are random Gaussians (mimics MFCC distribution)
    """
    features = []
    labels = []
    lengths = []

    for _ in range(num_samples):
        length = np.random.randint(seq_length // 2, seq_length)
        lengths.append(length)

        # Generate phoneme segments
        label_seq = []
        pos = 0
        current_label = np.random.randint(0, num_classes)

        while pos < length:
            # Phoneme duration: 5-20 frames
            duration = np.random.randint(5, 20)
            duration = min(duration, length - pos)
            label_seq.extend([current_label] * duration)
            pos += duration
            current_label = np.random.randint(0, num_classes)

        label_seq = label_seq[:length]

        # Generate random acoustic features (MFCC-like)
        feat = np.random.randn(length, feature_dim).astype(np.float32)

        features.append(torch.tensor(feat))
        labels.append(torch.tensor(label_seq, dtype=torch.long))

    return features, labels, lengths


class SyntheticDataset(Dataset):
    """Simple dataset wrapper for synthetic data."""

    def __init__(self, sequences: list[Tensor], labels: list[Tensor], lengths: list[int]):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "labels": self.labels[idx],
            "length": self.lengths[idx],
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(b["length"] for b in batch)

    sequences = []
    labels = []
    lengths = []

    for b in batch:
        seq = b["sequence"]
        lab = b["labels"]
        length = b["length"]

        # Pad to max_len
        if seq.size(0) < max_len:
            pad_len = max_len - seq.size(0)
            seq = torch.cat([seq, torch.zeros(pad_len, seq.size(1))])
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])

        sequences.append(seq)
        labels.append(lab)
        lengths.append(torch.tensor(length))

    return {
        "sequence": torch.stack(sequences),
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths),
    }


# =============================================================================
# Import Tests
# =============================================================================

def test_library_imports():
    """Test that torch_semimarkov imports correctly."""
    from torch_semimarkov import (
        SemiMarkovCRFHead,
        UncertaintySemiMarkovCRFHead,
        Segment,
        ViterbiResult,
    )

    assert SemiMarkovCRFHead is not None
    assert UncertaintySemiMarkovCRFHead is not None
    assert Segment is not None
    assert ViterbiResult is not None


def test_gencode_imports():
    """Test that GENCODE benchmark module imports correctly."""
    # These are the key components used in the benchmark
    from gencode.gencode_exon_intron import (
        NUM_CLASSES,
        LABEL_NAMES,
        BiLSTMEncoder,
        ExonIntronModel,
        compute_position_metrics,
        compute_boundary_metrics,
    )

    assert NUM_CLASSES == 5
    assert len(LABEL_NAMES) == 5


def test_timit_imports():
    """Test that TIMIT benchmark module imports correctly."""
    from timit.timit_phoneme import (
        NUM_PHONES,
        PHONES_39,
        PHONE_61_TO_39,
        BiLSTMEncoder,
        TIMITModel,
    )

    # Verify 39-phone set is correct
    assert NUM_PHONES == 39, f"Expected 39 phones, got {NUM_PHONES}"
    assert len(PHONES_39) == 39, f"PHONES_39 has {len(PHONES_39)} phones"

    # Verify all 61 phones have mappings
    from timit.timit_phoneme import TIMIT_61_PHONES
    for phone in TIMIT_61_PHONES:
        assert phone in PHONE_61_TO_39, f"Phone '{phone}' missing from mapping"

    # Verify mappings point to valid phones
    for target in PHONE_61_TO_39.values():
        assert target in PHONES_39, f"Target phone '{target}' not in PHONES_39"


def test_calibration_imports():
    """Test that calibration module imports correctly."""
    from calibration import (
        compute_ece,
        compute_mce,
        compute_brier_score,
        derive_boundary_probs_from_positions,
    )

    assert callable(compute_ece)
    assert callable(compute_brier_score)


# =============================================================================
# Model Tests
# =============================================================================

def test_semicrf_head_basic():
    """Test SemiMarkovCRFHead basic functionality."""
    from torch_semimarkov import SemiMarkovCRFHead

    batch_size = 4
    seq_len = 50
    hidden_dim = 64
    num_classes = 5
    max_duration = 20

    crf = SemiMarkovCRFHead(
        num_classes=num_classes,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
    )

    # Test forward pass
    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    lengths = torch.tensor([seq_len, 45, 40, 35])

    output = crf(hidden, lengths)
    assert "partition" in output
    assert output["partition"].shape == (batch_size,)

    # Test decode
    result = crf.decode_with_traceback(hidden, lengths)
    assert len(result.segments) == batch_size
    assert result.scores.shape == (batch_size,)


def test_semicrf_loss_computation():
    """Test SemiMarkovCRFHead loss computation."""
    from torch_semimarkov import SemiMarkovCRFHead

    batch_size = 4
    seq_len = 50
    hidden_dim = 64
    num_classes = 5
    max_duration = 20

    crf = SemiMarkovCRFHead(
        num_classes=num_classes,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
    )

    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    lengths = torch.tensor([seq_len, 45, 40, 35])
    labels = torch.randint(0, num_classes, (batch_size, seq_len))

    loss = crf.compute_loss(hidden, lengths, labels)

    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0  # NLL should be positive
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_linear_crf_baseline():
    """Test that K=1 (linear CRF baseline) works."""
    from torch_semimarkov import SemiMarkovCRFHead

    batch_size = 4
    seq_len = 50
    hidden_dim = 64
    num_classes = 5

    # K=1 is the linear CRF baseline
    crf = SemiMarkovCRFHead(
        num_classes=num_classes,
        max_duration=1,  # Linear CRF
        hidden_dim=hidden_dim,
    )

    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    lengths = torch.tensor([seq_len, 45, 40, 35])
    labels = torch.randint(0, num_classes, (batch_size, seq_len))

    # Forward
    output = crf(hidden, lengths)
    assert "partition" in output

    # Loss
    loss = crf.compute_loss(hidden, lengths, labels)
    assert not torch.isnan(loss)

    # Decode
    result = crf.decode_with_traceback(hidden, lengths)
    assert len(result.segments) == batch_size


# =============================================================================
# Training Pipeline Tests
# =============================================================================

def test_gencode_training_pipeline():
    """Test GENCODE-style training pipeline end-to-end."""
    from gencode.gencode_exon_intron import BiLSTMEncoder, ExonIntronModel

    # Generate synthetic data
    sequences, labels, lengths = generate_synthetic_genomic_data(
        num_samples=16, seq_length=100, num_classes=5
    )

    dataset = SyntheticDataset(sequences, labels, lengths)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Create model (Semi-CRF with K=20)
    encoder = BiLSTMEncoder(input_dim=5, hidden_dim=64, num_layers=1)
    model = ExonIntronModel(
        encoder=encoder,
        num_classes=5,
        max_duration=20,
        hidden_dim=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 2 epochs
    model.train()
    for epoch in range(2):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(
                batch["sequence"],
                batch["lengths"],
                batch["labels"],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Loss should decrease
        assert not np.isnan(epoch_loss)

    # Test decoding
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            result = model.decode(batch["sequence"], batch["lengths"])
            assert len(result.segments) == len(batch["lengths"])
            break


def test_timit_training_pipeline():
    """Test TIMIT-style training pipeline end-to-end."""
    from timit.timit_phoneme import BiLSTMEncoder, TIMITModel, NUM_PHONES

    # Generate synthetic acoustic data
    features, labels, lengths = generate_synthetic_acoustic_data(
        num_samples=16, seq_length=100, num_classes=NUM_PHONES, feature_dim=39
    )

    dataset = SyntheticDataset(features, labels, lengths)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Create model (Semi-CRF with K=15)
    encoder = BiLSTMEncoder(input_dim=39, hidden_dim=64, num_layers=1)
    model = TIMITModel(
        encoder=encoder,
        num_classes=NUM_PHONES,
        max_duration=15,
        hidden_dim=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 2 epochs
    model.train()
    for epoch in range(2):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(
                batch["sequence"],
                batch["lengths"],
                batch["labels"],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        assert not np.isnan(epoch_loss)

    # Test decoding
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            result = model.decode(batch["sequence"], batch["lengths"])
            assert len(result.segments) == len(batch["lengths"])
            break


def test_linear_vs_semicrf_comparison():
    """Test that both linear CRF and semi-CRF can be trained on same data."""
    from gencode.gencode_exon_intron import BiLSTMEncoder, ExonIntronModel

    # Generate data
    sequences, labels, lengths = generate_synthetic_genomic_data(
        num_samples=8, seq_length=80, num_classes=5
    )

    dataset = SyntheticDataset(sequences, labels, lengths)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Linear CRF (K=1)
    encoder1 = BiLSTMEncoder(input_dim=5, hidden_dim=64, num_layers=1)
    linear_model = ExonIntronModel(
        encoder=encoder1,
        num_classes=5,
        max_duration=1,  # K=1
        hidden_dim=64,
    )

    # Semi-CRF (K=20)
    encoder2 = BiLSTMEncoder(input_dim=5, hidden_dim=64, num_layers=1)
    semi_model = ExonIntronModel(
        encoder=encoder2,
        num_classes=5,
        max_duration=20,  # K=20
        hidden_dim=64,
    )

    # Both should compute loss without error
    for batch in dataloader:
        loss1 = linear_model.compute_loss(
            batch["sequence"], batch["lengths"], batch["labels"]
        )
        loss2 = semi_model.compute_loss(
            batch["sequence"], batch["lengths"], batch["labels"]
        )

        assert not torch.isnan(loss1)
        assert not torch.isnan(loss2)
        break


# =============================================================================
# Evaluation Metrics Tests
# =============================================================================

def test_position_metrics():
    """Test position-level F1 computation."""
    from gencode.gencode_exon_intron import compute_position_metrics

    # Create synthetic predictions and targets
    predictions = [
        np.array([0, 0, 1, 1, 2, 2, 2]),
        np.array([0, 1, 1, 2, 2]),
    ]
    targets = [
        np.array([0, 0, 1, 1, 2, 2, 3]),  # One wrong
        np.array([0, 1, 1, 2, 2]),        # All correct
    ]

    metrics = compute_position_metrics(predictions, targets, num_classes=5)

    assert "macro" in metrics
    assert 0 <= metrics["macro"] <= 1
    assert all(0 <= v <= 1 for v in metrics.values())


def test_boundary_metrics():
    """Test boundary detection metrics."""
    from gencode.gencode_exon_intron import compute_boundary_metrics

    predictions = [
        np.array([0, 0, 0, 1, 1, 1, 2, 2]),
        np.array([0, 0, 1, 1, 2]),
    ]
    targets = [
        np.array([0, 0, 0, 1, 1, 1, 2, 2]),  # Perfect match
        np.array([0, 0, 0, 1, 2]),            # One boundary off
    ]

    metrics = compute_boundary_metrics(predictions, targets)

    assert "boundary_precision" in metrics
    assert "boundary_recall" in metrics
    assert "boundary_f1" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


def test_phone_error_rate():
    """Test PER computation for TIMIT."""
    from timit.timit_phoneme import compute_phone_error_rate

    # Perfect prediction
    predictions = [[0, 0, 1, 1, 2, 2]]
    references = [[0, 0, 1, 1, 2, 2]]

    per = compute_phone_error_rate(predictions, references)
    assert per == 0.0  # Perfect match should be 0

    # One substitution
    predictions = [[0, 0, 1, 1, 3, 3]]
    references = [[0, 0, 1, 1, 2, 2]]

    per = compute_phone_error_rate(predictions, references)
    assert per > 0  # Should have some error


def test_calibration_metrics():
    """Test calibration metric computation."""
    from calibration import compute_ece, compute_brier_score

    # Generate synthetic probabilities and labels
    n = 100
    probs = np.random.rand(n)
    labels = (np.random.rand(n) < probs).astype(int)

    # compute_ece returns (ece, bin_edges, bin_accuracies, bin_confidences, bin_counts)
    ece, bin_edges, bin_accuracies, bin_confidences, bin_counts = compute_ece(probs, labels)
    assert 0 <= ece <= 1
    assert len(bin_edges) > 0

    brier = compute_brier_score(probs, labels)
    assert 0 <= brier <= 1


def test_boundary_derivation():
    """Test deriving boundary probs from position marginals."""
    from calibration import derive_boundary_probs_from_positions

    T = 50
    C = 5

    # Create synthetic position marginals (sum to 1 along class dim)
    position_marginals = np.random.rand(T, C)
    position_marginals = position_marginals / position_marginals.sum(axis=1, keepdims=True)

    boundary_probs = derive_boundary_probs_from_positions(position_marginals)

    assert boundary_probs.shape == (T,)
    assert all(0 <= p <= 1 for p in boundary_probs)


# =============================================================================
# Segment Convention Tests
# =============================================================================

def test_segment_end_convention():
    """Verify segment end convention (inclusive end)."""
    from torch_semimarkov import SemiMarkovCRFHead

    batch_size = 2
    seq_len = 30
    hidden_dim = 32
    num_classes = 3

    crf = SemiMarkovCRFHead(
        num_classes=num_classes,
        max_duration=10,
        hidden_dim=hidden_dim,
    )

    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    lengths = torch.tensor([seq_len, 25])

    result = crf.decode_with_traceback(hidden, lengths)

    for i, segs in enumerate(result.segments):
        seq_length = lengths[i].item()

        # Verify segments are contiguous and cover full sequence
        if len(segs) > 0:
            # First segment should start at 0
            assert segs[0].start == 0, f"First segment starts at {segs[0].start}, expected 0"

            # Segments should be contiguous
            for j in range(len(segs) - 1):
                # With inclusive end: seg[j].end + 1 == seg[j+1].start
                assert segs[j].end + 1 == segs[j + 1].start, \
                    f"Gap between segments: {segs[j]} and {segs[j+1]}"

            # Last segment should end at seq_length - 1 (inclusive)
            assert segs[-1].end == seq_length - 1, \
                f"Last segment ends at {segs[-1].end}, expected {seq_length - 1}"


# =============================================================================
# Main Test Runner
# =============================================================================

def run_gencode_tests(runner: TestRunner):
    """Run GENCODE-specific tests."""
    print("\nGENCODE Benchmark Tests:")
    print("-" * 40)
    runner.run_test("Import GENCODE module", test_gencode_imports)
    runner.run_test("GENCODE training pipeline", test_gencode_training_pipeline)
    runner.run_test("Position metrics", test_position_metrics)
    runner.run_test("Boundary metrics", test_boundary_metrics)


def run_timit_tests(runner: TestRunner):
    """Run TIMIT-specific tests."""
    print("\nTIMIT Benchmark Tests:")
    print("-" * 40)
    runner.run_test("Import TIMIT module", test_timit_imports)
    runner.run_test("TIMIT training pipeline", test_timit_training_pipeline)
    runner.run_test("Phone error rate", test_phone_error_rate)


def run_core_tests(runner: TestRunner):
    """Run core library tests."""
    print("\nCore Library Tests:")
    print("-" * 40)
    runner.run_test("Library imports", test_library_imports)
    runner.run_test("SemiMarkovCRFHead basic", test_semicrf_head_basic)
    runner.run_test("SemiMarkovCRFHead loss", test_semicrf_loss_computation)
    runner.run_test("Linear CRF baseline (K=1)", test_linear_crf_baseline)
    runner.run_test("Segment end convention", test_segment_end_convention)


def run_comparison_tests(runner: TestRunner):
    """Run comparison tests."""
    print("\nComparison Tests:")
    print("-" * 40)
    runner.run_test("Linear vs Semi-CRF comparison", test_linear_vs_semicrf_comparison)


def run_calibration_tests(runner: TestRunner):
    """Run calibration module tests."""
    print("\nCalibration Module Tests:")
    print("-" * 40)
    runner.run_test("Import calibration module", test_calibration_imports)
    runner.run_test("Calibration metrics (ECE, Brier)", test_calibration_metrics)
    runner.run_test("Boundary derivation", test_boundary_derivation)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for practical demonstrations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("tests", nargs="*", default=["all"],
                       help="Which tests to run: gencode, timit, calibration, or all (default)")

    args = parser.parse_args()

    # Validate test choices
    valid_choices = {"gencode", "timit", "calibration", "all"}
    for test in args.tests:
        if test not in valid_choices:
            parser.error(f"invalid choice: {test} (choose from {', '.join(valid_choices)})")

    print("=" * 60)
    print("Practical Demonstration Smoke Tests")
    print("=" * 60)

    runner = TestRunner(verbose=args.verbose)

    # Always run core tests
    run_core_tests(runner)

    # Run selected tests
    tests = set(args.tests)
    if "all" in tests:
        tests = {"gencode", "timit", "calibration"}

    if "gencode" in tests:
        run_gencode_tests(runner)

    if "timit" in tests:
        run_timit_tests(runner)

    if "calibration" in tests:
        run_calibration_tests(runner)

    # Run comparison if both benchmarks tested
    if "gencode" in tests or "timit" in tests:
        run_comparison_tests(runner)

    # Print summary
    success = runner.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
