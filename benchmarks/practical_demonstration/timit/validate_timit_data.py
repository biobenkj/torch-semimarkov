#!/usr/bin/env python3
"""
TIMIT Data Validation Script

Checks preprocessed MFCC features for:
- NaN/Inf values
- Proper normalization (mean near 0, std near 1)
- Label validity
- Sequence length distribution

Usage:
    python validate_timit_data.py --data-dir data/timit_benchmark/
    python validate_timit_data.py --data-dir data/timit_benchmark/ --split train
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def validate_timit_data(data_dir: Path, split: str = "train"):
    """
    Validate preprocessed TIMIT data.

    Args:
        data_dir: Path to preprocessed data directory
        split: "train" or "test"

    Returns:
        dict with validation results
    """
    data_path = data_dir / f"{split}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Validating {data_path}...")

    # Accumulators
    all_features = []
    all_lengths = []
    label_counts = defaultdict(int)
    issues = []

    nan_count = 0
    inf_count = 0
    utterance_count = 0

    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            utt = json.loads(line)
            utterance_count += 1

            features = np.array(utt["features"], dtype=np.float32)
            labels = np.array(utt["labels"], dtype=np.int64)

            # Check for NaN
            if np.isnan(features).any():
                nan_positions = np.where(np.isnan(features))
                issues.append(
                    f"Line {line_num} ({utt['utterance_id']}): NaN at positions {nan_positions}"
                )
                nan_count += 1

            # Check for Inf
            if np.isinf(features).any():
                inf_positions = np.where(np.isinf(features))
                issues.append(
                    f"Line {line_num} ({utt['utterance_id']}): Inf at positions {inf_positions}"
                )
                inf_count += 1

            # Check label validity (should be 0-38 for 39 phones)
            if labels.min() < 0 or labels.max() > 38:
                issues.append(
                    f"Line {line_num} ({utt['utterance_id']}): Invalid labels [{labels.min()}, {labels.max()}]"
                )

            # Accumulate for stats
            all_features.append(features)
            all_lengths.append(len(features))

            for lab in labels:
                label_counts[int(lab)] += 1

    # Compute global statistics
    all_features_concat = np.concatenate(all_features, axis=0)

    # Per-dimension stats
    feature_means = all_features_concat.mean(axis=0)
    feature_stds = all_features_concat.std(axis=0)

    # Global stats
    global_mean = all_features_concat.mean()
    global_std = all_features_concat.std()
    global_min = all_features_concat.min()
    global_max = all_features_concat.max()

    # Print results
    print(f"\n{'='*60}")
    print(f"TIMIT {split.upper()} Data Validation Report")
    print(f"{'='*60}")

    print("\n📊 Dataset Statistics:")
    print(f"  Utterances: {utterance_count}")
    print(f"  Total frames: {len(all_features_concat):,}")
    print(f"  Feature dimension: {all_features_concat.shape[1]}")
    print(
        f"  Sequence lengths: min={min(all_lengths)}, max={max(all_lengths)}, mean={np.mean(all_lengths):.1f}"
    )

    print("\n📈 Feature Statistics (Global):")
    print(f"  Mean: {global_mean:.4f} (ideal: ~0)")
    print(f"  Std:  {global_std:.4f} (ideal: ~1)")
    print(f"  Min:  {global_min:.4f}")
    print(f"  Max:  {global_max:.4f}")

    print("\n📈 Feature Statistics (Per-Dimension):")
    print(f"  Mean range: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
    print(f"  Std range:  [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")

    # Normalization assessment
    print("\n🔍 Normalization Check:")
    if abs(global_mean) < 0.1 and 0.5 < global_std < 2.0:
        print("  ✅ Features appear normalized (mean≈0, std≈1)")
    else:
        print("  ⚠️  Features may need normalization!")
        print("     Consider: features = (features - mean) / std")

    # NaN/Inf check
    print("\n🚨 NaN/Inf Check:")
    if nan_count == 0 and inf_count == 0:
        print("  ✅ No NaN or Inf values found")
    else:
        print(f"  ❌ Found {nan_count} utterances with NaN")
        print(f"  ❌ Found {inf_count} utterances with Inf")

    # Label distribution
    print("\n🏷️  Label Distribution:")
    print(f"  Unique labels: {len(label_counts)}")
    print(f"  Most common: {sorted(label_counts.items(), key=lambda x: -x[1])[:5]}")
    print(f"  Least common: {sorted(label_counts.items(), key=lambda x: x[1])[:5]}")

    # Issues summary
    if issues:
        print(f"\n❌ Issues Found ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\n✅ No data quality issues found!")

    return {
        "utterance_count": utterance_count,
        "total_frames": len(all_features_concat),
        "feature_dim": all_features_concat.shape[1],
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "issues": issues,
        "needs_normalization": not (abs(global_mean) < 0.1 and 0.5 < global_std < 2.0),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate preprocessed TIMIT data")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to preprocessed data")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    all_results = {}
    for split in splits:
        try:
            results = validate_timit_data(args.data_dir, split)
            all_results[split] = results
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {split}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    has_issues = False
    for split, results in all_results.items():
        if results["nan_count"] > 0 or results["inf_count"] > 0:
            print(f"❌ {split}: Contains NaN/Inf - FIX REQUIRED")
            has_issues = True
        elif results["needs_normalization"]:
            print(f"⚠️  {split}: Needs normalization - RECOMMENDED")
            has_issues = True
        else:
            print(f"✅ {split}: Data quality OK")

    if has_issues:
        print("\n⚠️  Address issues above before training!")
        exit(1)
    else:
        print("\n✅ All data validation checks passed!")
        exit(0)
