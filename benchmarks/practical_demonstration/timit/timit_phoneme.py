#!/usr/bin/env python3
"""
TIMIT Phoneme Segmentation Benchmark

This is the classic benchmark for demonstrating Semi-CRF advantages over linear CRFs.
TIMIT has been used since the original Semi-CRF paper (Sarawagi & Cohen, 2004) and
provides a well-studied setting with published baselines.

Three-Way Model Comparison:
    This benchmark supports comparing three CRF implementations:

    1. **pytorch-crf** (optional): External linear CRF library baseline
    2. **torch-semimarkov K=1**: Linear CRF via Triton streaming kernel
    3. **torch-semimarkov K>1**: Full semi-CRF with duration modeling

    The comparison validates that:
    - K=1 Triton matches pytorch-crf accuracy (correctness)
    - K=1 Triton is faster than pytorch-crf (performance)
    - Semi-CRF improves on linear CRF (duration modeling value)

Why Semi-CRFs help on TIMIT:
    - Phonemes have characteristic durations (vowels longer than stops)
    - Duration is linguistically meaningful and predictable
    - A linear CRF can't encode "this phoneme typically lasts 50-100ms"
    - A semi-CRF learns duration priors per phoneme class

Dataset:
    - 630 speakers (462 train, 168 test)
    - ~6300 utterances total
    - 61 phoneme classes (typically collapsed to 39)
    - Standard train/test split defined by NIST
    - Requires LDC license (widely available in practice)

Features:
    - 13 MFCCs + delta + delta-delta = 39 features
    - 10ms frame shift (100 Hz)
    - Alternative: 80-dim log mel filterbanks

Metrics:
    - Phone Error Rate (PER): Levenshtein distance / reference phones
    - Boundary F1: Exact match and within-tolerance
    - Segment F1: Full segment match (start, end, label)
    - Training/inference timing

Historical context:
    - Sarawagi & Cohen (2004): Semi-CRF improved ~1-2% over linear CRF
    - Modern encoders (BiLSTM, Transformer) have pushed overall PER down
    - But the relative advantage of duration modeling should persist

Requirements:
    pip install torchaudio librosa soundfile

    Optional (for three-way comparison with external baseline):
    pip install pytorch-crf

Note on data access:
    TIMIT requires a license from LDC (Linguistic Data Consortium).
    This code assumes the standard TIMIT directory structure:

    TIMIT/
    ├── TRAIN/
    │   ├── DR1/
    │   │   ├── FCJF0/
    │   │   │   ├── SA1.WAV
    │   │   │   ├── SA1.PHN
    │   │   │   ├── SA1.TXT
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── TEST/
        └── ...

Usage:
    # Preprocess TIMIT
    python timit_phoneme.py preprocess \
        --timit-dir /path/to/TIMIT \
        --output-dir data/timit_benchmark/

    # Train a specific model type
    python timit_phoneme.py train \
        --data-dir data/timit_benchmark/ \
        --model semicrf \
        --max-duration 30

    # Model types: pytorch-crf, linear (K=1 Triton), semicrf (K>1)
    python timit_phoneme.py train --model pytorch-crf ...
    python timit_phoneme.py train --model linear ...
    python timit_phoneme.py train --model semicrf ...

    # Three-way comparison (or two-way if pytorch-crf not installed)
    python timit_phoneme.py compare \
        --data-dir data/timit_benchmark/ \
        --output-json results/timit_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Conditional imports
try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import importlib.util

    HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None
except ImportError:
    HAS_SOUNDFILE = False

try:
    from torchcrf import CRF as TorchCRF

    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False
    TorchCRF = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# TIMIT Phone Set and Mappings
# =============================================================================

# Original 61 TIMIT phonemes
TIMIT_61_PHONES = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ax-h",
    "axr",
    "ay",
    "b",
    "bcl",
    "ch",
    "d",
    "dcl",
    "dh",
    "dx",
    "eh",
    "el",
    "em",
    "en",
    "eng",
    "epi",
    "er",
    "ey",
    "f",
    "g",
    "gcl",
    "h#",
    "hh",
    "hv",
    "ih",
    "ix",
    "iy",
    "jh",
    "k",
    "kcl",
    "l",
    "m",
    "n",
    "ng",
    "nx",
    "ow",
    "oy",
    "p",
    "pau",
    "pcl",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "tcl",
    "th",
    "uh",
    "uw",
    "ux",
    "v",
    "w",
    "y",
    "z",
    "zh",
]

# Standard 39-phone folding (Lee & Hon, 1989; Kaldi/ESPnet convention)
# Maps 61 phones to 39 classes
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONE_61_TO_39 = {
    # Vowels
    "iy": "iy",
    "ih": "ih",
    "eh": "eh",
    "ae": "ae",
    "ah": "ah",
    "uw": "uw",
    "uh": "uh",
    "aa": "aa",
    "ey": "ey",
    "ay": "ay",
    "oy": "oy",
    "aw": "aw",
    "ow": "ow",
    "er": "er",
    "ao": "aa",  # ao folds to aa (Kaldi convention)
    # Vowel reductions
    "ax": "ah",
    "ix": "ih",
    "axr": "er",
    "ax-h": "ah",
    "ux": "uw",
    # Semivowels
    "l": "l",
    "r": "r",
    "w": "w",
    "y": "y",
    "el": "l",
    "hh": "hh",
    "hv": "hh",
    # Nasals
    "m": "m",
    "n": "n",
    "ng": "ng",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "nx": "n",
    # Fricatives
    "f": "f",
    "th": "th",
    "s": "s",
    "sh": "sh",
    "v": "v",
    "dh": "dh",
    "z": "z",
    "zh": "sh",  # zh folds to sh (Kaldi convention)
    # Affricates
    "ch": "ch",
    "jh": "jh",
    # Stops
    "p": "p",
    "t": "t",
    "k": "k",
    "b": "b",
    "d": "d",
    "g": "g",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "dx": "dx",
    "q": "sil",
    # Silence
    "pau": "sil",
    "epi": "sil",
    "h#": "sil",
}

# Standard 39-phone set (Kaldi/ESPnet convention, Lee & Hon 1989)
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONES_39 = [
    "aa",
    "ae",
    "ah",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "dx",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "sil",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
]

PHONE_TO_IDX = {p: i for i, p in enumerate(PHONES_39)}
NUM_PHONES = len(PHONES_39)

# Typical phone durations (in frames at 10ms)
# Useful for understanding expected semi-CRF behavior
TYPICAL_DURATIONS = {
    "sil": (5, 50),  # Silence: variable
    "aa": (5, 15),  # Vowels: longer
    "iy": (5, 15),
    "p": (2, 8),  # Stops: short
    "t": (2, 8),
    "k": (2, 8),
    "s": (4, 15),  # Fricatives: medium
    "sh": (4, 15),
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PhoneSegment:
    """A phone segment with timing."""

    start_sample: int
    end_sample: int
    phone_61: str
    phone_39: str
    label_idx: int


@dataclass
class Utterance:
    """A TIMIT utterance."""

    utterance_id: str
    speaker_id: str
    dialect_region: str
    wav_path: Path
    phones: list[PhoneSegment]


class SegmentAnnotation(NamedTuple):
    """A segment with label (for metrics)."""

    start: int
    end: int
    label: int


# =============================================================================
# TIMIT Parsing
# =============================================================================


def parse_phn_file(phn_path: Path) -> list[tuple[int, int, str]]:
    """
    Parse a TIMIT .PHN file.

    Returns list of (start_sample, end_sample, phone) tuples.
    """
    phones = []
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = parts
                phones.append((int(start), int(end), phone.lower()))
    return phones


def load_timit_split(
    timit_dir: Path,
    split: Literal["train", "test"],
) -> list[Utterance]:
    """
    Load all utterances from a TIMIT split.
    """
    split_dir = timit_dir / split.upper()
    if not split_dir.exists():
        raise FileNotFoundError(f"TIMIT split directory not found: {split_dir}")

    utterances = []

    # Iterate through dialect regions
    for dr_dir in sorted(split_dir.iterdir()):
        if not dr_dir.is_dir() or not dr_dir.name.startswith("DR"):
            continue

        dialect_region = dr_dir.name

        # Iterate through speakers
        for speaker_dir in sorted(dr_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            # Find all .PHN files
            for phn_path in speaker_dir.glob("*.PHN"):
                # Skip SA sentences (read by all speakers, causes speaker bias)
                if phn_path.stem.upper().startswith("SA"):
                    continue

                wav_path = phn_path.with_suffix(".WAV")
                if not wav_path.exists():
                    # Try lowercase
                    wav_path = phn_path.with_suffix(".wav")

                if not wav_path.exists():
                    logger.warning(f"WAV not found for {phn_path}")
                    continue

                # Parse phone file
                raw_phones = parse_phn_file(phn_path)

                # Convert to PhoneSegments
                phone_segments = []
                for start, end, phone_61 in raw_phones:
                    phone_39 = PHONE_61_TO_39.get(phone_61, "sil")
                    label_idx = PHONE_TO_IDX.get(phone_39, PHONE_TO_IDX["sil"])

                    phone_segments.append(
                        PhoneSegment(
                            start_sample=start,
                            end_sample=end,
                            phone_61=phone_61,
                            phone_39=phone_39,
                            label_idx=label_idx,
                        )
                    )

                utterance_id = f"{dialect_region}_{speaker_id}_{phn_path.stem}"

                utterances.append(
                    Utterance(
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        dialect_region=dialect_region,
                        wav_path=wav_path,
                        phones=phone_segments,
                    )
                )

    logger.info(f"Loaded {len(utterances)} utterances from {split} split")
    return utterances


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_mfcc_features(
    audio_path: Path,
    n_mfcc: int = 13,
    n_fft: int = 400,  # 25ms at 16kHz
    hop_length: int = 160,  # 10ms at 16kHz
    sample_rate: int = 16000,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Returns:
        features: (T, D) where D = n_mfcc * 3 if include_deltas else n_mfcc
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if include_deltas:
        # Add delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])
    else:
        features = mfcc

    # Transpose to (T, D)
    return features.T


def extract_mel_features(
    audio_path: Path,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extract log mel spectrogram features.

    Returns:
        features: (T, n_mels)
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    y, sr = librosa.load(audio_path, sr=sample_rate)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Log compression
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.T


def samples_to_frames(
    sample_idx: int,
    hop_length: int = 160,
) -> int:
    """Convert sample index to frame index."""
    return sample_idx // hop_length


def align_phones_to_frames(
    phones: list[PhoneSegment],
    num_frames: int,
    hop_length: int = 160,
) -> tuple[np.ndarray, list[SegmentAnnotation]]:
    """
    Align phone segments to frame indices.

    Returns:
        labels: (T,) array of phone indices
        segments: list of SegmentAnnotation
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    segments = []

    for phone in phones:
        start_frame = samples_to_frames(phone.start_sample, hop_length)
        end_frame = samples_to_frames(phone.end_sample, hop_length)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))

        labels[start_frame:end_frame] = phone.label_idx
        segments.append(SegmentAnnotation(start_frame, end_frame, phone.label_idx))

    return labels, segments


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_timit(
    timit_dir: Path,
    output_dir: Path,
    feature_type: Literal["mfcc", "mel"] = "mfcc",
    n_mfcc: int = 13,
    n_mels: int = 80,
    hop_length: int = 160,
):
    """
    Preprocess TIMIT dataset into train/test splits.
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        logger.info(f"Processing {split} split...")

        utterances = load_timit_split(timit_dir, split)

        processed = []
        segment_lengths = defaultdict(list)

        for utt in utterances:
            try:
                # Extract features
                if feature_type == "mfcc":
                    features = extract_mfcc_features(
                        utt.wav_path,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                else:
                    features = extract_mel_features(
                        utt.wav_path,
                        n_mels=n_mels,
                        hop_length=hop_length,
                    )

                num_frames = len(features)

                # Align phones to frames
                labels, segments = align_phones_to_frames(utt.phones, num_frames, hop_length)

                # Collect segment statistics
                for seg in segments:
                    segment_lengths[seg.label].append(seg.end - seg.start)

                processed.append(
                    {
                        "utterance_id": utt.utterance_id,
                        "speaker_id": utt.speaker_id,
                        "features": features.tolist(),
                        "labels": labels.tolist(),
                        "segments": [(s.start, s.end, s.label) for s in segments],
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to process {utt.utterance_id}: {e}")
                continue

        # Save processed data
        output_file = output_dir / f"{split}.jsonl"
        logger.info(f"Saving {len(processed)} utterances to {output_file}")

        with open(output_file, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")

        # Save statistics
        stats = {}
        for label_idx, lengths in segment_lengths.items():
            lengths = np.array(lengths)
            phone_name = PHONES_39[label_idx] if label_idx < len(PHONES_39) else "unk"
            stats[phone_name] = {
                "count": len(lengths),
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "p95": float(np.percentile(lengths, 95)),
            }

        stats_file = output_dir / f"{split}_segment_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_file}")

    # Save phone mapping
    mapping_file = output_dir / "phone_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "phones_39": PHONES_39,
                "phone_to_idx": PHONE_TO_IDX,
                "phone_61_to_39": PHONE_61_TO_39,
            },
            f,
            indent=2,
        )

    logger.info("Preprocessing complete!")


# =============================================================================
# Dataset
# =============================================================================


def compute_normalization_stats(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean and std from training data.

    Args:
        data_path: Path to JSONL data file

    Returns:
        mean: (feature_dim,) array of per-dimension means
        std: (feature_dim,) array of per-dimension standard deviations
    """
    all_features = []
    with open(data_path) as f:
        for line in f:
            utt = json.loads(line)
            features = np.array(utt["features"], dtype=np.float32)
            all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    # Avoid division by zero for constant features
    std = np.maximum(std, 1e-8)
    return mean, std


class TIMITDataset(Dataset):
    """Dataset for TIMIT phoneme segmentation."""

    def __init__(
        self,
        data_path: Path,
        max_length: int | None = None,
        normalize: bool = True,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        """
        Initialize TIMIT dataset.

        Args:
            data_path: Path to JSONL data file
            max_length: Optional maximum sequence length (truncates longer sequences)
            normalize: Whether to apply z-score normalization to features
            mean: Pre-computed per-dimension means (if None and normalize=True, computed from data)
            std: Pre-computed per-dimension stds (if None and normalize=True, computed from data)
        """
        self.max_length = max_length
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.utterances = []

        with open(data_path) as f:
            for line in f:
                self.utterances.append(json.loads(line))

        # Compute stats from this dataset if not provided and normalization is requested
        if self.normalize and self.mean is None:
            logger.info("Computing normalization statistics from data...")
            self.mean, self.std = compute_normalization_stats(data_path)
            logger.info(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
            logger.info(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")

        logger.info(f"Loaded {len(self.utterances)} utterances from {data_path}")

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        utt = self.utterances[idx]

        features = np.array(utt["features"], dtype=np.float32)
        labels = np.array(utt["labels"], dtype=np.int64)

        # Apply z-score normalization
        if self.normalize and self.mean is not None:
            features = (features - self.mean) / self.std

        # Truncate if needed
        if self.max_length and len(features) > self.max_length:
            features = features[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "length": torch.tensor(len(features), dtype=torch.long),
            "utterance_id": utt["utterance_id"],
        }


def collate_timit(batch: list[dict], fixed_length: int | None = None) -> dict[str, Tensor]:
    """Collate TIMIT batch with padding.

    Args:
        batch: List of sample dictionaries.
        fixed_length: If provided, force all sequences to this exact length.
            Sequences shorter are padded, longer are truncated. This is useful
            for debugging to eliminate variable-length boundary handling issues.
    """
    if fixed_length is not None:
        max_len = fixed_length
    else:
        max_len = max(b["length"].item() for b in batch)

    features = []
    labels = []
    lengths = []
    utterance_ids = []

    for b in batch:
        feat = b["features"]
        lab = b["labels"]
        seq_len = b["length"].item()

        # Handle fixed length: truncate or pad
        if fixed_length is not None:
            if seq_len > fixed_length:
                # Truncate
                feat = feat[:fixed_length]
                lab = lab[:fixed_length]
                seq_len = fixed_length
            elif seq_len < fixed_length:
                # Pad
                pad_len = fixed_length - seq_len
                feat = F.pad(feat, (0, 0, 0, pad_len))
                lab = F.pad(lab, (0, pad_len), value=0)
            # For fixed length, report the fixed length as the actual length
            lengths.append(torch.tensor(fixed_length, dtype=torch.long))
        else:
            # Variable length: just pad to max_len in batch
            if seq_len < max_len:
                pad_len = max_len - seq_len
                feat = F.pad(feat, (0, 0, 0, pad_len))
                lab = F.pad(lab, (0, pad_len), value=0)
            lengths.append(b["length"])

        features.append(feat)
        labels.append(lab)
        utterance_ids.append(b["utterance_id"])

    return {
        "features": torch.stack(features),
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths),
        "utterance_ids": utterance_ids,
    }


def make_collate_fn(fixed_length: int | None = None):
    """Create a collate function with optional fixed length.

    Args:
        fixed_length: If provided, force all sequences to this length.

    Returns:
        A collate function suitable for DataLoader.
    """

    def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
        return collate_timit(batch, fixed_length=fixed_length)

    return collate_fn


# =============================================================================
# Models
# =============================================================================


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for acoustic features."""

    def __init__(
        self,
        input_dim: int = 39,  # 13 MFCC * 3
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            hidden: (batch, T, hidden_dim)
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        output, _ = self.lstm(x)
        return output


class TIMITModel(nn.Module):
    """
    Combined encoder + CRF head for TIMIT phoneme segmentation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_PHONES,
        max_duration: int = 1,
        hidden_dim: int = 256,
        duration_distribution: str = "learned",
    ):
        super().__init__()
        self.encoder = encoder
        self.max_duration = max_duration

        # Use UncertaintySemiMarkovCRFHead for calibration/uncertainty estimation
        from torch_semimarkov import UncertaintySemiMarkovCRFHead

        self.crf = UncertaintySemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=hidden_dim,
            duration_distribution=duration_distribution,
        )

    def forward(self, features: Tensor, lengths: Tensor) -> dict:
        hidden = self.encoder(features)
        return self.crf(hidden, lengths)

    def compute_loss(
        self,
        features: Tensor,
        lengths: Tensor,
        labels: Tensor,
        backend: str = "exact",
        use_triton: bool = False,
    ) -> Tensor:
        """
        Compute NLL loss for phoneme segmentation.

        Args:
            features: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            backend: "exact", "streaming", or "auto"
            use_triton: Whether to use Triton kernels (streaming only)
        """
        hidden = self.encoder(features)
        return self.crf.compute_loss(
            hidden, lengths, labels, backend=backend, use_triton=use_triton
        )

    def decode(self, features: Tensor, lengths: Tensor):
        hidden = self.encoder(features)
        return self.crf.decode_with_traceback(hidden, lengths)


class TIMITModelPytorchCRF(nn.Module):
    """
    TIMIT model using pytorch-crf for baseline comparison.

    Uses the same BiLSTMEncoder but replaces SemiMarkovCRFHead with torchcrf.CRF.
    This enables fair comparison between pytorch-crf and torch-semimarkov K=1.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_PHONES,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model. Install with: pip install pytorch-crf"
            )
        self.encoder = encoder
        self.emission_proj = nn.Linear(hidden_dim, num_classes)
        self.crf = TorchCRF(num_classes, batch_first=True)

    def forward(self, features: Tensor, _lengths: Tensor) -> Tensor:
        """Forward pass returning emission scores."""
        hidden = self.encoder(features)
        return self.emission_proj(hidden)

    def compute_loss(
        self,
        features: Tensor,
        lengths: Tensor,
        labels: Tensor,
        **_kwargs,
    ) -> Tensor:
        """
        Compute NLL loss using pytorch-crf.

        Args:
            features: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            **_kwargs: Ignored (for API compatibility with TIMITModel)
        """
        hidden = self.encoder(features)
        emissions = self.emission_proj(hidden)

        # pytorch-crf expects mask of shape (batch, seq_len)
        _, seq_len = features.shape[:2]
        mask = torch.arange(seq_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)

        # pytorch-crf.forward() returns log-likelihood, we want NLL
        log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(self, features: Tensor, lengths: Tensor) -> list[list[int]]:
        """
        Viterbi decode to get best label sequences.

        Returns:
            List of label sequences (one per batch element).
        """
        hidden = self.encoder(features)
        emissions = self.emission_proj(hidden)

        _, seq_len = features.shape[:2]
        mask = torch.arange(seq_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Returns list of lists (batch_size x variable_length)
        return self.crf.decode(emissions, mask=mask)


# =============================================================================
# Evaluation Metrics
# =============================================================================


def levenshtein_distance(s1: list, s2: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_phone_error_rate(
    predictions: list[list[int]],
    references: list[list[int]],
) -> float:
    """
    Compute Phone Error Rate (PER).

    PER = (substitutions + insertions + deletions) / reference_length
    """
    total_distance = 0
    total_ref_length = 0

    for pred, ref in zip(predictions, references, strict=False):
        # Convert frame-level to segment-level (collapse consecutive)
        pred_segments = collapse_to_segments(pred)
        ref_segments = collapse_to_segments(ref)

        total_distance += levenshtein_distance(pred_segments, ref_segments)
        total_ref_length += len(ref_segments)

    return total_distance / total_ref_length if total_ref_length > 0 else 0


def collapse_to_segments(labels: list[int]) -> list[int]:
    """Collapse frame-level labels to segment-level (remove consecutive duplicates)."""
    if not labels:
        return []

    segments = [labels[0]]
    for label in labels[1:]:
        if label != segments[-1]:
            segments.append(label)

    return segments


def extract_boundaries(labels: list[int]) -> set[int]:
    """Extract boundary positions from label sequence."""
    boundaries = set()
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_metrics(
    predictions: list[list[int]],
    references: list[list[int]],
    tolerances: list[int] | None = None,
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    if tolerances is None:
        tolerances = [0, 1, 2]
    results = {f"tol_{t}": {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}

    for pred, ref in zip(predictions, references, strict=False):
        pred_bounds = extract_boundaries(pred)
        ref_bounds = extract_boundaries(ref)

        for tol in tolerances:
            key = f"tol_{tol}"
            matched_ref = set()

            for pb in pred_bounds:
                for rb in ref_bounds:
                    if abs(pb - rb) <= tol and rb not in matched_ref:
                        results[key]["tp"] += 1
                        matched_ref.add(rb)
                        break
                else:
                    results[key]["fp"] += 1

            results[key]["fn"] += len(ref_bounds) - len(matched_ref)

    metrics = {}
    for tol in tolerances:
        key = f"tol_{tol}"
        tp = results[key]["tp"]
        fp = results[key]["fp"]
        fn = results[key]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"boundary_f1_tol{tol}"] = f1
        if tol == 0:
            metrics["boundary_precision"] = precision
            metrics["boundary_recall"] = recall

    return metrics


def compute_segment_metrics(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
) -> dict[str, float]:
    """Compute segment-level metrics."""
    tp = 0
    total_pred = 0
    total_true = 0

    for pred_segs, true_segs in zip(pred_segments, true_segments, strict=False):
        pred_set = {(s.start, s.end, s.label) for s in pred_segs}
        true_set = {(s.start, s.end, s.label) for s in true_segs}

        tp += len(pred_set & true_set)
        total_pred += len(pred_set)
        total_true += len(true_set)

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
    }


def compute_duration_stats(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
) -> dict[str, dict[str, float]]:
    """
    Compute per-phoneme duration statistics for predictions vs references.

    Returns dict with:
        - per_phone: {phone_name: {pred_mean, pred_std, ref_mean, ref_std, mae, count}}
        - overall: {mean_absolute_error, correlation}
    """
    pred_durations = defaultdict(list)
    ref_durations = defaultdict(list)

    for pred_segs, ref_segs in zip(pred_segments, true_segments, strict=False):
        for seg in pred_segs:
            pred_durations[seg.label].append(seg.end - seg.start)
        for seg in ref_segs:
            ref_durations[seg.label].append(seg.end - seg.start)

    per_phone = {}
    all_pred_means = []
    all_ref_means = []

    for label in range(NUM_PHONES):
        phone_name = PHONES_39[label]
        pred_d = np.array(pred_durations[label]) if pred_durations[label] else np.array([0])
        ref_d = np.array(ref_durations[label]) if ref_durations[label] else np.array([0])

        pred_mean = float(np.mean(pred_d))
        ref_mean = float(np.mean(ref_d))

        per_phone[phone_name] = {
            "pred_mean": pred_mean,
            "pred_std": float(np.std(pred_d)),
            "ref_mean": ref_mean,
            "ref_std": float(np.std(ref_d)),
            "mae": abs(pred_mean - ref_mean),
            "pred_count": len(pred_durations[label]),
            "ref_count": len(ref_durations[label]),
        }

        if ref_durations[label]:  # Only include phones that appear in reference
            all_pred_means.append(pred_mean)
            all_ref_means.append(ref_mean)

    # Overall correlation between predicted and reference mean durations
    if len(all_pred_means) > 1:
        correlation = float(np.corrcoef(all_pred_means, all_ref_means)[0, 1])
    else:
        correlation = 0.0

    overall_mae = float(
        np.mean([abs(p - r) for p, r in zip(all_pred_means, all_ref_means, strict=True)])
    )

    return {
        "per_phone": per_phone,
        "overall": {
            "mean_absolute_error": overall_mae,
            "duration_correlation": correlation,
        },
    }


def labels_to_segments(labels: list[int]) -> list[SegmentAnnotation]:
    """Convert label sequence to segments."""
    if not labels:
        return []

    segments = []
    current_label = labels[0]
    current_start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append(SegmentAnnotation(current_start, i, current_label))
            current_label = labels[i]
            current_start = i

    segments.append(SegmentAnnotation(current_start, len(labels), current_label))
    return segments


@dataclass
class TIMITMetrics:
    """Metrics for TIMIT evaluation."""

    phone_error_rate: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_f1_tolerances: dict[int, float]
    segment_precision: float
    segment_recall: float
    segment_f1: float
    # Duration analysis
    duration_stats: dict | None = None
    # Timing metrics (optional, set during training)
    training_time_per_epoch: float = 0.0  # seconds
    total_training_time: float = 0.0  # seconds
    inference_time: float = 0.0  # seconds for full test set

    def to_dict(self) -> dict:
        """Convert metrics to JSON-serializable dict."""
        result = {
            "phone_error_rate": self.phone_error_rate,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "boundary_f1_tolerances": {str(k): v for k, v in self.boundary_f1_tolerances.items()},
            "segment_precision": self.segment_precision,
            "segment_recall": self.segment_recall,
            "segment_f1": self.segment_f1,
            "training_time_per_epoch": self.training_time_per_epoch,
            "total_training_time": self.total_training_time,
            "inference_time": self.inference_time,
        }
        if self.duration_stats:
            result["duration_stats"] = self.duration_stats
        return result


# =============================================================================
# Training Loop
# =============================================================================


def train_epoch(
    model: TIMITModel | TIMITModelPytorchCRF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    backend: str = "streaming",
    use_triton: bool = True,
    crf_reg: float = 0.0,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only).
            Helps prevent gradient explosion from unbounded transition/duration_bias.

    Returns:
        Tuple of (average_loss, elapsed_time_seconds).
    """
    model.train()
    total_loss = 0
    num_batches = 0

    start_time = time.perf_counter()

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(features, lengths, labels, backend=backend, use_triton=use_triton)

        # Add CRF parameter regularization for Semi-Markov models
        if crf_reg > 0 and isinstance(model, TIMITModel):
            loss = loss + crf_reg * model.crf.parameter_penalty()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    elapsed = time.perf_counter() - start_time
    return total_loss / num_batches, elapsed


@torch.no_grad()
def evaluate(
    model: TIMITModel | TIMITModelPytorchCRF,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[TIMITMetrics, float]:
    """Evaluate model.

    Returns:
        Tuple of (metrics, elapsed_time_seconds).
    """
    model.eval()

    all_predictions = []
    all_references = []
    all_pred_segments = []
    all_true_segments = []

    # Check if this is a pytorch-crf model (returns list) vs TIMITModel (returns ViterbiResult)
    is_pytorch_crf = isinstance(model, TIMITModelPytorchCRF)

    start_time = time.perf_counter()

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        result = model.decode(features, lengths)

        for i in range(len(lengths)):
            seq_len = lengths[i].item()

            if is_pytorch_crf:
                # pytorch-crf returns list of label sequences directly
                pred_labels = result[i][:seq_len]
            else:
                # TIMITModel returns ViterbiResult with segments
                # NOTE: torch_semimarkov.Segment uses INCLUSIVE end (end=5 means position 5 included)
                # Convert to exclusive for iteration: range(start, end+1)
                pred_labels = [0] * seq_len
                for seg in result.segments[i]:
                    for j in range(seg.start, min(seg.end + 1, seq_len)):
                        pred_labels[j] = seg.label

            ref_labels = labels[i, :seq_len].cpu().tolist()

            all_predictions.append(pred_labels)
            all_references.append(ref_labels)

            # Both paths use labels_to_segments for consistent segment merging
            # This ensures consecutive frames with the same label are merged into single segments
            # (critical for K=1 semimarkov which returns per-frame segments from Viterbi)
            pred_segs = labels_to_segments(pred_labels)
            true_segs = labels_to_segments(ref_labels)

            all_pred_segments.append(pred_segs)
            all_true_segments.append(true_segs)

    elapsed = time.perf_counter() - start_time

    per = compute_phone_error_rate(all_predictions, all_references)
    boundary_metrics = compute_boundary_metrics(all_predictions, all_references)
    segment_metrics = compute_segment_metrics(all_pred_segments, all_true_segments)
    duration_stats = compute_duration_stats(all_pred_segments, all_true_segments)

    metrics = TIMITMetrics(
        phone_error_rate=per,
        boundary_precision=boundary_metrics["boundary_precision"],
        boundary_recall=boundary_metrics["boundary_recall"],
        boundary_f1=boundary_metrics.get("boundary_f1_tol0", 0),
        boundary_f1_tolerances={
            int(k.split("tol")[1]): v
            for k, v in boundary_metrics.items()
            if k.startswith("boundary_f1_tol")
        },
        segment_precision=segment_metrics["segment_precision"],
        segment_recall=segment_metrics["segment_recall"],
        segment_f1=segment_metrics["segment_f1"],
        duration_stats=duration_stats,
    )
    return metrics, elapsed


def train_model(
    data_dir: Path,
    model_type: Literal["pytorch-crf", "linear", "semicrf"] = "semicrf",
    max_duration: int = 30,
    hidden_dim: int = 256,
    num_layers: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    backend: str = "streaming",
    use_triton: bool = True,
    log_every: int = 1,
    crf_reg: float = 0.0,
    fixed_length: int | None = None,
) -> tuple[TIMITModel | TIMITModelPytorchCRF, TIMITMetrics]:
    """Train a model and return it with metrics.

    Args:
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only).
        fixed_length: If provided, force all sequences to this length (for debugging).
    """
    device = torch.device(device)

    # Load data with normalization
    # Training dataset computes normalization stats from its own data
    train_dataset = TIMITDataset(data_dir / "train.jsonl", normalize=True)

    # Test dataset uses training stats to ensure consistent normalization
    test_dataset = TIMITDataset(
        data_dir / "test.jsonl",
        normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    # Determine feature dimension from first sample
    sample = train_dataset[0]
    input_dim = sample["features"].shape[-1]

    # Create collate function (with optional fixed length for debugging)
    collate_fn = make_collate_fn(fixed_length=fixed_length)
    if fixed_length is not None:
        logger.info(f"Using fixed sequence length: {fixed_length}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Build model
    encoder = BiLSTMEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if model_type == "pytorch-crf":
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model type. " "Install with: pip install pytorch-crf"
            )
        model = TIMITModelPytorchCRF(
            encoder=encoder,
            hidden_dim=hidden_dim,
        ).to(device)
        k = 1  # For logging purposes
    elif model_type == "linear":
        k = 1
        model = TIMITModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
        ).to(device)
    else:  # semicrf
        k = max_duration
        model = TIMITModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
        ).to(device)

    logger.info(
        f"Model: {model_type}, K={k}, params={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_per = float("inf")
    best_metrics = None
    total_train_time = 0.0
    epoch_times = []

    for epoch in range(epochs):
        train_loss, epoch_time = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            backend=backend,
            use_triton=use_triton,
            crf_reg=crf_reg,
        )
        total_train_time += epoch_time
        epoch_times.append(epoch_time)
        scheduler.step()

        # Log CRF parameter magnitudes for debugging gradient explosion
        if isinstance(model, TIMITModel):
            trans_max = model.crf.transition.abs().max().item()
            dur_max = model.crf.duration_bias.abs().max().item()
            logger.debug(
                f"Epoch {epoch+1} CRF params: "
                f"transition_max={trans_max:.4f}, duration_bias_max={dur_max:.4f}"
            )
            # Warn if parameters are drifting to extreme values
            if trans_max > 20 or dur_max > 20:
                logger.warning(
                    f"Epoch {epoch+1}: CRF parameters drifting high! "
                    f"trans_max={trans_max:.2f}, dur_max={dur_max:.2f}. "
                    f"Consider increasing --crf-reg."
                )

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            test_metrics, inference_time = evaluate(model, test_loader, device)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"PER: {test_metrics.phone_error_rate:.4f} | "
                f"Boundary F1: {test_metrics.boundary_f1:.4f} | "
                f"Segment F1: {test_metrics.segment_f1:.4f} | "
                f"Train: {epoch_time:.1f}s | Infer: {inference_time:.1f}s"
            )

            if test_metrics.phone_error_rate < best_per:
                best_per = test_metrics.phone_error_rate
                # Update metrics with timing info
                test_metrics.total_training_time = total_train_time
                test_metrics.training_time_per_epoch = sum(epoch_times) / len(epoch_times)
                test_metrics.inference_time = inference_time
                best_metrics = test_metrics

    return model, best_metrics


def _print_duration_analysis(results: dict, has_pytorch_crf: bool = False):
    """Print duration distribution analysis comparing models."""
    print("\n" + "=" * 60)
    print("DURATION ANALYSIS (frames @ 10ms)")
    print("=" * 60)
    print("\nThis shows how well each model captures phoneme duration patterns.")
    print("Lower MAE = better duration modeling. Semi-CRF should excel here.\n")

    # Get reference durations from any model (they're all the same)
    ref_model = "semi_crf"
    ref_stats = results[ref_model].duration_stats["per_phone"]

    # Select phones to display (most frequent + phonetically interesting)
    # Prioritize: vowels (long), stops (short), fricatives (medium)
    interesting_phones = ["aa", "iy", "eh", "ah", "p", "t", "k", "s", "sh", "n", "l", "sil"]
    display_phones = [p for p in interesting_phones if ref_stats[p]["ref_count"] > 50]

    if has_pytorch_crf:
        print(
            f"{'Phone':<6} {'Ref':>6} {'p-crf':>6} {'K=1':>6} {'Semi':>6} │ {'MAE p-crf':>9} {'MAE K=1':>9} {'MAE Semi':>9}"
        )
        print("-" * 85)

        p_stats = results["pytorch_crf"].duration_stats["per_phone"]
        l_stats = results["linear_crf_triton"].duration_stats["per_phone"]
        s_stats = results["semi_crf"].duration_stats["per_phone"]

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            p_mean = p_stats[phone]["pred_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            s_mean = s_stats[phone]["pred_mean"]
            p_mae = p_stats[phone]["mae"]
            l_mae = l_stats[phone]["mae"]
            s_mae = s_stats[phone]["mae"]

            # Highlight if semi-CRF is better
            s_marker = "*" if s_mae < p_mae and s_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {p_mean:>6.1f} {l_mean:>6.1f} {s_mean:>6.1f} │ "
                f"{p_mae:>9.2f} {l_mae:>9.2f} {s_mae:>8.2f}{s_marker}"
            )

        # Overall stats
        print("-" * 85)
        p_overall = results["pytorch_crf"].duration_stats["overall"]
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        s_overall = results["semi_crf"].duration_stats["overall"]

        print(
            f"{'Overall MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['mean_absolute_error']:>9.2f} {l_overall['mean_absolute_error']:>9.2f} "
            f"{s_overall['mean_absolute_error']:>8.2f}"
        )
        print(
            f"{'Duration r':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['duration_correlation']:>9.3f} {l_overall['duration_correlation']:>9.3f} "
            f"{s_overall['duration_correlation']:>8.3f}"
        )
    else:
        print(f"{'Phone':<6} {'Ref':>6} {'K=1':>6} {'Semi':>6} │ {'MAE K=1':>9} {'MAE Semi':>9}")
        print("-" * 60)

        l_stats = results["linear_crf_triton"].duration_stats["per_phone"]
        s_stats = results["semi_crf"].duration_stats["per_phone"]

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            s_mean = s_stats[phone]["pred_mean"]
            l_mae = l_stats[phone]["mae"]
            s_mae = s_stats[phone]["mae"]

            s_marker = "*" if s_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {l_mean:>6.1f} {s_mean:>6.1f} │ "
                f"{l_mae:>9.2f} {s_mae:>8.2f}{s_marker}"
            )

        # Overall stats
        print("-" * 60)
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        s_overall = results["semi_crf"].duration_stats["overall"]

        print(
            f"{'Overall MAE':<6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['mean_absolute_error']:>9.2f} {s_overall['mean_absolute_error']:>8.2f}"
        )
        print(
            f"{'Duration r':<6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['duration_correlation']:>9.3f} {s_overall['duration_correlation']:>8.3f}"
        )

    print("\n* = Semi-CRF has lowest MAE for this phone")
    print("Duration r = correlation between predicted and reference mean durations")


def compare_models(data_dir: Path, max_duration: int = 30, **kwargs):
    """
    Compare CRF models: pytorch-crf (optional), linear CRF (K=1), and semi-CRF.

    If pytorch-crf is installed, runs a three-way comparison. Otherwise, runs
    a two-way comparison between torch-semimarkov K=1 and K>1.
    """
    results = {}

    # 1. pytorch-crf baseline (optional - skip if not installed)
    if HAS_TORCHCRF:
        logger.info("=" * 60)
        logger.info("Training PYTORCH-CRF (external library baseline)")
        logger.info("=" * 60)
        _, pytorch_crf_metrics = train_model(data_dir, model_type="pytorch-crf", **kwargs)
        results["pytorch_crf"] = pytorch_crf_metrics
    else:
        logger.warning("=" * 60)
        logger.warning("pytorch-crf not installed, skipping baseline comparison")
        logger.warning("Install with: pip install pytorch-crf")
        logger.warning("=" * 60)

    # 2. torch-semimarkov K=1 (linear CRF via Triton streaming)
    logger.info("=" * 60)
    logger.info("Training LINEAR CRF (torch-semimarkov K=1, Triton)")
    logger.info("=" * 60)
    _, linear_metrics = train_model(data_dir, model_type="linear", **kwargs)
    results["linear_crf_triton"] = linear_metrics

    # 3. torch-semimarkov K>1 (semi-CRF)
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF (torch-semimarkov K={max_duration})")
    logger.info("=" * 60)
    _, semicrf_metrics = train_model(
        data_dir, model_type="semicrf", max_duration=max_duration, **kwargs
    )
    results["semi_crf"] = semicrf_metrics

    # Print comparison
    logger.info("\n" + "=" * 60)
    if HAS_TORCHCRF:
        logger.info("COMPARISON: pytorch-crf vs K=1 Triton vs Semi-CRF")
    else:
        logger.info("COMPARISON: K=1 Triton vs Semi-CRF")
    logger.info("=" * 60)

    if HAS_TORCHCRF:
        # Three-way comparison
        print(
            f"\n{'Metric':<25} {'pytorch-crf':>15} {'K=1 Triton':>15} "
            f"{'Semi-CRF':>15} {'Δ vs baseline':>15}"
        )
        print("-" * 85)

        # PER (lower is better)
        p_per = results["pytorch_crf"].phone_error_rate
        l_per = results["linear_crf_triton"].phone_error_rate
        s_per = results["semi_crf"].phone_error_rate
        print(
            f"{'Phone Error Rate':<25} {p_per:>15.4f} {l_per:>15.4f} "
            f"{s_per:>15.4f} {s_per - p_per:>+15.4f}"
        )

        # F1 scores (higher is better)
        for metric in ["boundary_f1", "segment_f1"]:
            p_val = getattr(results["pytorch_crf"], metric)
            l_val = getattr(results["linear_crf_triton"], metric)
            s_val = getattr(results["semi_crf"], metric)
            print(
                f"{metric:<25} {p_val:>15.4f} {l_val:>15.4f} "
                f"{s_val:>15.4f} {s_val - p_val:>+15.4f}"
            )

        print("\nBoundary F1 at different tolerances:")
        for tol in [0, 1, 2]:
            p_val = results["pytorch_crf"].boundary_f1_tolerances.get(tol, 0)
            l_val = results["linear_crf_triton"].boundary_f1_tolerances.get(tol, 0)
            s_val = results["semi_crf"].boundary_f1_tolerances.get(tol, 0)
            print(
                f"  tol={tol:<2} {p_val:>15.4f} {l_val:>15.4f} "
                f"{s_val:>15.4f} {s_val - p_val:>+15.4f}"
            )

        # Timing metrics
        print("\nTiming (lower is better):")
        p_time = results["pytorch_crf"].training_time_per_epoch
        l_time = results["linear_crf_triton"].training_time_per_epoch
        s_time = results["semi_crf"].training_time_per_epoch
        print(
            f"{'Train time (s/epoch)':<25} {p_time:>15.2f} {l_time:>15.2f} "
            f"{s_time:>15.2f} {l_time - p_time:>+15.2f}*"
        )
        p_infer = results["pytorch_crf"].inference_time
        l_infer = results["linear_crf_triton"].inference_time
        s_infer = results["semi_crf"].inference_time
        print(
            f"{'Inference time (s)':<25} {p_infer:>15.2f} {l_infer:>15.2f} "
            f"{s_infer:>15.2f} {l_infer - p_infer:>+15.2f}*"
        )
        print("* Δ shows K=1 Triton vs pytorch-crf (negative = faster)")

        # Note about K=1 vs pytorch-crf equivalence
        print("\nNote: K=1 Triton and pytorch-crf should produce similar accuracy")
        print("(validates that K=1 is a correct linear CRF implementation)")

        # Duration analysis
        _print_duration_analysis(results, has_pytorch_crf=True)
    else:
        # Two-way comparison (no pytorch-crf)
        print(f"\n{'Metric':<25} {'K=1 Triton':>15} {'Semi-CRF':>15} {'Δ':>12}")
        print("-" * 67)

        # PER (lower is better)
        l_per = results["linear_crf_triton"].phone_error_rate
        s_per = results["semi_crf"].phone_error_rate
        print(f"{'Phone Error Rate':<25} {l_per:>15.4f} {s_per:>15.4f} {s_per - l_per:>+12.4f}")

        # F1 scores (higher is better)
        for metric in ["boundary_f1", "segment_f1"]:
            l_val = getattr(results["linear_crf_triton"], metric)
            s_val = getattr(results["semi_crf"], metric)
            print(f"{metric:<25} {l_val:>15.4f} {s_val:>15.4f} {s_val - l_val:>+12.4f}")

        print("\nBoundary F1 at different tolerances:")
        for tol in [0, 1, 2]:
            l_val = results["linear_crf_triton"].boundary_f1_tolerances.get(tol, 0)
            s_val = results["semi_crf"].boundary_f1_tolerances.get(tol, 0)
            print(f"  tol={tol:<2} {l_val:>15.4f} {s_val:>15.4f} {s_val - l_val:>+12.4f}")

        # Timing metrics
        print("\nTiming (lower is better):")
        l_time = results["linear_crf_triton"].training_time_per_epoch
        s_time = results["semi_crf"].training_time_per_epoch
        print(
            f"{'Train time (s/epoch)':<25} {l_time:>15.2f} {s_time:>15.2f} {s_time - l_time:>+12.2f}"
        )
        l_infer = results["linear_crf_triton"].inference_time
        s_infer = results["semi_crf"].inference_time
        print(
            f"{'Inference time (s)':<25} {l_infer:>15.2f} {s_infer:>15.2f} {s_infer - l_infer:>+12.2f}"
        )

        # Duration analysis
        _print_duration_analysis(results, has_pytorch_crf=False)

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess TIMIT")
    preprocess_parser.add_argument(
        "--timit-dir", type=Path, required=True, help="TIMIT root directory"
    )
    preprocess_parser.add_argument("--output-dir", type=Path, required=True)
    preprocess_parser.add_argument("--feature-type", choices=["mfcc", "mel"], default="mfcc")
    preprocess_parser.add_argument("--n-mfcc", type=int, default=13)
    preprocess_parser.add_argument("--n-mels", type=int, default=80)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument(
        "--model",
        choices=["pytorch-crf", "linear", "semicrf"],
        default="semicrf",
        help="Model type: pytorch-crf (external lib), linear (K=1 Triton), semicrf (K>1)",
    )
    train_parser.add_argument("--max-duration", type=int, default=30)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=3)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument(
        "--crf-reg",
        type=float,
        default=0.0,
        help="L2 regularization coefficient for CRF parameters (transition, duration_bias). "
        "Helps prevent gradient explosion in Semi-Markov CRF training. Typical values: 0.001-0.1",
    )
    train_parser.add_argument(
        "--backend",
        choices=["exact", "streaming", "auto"],
        default="streaming",
        help="CRF backend: exact (materialize edges), streaming (on-the-fly), auto (heuristic)",
    )
    train_parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch reference implementation)",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--fixed-length",
        type=int,
        default=None,
        help="Force all sequences to this fixed length (for debugging boundary handling). "
        "Sequences shorter are padded, longer are truncated.",
    )

    # Compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare CRF models (pytorch-crf if installed, K=1 Triton, Semi-CRF)",
    )
    compare_parser.add_argument("--data-dir", type=Path, required=True)
    compare_parser.add_argument("--max-duration", type=int, default=30)
    compare_parser.add_argument("--hidden-dim", type=int, default=256)
    compare_parser.add_argument("--num-layers", type=int, default=3)
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument(
        "--output-json", type=Path, default=None, help="Save results to JSON file"
    )
    compare_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics every N epochs (default: 1)",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_timit(
            args.timit_dir,
            args.output_dir,
            feature_type=args.feature_type,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
        )
    elif args.command == "train":
        train_model(
            args.data_dir,
            model_type=args.model,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            backend=args.backend,
            use_triton=not args.no_triton,
            log_every=args.log_every,
            crf_reg=args.crf_reg,
            fixed_length=args.fixed_length,
        )
    elif args.command == "compare":
        results = compare_models(
            args.data_dir,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=args.log_every,
        )
        if args.output_json:
            from datetime import datetime

            output = {
                "task": "timit",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_duration": args.max_duration,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                "linear_crf_triton": results["linear_crf_triton"].to_dict(),
                "semi_crf": results["semi_crf"].to_dict(),
            }
            # Include pytorch-crf results if available
            if "pytorch_crf" in results:
                output["pytorch_crf"] = results["pytorch_crf"].to_dict()
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
