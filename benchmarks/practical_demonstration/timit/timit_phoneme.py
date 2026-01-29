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
    - A linear CRF cannot encode "this phoneme typically lasts 50-100ms"
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

    def decode(self, features: Tensor, lengths: Tensor, backend: str = "streaming"):
        hidden = self.encoder(features)
        return self.crf.decode_with_traceback(hidden, lengths, backend=backend)


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


# =============================================================================
# Enhanced Duration Analysis Functions
# =============================================================================


def load_corpus_duration_stats(data_dir: Path) -> dict:
    """Load raw TIMIT duration statistics from preprocessing.

    Args:
        data_dir: Path to preprocessed TIMIT data directory

    Returns:
        Dictionary with per-phoneme statistics from train_segment_stats.json
    """
    stats_file = data_dir / "train_segment_stats.json"
    if not stats_file.exists():
        logger.warning(f"Corpus stats file not found: {stats_file}")
        return {}

    with open(stats_file) as f:
        return json.load(f)


def compute_kl_divergence(
    pred_durations: list[int], ref_durations: list[int], max_dur: int = 50
) -> float:
    """Compute KL divergence between predicted and reference duration distributions.

    Args:
        pred_durations: List of predicted segment durations
        ref_durations: List of reference segment durations
        max_dur: Maximum duration for histogram binning

    Returns:
        KL divergence D_KL(pred || ref), or 0 if insufficient data
    """
    if len(pred_durations) < 5 or len(ref_durations) < 5:
        return 0.0

    # Bin durations into histogram
    bins = range(1, max_dur + 2)
    pred_hist, _ = np.histogram(pred_durations, bins=bins, density=True)
    ref_hist, _ = np.histogram(ref_durations, bins=bins, density=True)

    # Add smoothing to avoid log(0)
    eps = 1e-10
    pred_hist = pred_hist + eps
    ref_hist = ref_hist + eps

    # Normalize
    pred_hist = pred_hist / pred_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # KL divergence: D_KL(pred || ref) = sum(pred * log(pred / ref))
    return float(np.sum(pred_hist * np.log(pred_hist / ref_hist)))


def compute_js_divergence(
    pred_durations: list[int], ref_durations: list[int], max_dur: int = 50
) -> float:
    """Compute Jensen-Shannon divergence (symmetric alternative to KL).

    Args:
        pred_durations: List of predicted segment durations
        ref_durations: List of reference segment durations
        max_dur: Maximum duration for histogram binning

    Returns:
        JS divergence, or 0 if insufficient data
    """
    if len(pred_durations) < 5 or len(ref_durations) < 5:
        return 0.0

    # Bin durations into histogram
    bins = range(1, max_dur + 2)
    pred_hist, _ = np.histogram(pred_durations, bins=bins, density=True)
    ref_hist, _ = np.histogram(ref_durations, bins=bins, density=True)

    # Add smoothing
    eps = 1e-10
    pred_hist = pred_hist + eps
    ref_hist = ref_hist + eps

    # Normalize
    pred_hist = pred_hist / pred_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # Midpoint distribution
    m = (pred_hist + ref_hist) / 2

    # JS divergence = (KL(pred || m) + KL(ref || m)) / 2
    kl_pm = np.sum(pred_hist * np.log(pred_hist / m))
    kl_rm = np.sum(ref_hist * np.log(ref_hist / m))

    return float((kl_pm + kl_rm) / 2)


def compute_enhanced_duration_stats(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
    max_dur: int = 50,
) -> dict:
    """Compute enhanced duration statistics including KL divergence.

    Returns dict with:
        - per_phone: {phone_name: {..., kl_div, js_div}}
        - overall: {..., weighted_kl, weighted_js}
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
    weighted_kl = 0.0
    weighted_js = 0.0
    total_ref_count = 0

    for label in range(NUM_PHONES):
        phone_name = PHONES_39[label]
        pred_d = list(pred_durations[label]) if pred_durations[label] else [0]
        ref_d = list(ref_durations[label]) if ref_durations[label] else [0]

        pred_mean = float(np.mean(pred_d))
        ref_mean = float(np.mean(ref_d))

        # Compute KL and JS divergence
        kl_div = compute_kl_divergence(pred_d, ref_d, max_dur)
        js_div = compute_js_divergence(pred_d, ref_d, max_dur)

        per_phone[phone_name] = {
            "pred_mean": pred_mean,
            "pred_std": float(np.std(pred_d)),
            "ref_mean": ref_mean,
            "ref_std": float(np.std(ref_d)),
            "mae": abs(pred_mean - ref_mean),
            "pred_count": len(pred_durations[label]),
            "ref_count": len(ref_durations[label]),
            "kl_divergence": kl_div,
            "js_divergence": js_div,
        }

        if ref_durations[label]:
            all_pred_means.append(pred_mean)
            all_ref_means.append(ref_mean)
            # Weighted by reference count
            weighted_kl += kl_div * len(ref_durations[label])
            weighted_js += js_div * len(ref_durations[label])
            total_ref_count += len(ref_durations[label])

    # Overall correlation
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
            "weighted_kl_divergence": weighted_kl / total_ref_count if total_ref_count > 0 else 0,
            "weighted_js_divergence": weighted_js / total_ref_count if total_ref_count > 0 else 0,
        },
    }


def export_duration_analysis(results: dict, corpus_stats: dict, output_path: Path):
    """Export detailed duration analysis to JSON and CSV.

    Args:
        results: Dictionary of model results (each with TIMITMetrics)
        corpus_stats: Raw TIMIT corpus statistics from preprocessing
        output_path: Base path for output files (will create .json and .csv)
    """
    import csv

    output = {
        "corpus_stats": corpus_stats,
        "model_comparisons": {},
    }

    for model_name, metrics in results.items():
        if hasattr(metrics, "duration_stats") and metrics.duration_stats:
            output["model_comparisons"][model_name] = {
                "per_phone": metrics.duration_stats["per_phone"],
                "overall": metrics.duration_stats["overall"],
            }

    # JSON export
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Duration analysis JSON saved to {json_path}")

    # CSV export (flattened per-phoneme table)
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phone",
                "corpus_mean",
                "corpus_std",
                "corpus_p95",
                "model",
                "pred_mean",
                "pred_std",
                "mae",
                "kl_div",
            ]
        )

        for phone in PHONES_39:
            corpus = corpus_stats.get(phone, {})
            corpus_mean = corpus.get("mean", 0)
            corpus_std = corpus.get("std", 0)
            corpus_p95 = corpus.get("p95", 0)

            for model_name, metrics in results.items():
                if hasattr(metrics, "duration_stats") and metrics.duration_stats:
                    phone_stats = metrics.duration_stats["per_phone"].get(phone, {})
                    writer.writerow(
                        [
                            phone,
                            f"{corpus_mean:.2f}",
                            f"{corpus_std:.2f}",
                            f"{corpus_p95:.2f}",
                            model_name,
                            f"{phone_stats.get('pred_mean', 0):.2f}",
                            f"{phone_stats.get('pred_std', 0):.2f}",
                            f"{phone_stats.get('mae', 0):.2f}",
                            f"{phone_stats.get('kl_divergence', 0):.4f}",
                        ]
                    )

    logger.info(f"Duration analysis CSV saved to {csv_path}")


def plot_duration_distributions(
    results: dict, corpus_stats: dict, output_dir: Path, max_dur: int = 50
):
    """Generate per-phoneme duration distribution plots.

    Args:
        results: Dictionary of model results
        corpus_stats: Raw TIMIT corpus statistics
        output_dir: Directory for output plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping duration plots")
        return

    # Select interesting phonemes (vowels, stops, fricatives)
    phones_to_plot = ["aa", "iy", "p", "t", "s", "sil"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, phone in zip(axes.flat, phones_to_plot, strict=False):
        # Plot corpus reference if available
        corpus = corpus_stats.get(phone, {})
        if corpus:
            ax.axvline(
                corpus.get("mean", 0),
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"TIMIT (μ={corpus.get('mean', 0):.1f})",
            )

        # Plot each model's predicted distribution
        colors = ["blue", "green", "red", "orange"]
        for (model_name, metrics), color in zip(results.items(), colors, strict=False):
            if hasattr(metrics, "duration_stats") and metrics.duration_stats:
                phone_stats = metrics.duration_stats["per_phone"].get(phone, {})
                pred_mean = phone_stats.get("pred_mean", 0)
                pred_std = phone_stats.get("pred_std", 1)

                # Draw normal approximation
                x = np.linspace(max(0, pred_mean - 3 * pred_std), pred_mean + 3 * pred_std, 100)
                y = np.exp(-0.5 * ((x - pred_mean) / pred_std) ** 2) / (
                    pred_std * np.sqrt(2 * np.pi)
                )
                ax.plot(x, y, color=color, label=f"{model_name} (μ={pred_mean:.1f})")

        ax.set_title(f"/{phone}/")
        ax.set_xlabel("Duration (frames)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, max_dur)

    plt.suptitle("Duration Distributions: TIMIT Corpus vs Model Predictions")
    plt.tight_layout()

    output_path = output_dir / "duration_distributions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Duration plot saved to {output_path}")


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
    # Throughput metrics (for scaling analysis)
    throughput_utterances_per_sec: float = 0.0  # avg utterances/sec per epoch
    throughput_utterances_per_sec_std: float = 0.0  # std dev across epochs
    throughput_frames_per_sec: float = 0.0  # avg frames/sec per epoch
    throughput_frames_per_sec_std: float = 0.0  # std dev across epochs
    num_train_utterances: int = 0  # dataset size for context
    batch_size: int = 0  # batch size for context

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
            "throughput_utterances_per_sec": self.throughput_utterances_per_sec,
            "throughput_utterances_per_sec_std": self.throughput_utterances_per_sec_std,
            "throughput_frames_per_sec": self.throughput_frames_per_sec,
            "throughput_frames_per_sec_std": self.throughput_frames_per_sec_std,
            "num_train_utterances": self.num_train_utterances,
            "batch_size": self.batch_size,
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
) -> tuple[float, float, int, int]:
    """Train for one epoch.

    Args:
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only).
            Helps prevent gradient explosion from unbounded transition/duration_bias.

    Returns:
        Tuple of (average_loss, elapsed_time_seconds, num_utterances, num_frames).
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_utterances = 0
    total_frames = 0

    start_time = time.perf_counter()

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Track throughput metrics
        total_utterances += features.shape[0]
        total_frames += lengths.sum().item()

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
    return total_loss / num_batches, elapsed, total_utterances, total_frames


@torch.no_grad()
def evaluate(
    model: TIMITModel | TIMITModelPytorchCRF,
    dataloader: DataLoader,
    device: torch.device,
    backend: str = "streaming",
) -> tuple[TIMITMetrics, float]:
    """Evaluate model.

    Args:
        backend: Backend for decode (only used for TIMITModel, ignored for pytorch-crf).

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

        # pytorch-crf doesn't support backend parameter
        if is_pytorch_crf:
            result = model.decode(features, lengths)
        else:
            result = model.decode(features, lengths, backend=backend)

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
    epoch_utterances = []
    epoch_frames = []
    epoch_utt_rates = []  # utterances/sec per epoch
    epoch_frame_rates = []  # frames/sec per epoch

    for epoch in range(epochs):
        train_loss, epoch_time, num_utterances, num_frames = train_epoch(
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
        epoch_utterances.append(num_utterances)
        epoch_frames.append(num_frames)
        # Track per-epoch throughput rates
        epoch_utt_rates.append(num_utterances / epoch_time if epoch_time > 0 else 0)
        epoch_frame_rates.append(num_frames / epoch_time if epoch_time > 0 else 0)
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
            test_metrics, inference_time = evaluate(model, test_loader, device, backend=backend)

            # Calculate throughput for this epoch
            utt_per_sec = num_utterances / epoch_time if epoch_time > 0 else 0
            frames_per_sec = num_frames / epoch_time if epoch_time > 0 else 0

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"PER: {test_metrics.phone_error_rate:.4f} | "
                f"Boundary F1: {test_metrics.boundary_f1:.4f} | "
                f"Segment F1: {test_metrics.segment_f1:.4f} | "
                f"Train: {epoch_time:.1f}s ({utt_per_sec:.1f} utt/s, {frames_per_sec/1000:.1f}k fr/s) | "
                f"Infer: {inference_time:.1f}s"
            )

            if test_metrics.phone_error_rate < best_per:
                best_per = test_metrics.phone_error_rate
                # Update metrics with timing info
                test_metrics.total_training_time = total_train_time
                test_metrics.training_time_per_epoch = sum(epoch_times) / len(epoch_times)
                test_metrics.inference_time = inference_time
                # Update metrics with throughput info (mean and std across epochs)
                utt_rates = np.array(epoch_utt_rates)
                frame_rates = np.array(epoch_frame_rates)
                test_metrics.throughput_utterances_per_sec = float(np.mean(utt_rates))
                test_metrics.throughput_utterances_per_sec_std = float(np.std(utt_rates))
                test_metrics.throughput_frames_per_sec = float(np.mean(frame_rates))
                test_metrics.throughput_frames_per_sec_std = float(np.std(frame_rates))
                test_metrics.num_train_utterances = len(train_dataset)
                test_metrics.batch_size = batch_size
                best_metrics = test_metrics

    return model, best_metrics


def _print_duration_analysis(results: dict, has_pytorch_crf: bool = False):
    """Print duration distribution analysis comparing models (4-way)."""
    print("\n" + "=" * 60)
    print("DURATION ANALYSIS (frames @ 10ms)")
    print("=" * 60)
    print("\nThis shows how well each model captures phoneme duration patterns.")
    print("Lower MAE = better duration modeling. Semi-CRF should excel here.")
    print("Semi PyTorch and Semi Triton should have similar MAE (validates Triton).\n")

    # Get reference durations from semi_crf_triton (or pytorch, they should be same)
    ref_model = "semi_crf_triton"
    ref_stats = results[ref_model].duration_stats["per_phone"]

    # Select phones to display (most frequent + phonetically interesting)
    interesting_phones = ["aa", "iy", "eh", "ah", "p", "t", "k", "s", "sh", "n", "l", "sil"]
    display_phones = [p for p in interesting_phones if ref_stats[p]["ref_count"] > 50]

    # Get stats from all models
    l_stats = results["linear_crf_triton"].duration_stats["per_phone"]
    py_stats = results["semi_crf_pytorch"].duration_stats["per_phone"]
    tr_stats = results["semi_crf_triton"].duration_stats["per_phone"]
    p_stats = results["pytorch_crf"].duration_stats["per_phone"] if has_pytorch_crf else None

    if has_pytorch_crf:
        print(
            f"{'Phone':<6} {'Ref':>6} {'p-crf':>6} {'K=1':>6} {'Py':>6} {'Tr':>6} │ "
            f"{'MAE p':>6} {'MAE K1':>7} {'MAE Py':>7} {'MAE Tr':>7}"
        )
        print("-" * 90)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            p_mean = p_stats[phone]["pred_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            py_mean = py_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            p_mae = p_stats[phone]["mae"]
            l_mae = l_stats[phone]["mae"]
            py_mae = py_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            # Highlight if semi-CRF is better than linear
            best_semi_mae = min(py_mae, tr_mae)
            s_marker = "*" if best_semi_mae < p_mae and best_semi_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {p_mean:>6.1f} {l_mean:>6.1f} "
                f"{py_mean:>6.1f} {tr_mean:>6.1f} │ "
                f"{p_mae:>6.2f} {l_mae:>7.2f} {py_mae:>7.2f} {tr_mae:>6.2f}{s_marker}"
            )

        # Overall stats
        print("-" * 90)
        p_overall = results["pytorch_crf"].duration_stats["overall"]
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        py_overall = results["semi_crf_pytorch"].duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['mean_absolute_error']:>6.2f} "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{py_overall['mean_absolute_error']:>7.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['duration_correlation']:>6.3f} "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{py_overall['duration_correlation']:>7.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )
    else:
        print(
            f"{'Phone':<6} {'Ref':>6} {'K=1':>6} {'Py':>6} {'Tr':>6} │ "
            f"{'MAE K1':>7} {'MAE Py':>7} {'MAE Tr':>7}"
        )
        print("-" * 70)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            py_mean = py_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            l_mae = l_stats[phone]["mae"]
            py_mae = py_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            best_semi_mae = min(py_mae, tr_mae)
            s_marker = "*" if best_semi_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {l_mean:>6.1f} "
                f"{py_mean:>6.1f} {tr_mean:>6.1f} │ "
                f"{l_mae:>7.2f} {py_mae:>7.2f} {tr_mae:>6.2f}{s_marker}"
            )

        # Overall stats
        print("-" * 70)
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        py_overall = results["semi_crf_pytorch"].duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{py_overall['mean_absolute_error']:>7.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{py_overall['duration_correlation']:>7.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )

    print("\n* = Semi-CRF has lowest MAE for this phone")
    print("Py = Semi-CRF PyTorch (baseline), Tr = Semi-CRF Triton (optimized)")
    print("Corr = correlation between predicted and reference mean durations")


def compare_models(data_dir: Path, max_duration: int = 30, **kwargs):
    """
    Compare CRF models in a 4-way comparison validating Triton implementations.

    Models compared:
    1. pytorch-crf (optional): External linear CRF baseline
    2. K=1 Triton: Linear CRF via torch-semimarkov streaming kernel
    3. Semi-CRF PyTorch: K>1 with PyTorch streaming (reference baseline)
    4. Semi-CRF Triton: K>1 with Triton streaming kernel (optimized)

    Validates:
    - linear_crf_triton ≈ pytorch_crf (Triton K=1 matches external linear CRF)
    - semi_crf_triton ≈ semi_crf_pytorch (Triton K>1 matches reference semi-CRF)
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
    logger.info("Training LINEAR CRF (torch-semimarkov K=1, PyTorch)")
    logger.info("=" * 60)
    _, linear_metrics = train_model(
        data_dir, model_type="linear", backend="streaming", use_triton=True, **kwargs
    )
    results["linear_crf_triton"] = linear_metrics

    # 3. Semi-CRF with PyTorch streaming (reference semi-CRF baseline)
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF PYTORCH (K={max_duration}, streaming baseline)")
    logger.info("=" * 60)
    _, pytorch_metrics = train_model(
        data_dir,
        model_type="semicrf",
        max_duration=max_duration,
        backend="streaming",
        use_triton=False,  # Pure PyTorch - validates Triton correctness
        **kwargs,
    )
    results["semi_crf_pytorch"] = pytorch_metrics

    # 4. Semi-CRF with Triton streaming (optimized memory)
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF TRITON (K={max_duration}, streaming kernel)")
    logger.info("=" * 60)
    _, triton_metrics = train_model(
        data_dir,
        model_type="semicrf",
        max_duration=max_duration,
        backend="streaming",
        use_triton=True,
        **kwargs,
    )
    results["semi_crf_triton"] = triton_metrics

    # Load corpus duration statistics for comparison with raw TIMIT
    corpus_stats = load_corpus_duration_stats(data_dir)

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("4-WAY COMPARISON: Linear CRF vs Semi-CRF (baseline vs PyTorch/Triton)")
    logger.info("=" * 60)

    # Print 4-way comparison table
    _print_four_way_comparison(results, has_pytorch_crf=HAS_TORCHCRF)

    # Duration analysis (with raw TIMIT stats)
    _print_duration_analysis(results, has_pytorch_crf=HAS_TORCHCRF)

    # Print corpus comparison if stats available
    if corpus_stats:
        _print_corpus_comparison(results, corpus_stats)

    return results


def _print_corpus_comparison(results: dict, corpus_stats: dict):
    """Print comparison between model predictions and raw TIMIT corpus statistics."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH RAW TIMIT CORPUS")
    print("=" * 60)
    print("\nThis compares model predictions directly against corpus statistics")
    print("from train_segment_stats.json (computed during preprocessing).\n")

    # Select interesting phonemes
    interesting_phones = ["aa", "iy", "eh", "ah", "p", "t", "s", "sil"]
    display_phones = [p for p in interesting_phones if p in corpus_stats]

    print(f"{'Phone':<6} {'Corpus':>12} {'Semi Triton':>12} {'Diff':>10} {'Semi vs Lin':>12}")
    print(f"{'':6} {'mean±std':>12} {'mean±std':>12} {'(frames)':>10} {'improvement':>12}")
    print("-" * 65)

    semi_metrics = results["semi_crf_triton"]
    linear_metrics = results["linear_crf_triton"]

    for phone in display_phones:
        corpus = corpus_stats.get(phone, {})
        corpus_mean = corpus.get("mean", 0)
        corpus_std = corpus.get("std", 0)

        semi_stats = semi_metrics.duration_stats["per_phone"].get(phone, {})
        linear_stats = linear_metrics.duration_stats["per_phone"].get(phone, {})

        semi_mean = semi_stats.get("pred_mean", 0)
        semi_std = semi_stats.get("pred_std", 0)
        semi_mae = semi_stats.get("mae", 0)
        linear_mae = linear_stats.get("mae", 0)

        # Improvement: how much better semi-CRF is vs linear
        improvement = linear_mae - semi_mae

        print(
            f"{phone:<6} {corpus_mean:>5.1f}±{corpus_std:<4.1f} "
            f"{semi_mean:>5.1f}±{semi_std:<4.1f} "
            f"{semi_mean - corpus_mean:>+10.1f} {improvement:>+12.2f}"
        )

    print("-" * 65)
    print("Positive 'Semi vs Lin improvement' = Semi-CRF captures duration better")


def _print_four_way_comparison(results: dict, has_pytorch_crf: bool = False):
    """Print 4-way comparison table."""
    # Header
    if has_pytorch_crf:
        print(
            f"\n{'Metric':<20} {'pytorch-crf':>12} {'K=1 Triton':>12} "
            f"{'Semi PyTorch':>12} {'Semi Triton':>12} {'Δ Linear':>10} {'Δ Semi':>10}"
        )
        print("-" * 100)
    else:
        print(
            f"\n{'Metric':<20} {'K=1 Triton':>12} "
            f"{'Semi PyTorch':>12} {'Semi Triton':>12} {'Δ Semi':>10}"
        )
        print("-" * 80)

    # Get metrics
    l_metrics = results["linear_crf_triton"]
    py_metrics = results["semi_crf_pytorch"]
    tr_metrics = results["semi_crf_triton"]
    p_metrics = results.get("pytorch_crf")

    # Phone Error Rate (lower is better)
    if has_pytorch_crf:
        delta_linear = l_metrics.phone_error_rate - p_metrics.phone_error_rate
        delta_semi = tr_metrics.phone_error_rate - py_metrics.phone_error_rate
        print(
            f"{'Phone Error Rate':<20} {p_metrics.phone_error_rate:>12.4f} "
            f"{l_metrics.phone_error_rate:>12.4f} {py_metrics.phone_error_rate:>12.4f} "
            f"{tr_metrics.phone_error_rate:>12.4f} {delta_linear:>+10.4f} {delta_semi:>+10.4f}"
        )
    else:
        delta_semi = tr_metrics.phone_error_rate - py_metrics.phone_error_rate
        print(
            f"{'Phone Error Rate':<20} {l_metrics.phone_error_rate:>12.4f} "
            f"{py_metrics.phone_error_rate:>12.4f} {tr_metrics.phone_error_rate:>12.4f} "
            f"{delta_semi:>+10.4f}"
        )

    # F1 scores (higher is better)
    for metric_name, display_name in [("boundary_f1", "Boundary F1"), ("segment_f1", "Segment F1")]:
        l_val = getattr(l_metrics, metric_name)
        py_val = getattr(py_metrics, metric_name)
        tr_val = getattr(tr_metrics, metric_name)

        if has_pytorch_crf:
            p_val = getattr(p_metrics, metric_name)
            delta_linear = l_val - p_val
            delta_semi = tr_val - py_val
            print(
                f"{display_name:<20} {p_val:>12.4f} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_linear:>+10.4f} {delta_semi:>+10.4f}"
            )
        else:
            delta_semi = tr_val - py_val
            print(
                f"{display_name:<20} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_semi:>+10.4f}"
            )

    # Boundary F1 tolerances
    print("\nBoundary F1 at different tolerances:")
    for tol in [0, 1, 2]:
        l_val = l_metrics.boundary_f1_tolerances.get(tol, 0)
        py_val = py_metrics.boundary_f1_tolerances.get(tol, 0)
        tr_val = tr_metrics.boundary_f1_tolerances.get(tol, 0)

        if has_pytorch_crf:
            p_val = p_metrics.boundary_f1_tolerances.get(tol, 0)
            delta_linear = l_val - p_val
            delta_semi = tr_val - py_val
            print(
                f"  tol={tol:<2} {p_val:>12.4f} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_linear:>+10.4f} {delta_semi:>+10.4f}"
            )
        else:
            delta_semi = tr_val - py_val
            print(
                f"  tol={tol:<2} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_semi:>+10.4f}"
            )

    # Timing comparison
    print("\nTiming (lower is better):")
    l_time = l_metrics.training_time_per_epoch
    py_time = py_metrics.training_time_per_epoch
    tr_time = tr_metrics.training_time_per_epoch

    if has_pytorch_crf:
        p_time = p_metrics.training_time_per_epoch
        speedup_linear = p_time / l_time if l_time > 0 else 0
        speedup_semi = py_time / tr_time if tr_time > 0 else 0
        print(
            f"{'Train (s/epoch)':<20} {p_time:>12.2f} {l_time:>12.2f} "
            f"{py_time:>12.2f} {tr_time:>12.2f} {speedup_linear:>9.2f}x {speedup_semi:>9.2f}x"
        )
    else:
        speedup_semi = py_time / tr_time if tr_time > 0 else 0
        print(
            f"{'Train (s/epoch)':<20} {l_time:>12.2f} "
            f"{py_time:>12.2f} {tr_time:>12.2f} {speedup_semi:>9.2f}x"
        )

    l_infer = l_metrics.inference_time
    py_infer = py_metrics.inference_time
    tr_infer = tr_metrics.inference_time

    if has_pytorch_crf:
        p_infer = p_metrics.inference_time
        speedup_linear = p_infer / l_infer if l_infer > 0 else 0
        speedup_semi = py_infer / tr_infer if tr_infer > 0 else 0
        print(
            f"{'Inference (s)':<20} {p_infer:>12.2f} {l_infer:>12.2f} "
            f"{py_infer:>12.2f} {tr_infer:>12.2f} {speedup_linear:>9.2f}x {speedup_semi:>9.2f}x"
        )
    else:
        speedup_semi = py_infer / tr_infer if tr_infer > 0 else 0
        print(
            f"{'Inference (s)':<20} {l_infer:>12.2f} "
            f"{py_infer:>12.2f} {tr_infer:>12.2f} {speedup_semi:>9.2f}x"
        )

    # Validation notes
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if has_pytorch_crf:
        print("Linear CRF: K=1 Triton should match pytorch-crf accuracy (Δ Linear ≈ 0)")
    print("Semi-CRF: Triton should match PyTorch baseline accuracy (Δ Semi ≈ 0)")
    print("Timing: Triton speedup shown as multiplier (e.g., 2.0x = twice as fast)")


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
        choices=["streaming", "binary_tree_sharded", "exact", "auto"],
        default="streaming",
        help="CRF backend: streaming (Triton), binary_tree_sharded (sharded matmuls), "
        "exact (edge tensor), auto (heuristic)",
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
    compare_parser.add_argument(
        "--export-duration",
        type=Path,
        default=None,
        help="Export duration analysis to this path (creates .json and .csv files)",
    )
    compare_parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for duration distribution plots (requires matplotlib)",
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
        _model, metrics = train_model(
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
        # Print training summary with throughput
        k = 1 if args.model in ("pytorch-crf", "linear") else args.max_duration
        triton_str = "Triton" if not args.no_triton else "PyTorch"
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Model: {args.model} (K={k}, {triton_str})")
        print(f"  Backend: {args.backend}")
        print(f"  Batch size: {metrics.batch_size}")
        print(f"  Dataset: {metrics.num_train_utterances} utterances")
        print(f"  Epochs: {args.epochs}")
        print(f"  Total training time: {metrics.total_training_time:.1f}s")
        print(f"  Avg time per epoch: {metrics.training_time_per_epoch:.2f}s")
        print(
            f"  Throughput: {metrics.throughput_utterances_per_sec:.1f} ± "
            f"{metrics.throughput_utterances_per_sec_std:.1f} utt/s"
        )
        print(
            f"             {metrics.throughput_frames_per_sec/1000:.1f} ± "
            f"{metrics.throughput_frames_per_sec_std/1000:.1f}k frames/s"
        )
        print("-" * 60)
        print(f"  Best PER: {metrics.phone_error_rate:.4f}")
        print(f"  Boundary F1: {metrics.boundary_f1:.4f}")
        print(f"  Segment F1: {metrics.segment_f1:.4f}")
        print("=" * 60)
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
                "semi_crf_pytorch": results["semi_crf_pytorch"].to_dict(),
                "semi_crf_triton": results["semi_crf_triton"].to_dict(),
            }
            # Include pytorch-crf results if available
            if "pytorch_crf" in results:
                output["pytorch_crf"] = results["pytorch_crf"].to_dict()
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")

        # Export duration analysis if requested
        if args.export_duration:
            corpus_stats = load_corpus_duration_stats(args.data_dir)
            args.export_duration.parent.mkdir(parents=True, exist_ok=True)
            export_duration_analysis(results, corpus_stats, args.export_duration)

        # Generate duration plots if requested
        if args.plot_dir:
            corpus_stats = load_corpus_duration_stats(args.data_dir)
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_duration_distributions(results, corpus_stats, args.plot_dir)


if __name__ == "__main__":
    main()
