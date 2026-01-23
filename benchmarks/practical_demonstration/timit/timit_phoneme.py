#!/usr/bin/env python3
"""
TIMIT Phoneme Segmentation Benchmark

This is the classic benchmark for demonstrating Semi-CRF advantages over linear CRFs.
TIMIT has been used since the original Semi-CRF paper (Sarawagi & Cohen, 2004) and
provides a well-studied setting with published baselines.

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

Historical context:
    - Sarawagi & Cohen (2004): Semi-CRF improved ~1-2% over linear CRF
    - Modern encoders (BiLSTM, Transformer) have pushed overall PER down
    - But the relative advantage of duration modeling should persist

Requirements:
    pip install torchaudio librosa soundfile

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

    # Train and evaluate
    python timit_phoneme.py train \
        --data-dir data/timit_benchmark/ \
        --model semicrf \
        --max-duration 30

    # Compare linear CRF vs semi-CRF
    python timit_phoneme.py compare \
        --data-dir data/timit_benchmark/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, NamedTuple

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
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# TIMIT Phone Set and Mappings
# =============================================================================

# Original 61 TIMIT phonemes
TIMIT_61_PHONES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv',
    'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx',
    'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl',
    'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
]

# Standard 39-phone folding (Lee & Hon, 1989; Kaldi/ESPnet convention)
# Maps 61 phones to 39 classes
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONE_61_TO_39 = {
    # Vowels
    'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ah': 'ah',
    'uw': 'uw', 'uh': 'uh', 'aa': 'aa', 'ey': 'ey',
    'ay': 'ay', 'oy': 'oy', 'aw': 'aw', 'ow': 'ow', 'er': 'er',
    'ao': 'aa',  # ao folds to aa (Kaldi convention)

    # Vowel reductions
    'ax': 'ah', 'ix': 'ih', 'axr': 'er', 'ax-h': 'ah', 'ux': 'uw',

    # Semivowels
    'l': 'l', 'r': 'r', 'w': 'w', 'y': 'y',
    'el': 'l', 'hh': 'hh', 'hv': 'hh',

    # Nasals
    'm': 'm', 'n': 'n', 'ng': 'ng',
    'em': 'm', 'en': 'n', 'eng': 'ng', 'nx': 'n',

    # Fricatives
    'f': 'f', 'th': 'th', 's': 's', 'sh': 'sh',
    'v': 'v', 'dh': 'dh', 'z': 'z',
    'zh': 'sh',  # zh folds to sh (Kaldi convention)

    # Affricates
    'ch': 'ch', 'jh': 'jh',

    # Stops
    'p': 'p', 't': 't', 'k': 'k', 'b': 'b', 'd': 'd', 'g': 'g',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil',
    'bcl': 'sil', 'dcl': 'sil', 'gcl': 'sil',
    'dx': 'dx', 'q': 'sil',

    # Silence
    'pau': 'sil', 'epi': 'sil', 'h#': 'sil',
}

# Standard 39-phone set (Kaldi/ESPnet convention, Lee & Hon 1989)
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONES_39 = [
    'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx',
    'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
    'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'
]

PHONE_TO_IDX = {p: i for i, p in enumerate(PHONES_39)}
NUM_PHONES = len(PHONES_39)

# Typical phone durations (in frames at 10ms)
# Useful for understanding expected semi-CRF behavior
TYPICAL_DURATIONS = {
    'sil': (5, 50),      # Silence: variable
    'aa': (5, 15),       # Vowels: longer
    'iy': (5, 15),
    'p': (2, 8),         # Stops: short
    't': (2, 8),
    'k': (2, 8),
    's': (4, 15),        # Fricatives: medium
    'sh': (4, 15),
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
                    phone_39 = PHONE_61_TO_39.get(phone_61, 'sil')
                    label_idx = PHONE_TO_IDX.get(phone_39, PHONE_TO_IDX['sil'])
                    
                    phone_segments.append(PhoneSegment(
                        start_sample=start,
                        end_sample=end,
                        phone_61=phone_61,
                        phone_39=phone_39,
                        label_idx=label_idx,
                    ))
                
                utterance_id = f"{dialect_region}_{speaker_id}_{phn_path.stem}"
                
                utterances.append(Utterance(
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    dialect_region=dialect_region,
                    wav_path=wav_path,
                    phones=phone_segments,
                ))
    
    logger.info(f"Loaded {len(utterances)} utterances from {split} split")
    return utterances


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mfcc_features(
    audio_path: Path,
    n_mfcc: int = 13,
    n_fft: int = 400,        # 25ms at 16kHz
    hop_length: int = 160,   # 10ms at 16kHz
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
        y=y, sr=sr,
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
        y=y, sr=sr,
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
                labels, segments = align_phones_to_frames(
                    utt.phones, num_frames, hop_length
                )
                
                # Collect segment statistics
                for seg in segments:
                    segment_lengths[seg.label].append(seg.end - seg.start)
                
                processed.append({
                    "utterance_id": utt.utterance_id,
                    "speaker_id": utt.speaker_id,
                    "features": features.tolist(),
                    "labels": labels.tolist(),
                    "segments": [(s.start, s.end, s.label) for s in segments],
                })
                
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
        json.dump({
            "phones_39": PHONES_39,
            "phone_to_idx": PHONE_TO_IDX,
            "phone_61_to_39": PHONE_61_TO_39,
        }, f, indent=2)
    
    logger.info("Preprocessing complete!")


# =============================================================================
# Dataset
# =============================================================================

class TIMITDataset(Dataset):
    """Dataset for TIMIT phoneme segmentation."""
    
    def __init__(self, data_path: Path, max_length: int | None = None):
        self.max_length = max_length
        self.utterances = []
        
        with open(data_path) as f:
            for line in f:
                self.utterances.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.utterances)} utterances from {data_path}")
    
    def __len__(self) -> int:
        return len(self.utterances)
    
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        utt = self.utterances[idx]
        
        features = np.array(utt["features"], dtype=np.float32)
        labels = np.array(utt["labels"], dtype=np.int64)
        
        # Truncate if needed
        if self.max_length and len(features) > self.max_length:
            features = features[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "length": torch.tensor(len(features), dtype=torch.long),
            "utterance_id": utt["utterance_id"],
        }


def collate_timit(batch: list[dict]) -> dict[str, Tensor]:
    """Collate TIMIT batch with padding."""
    max_len = max(b["length"].item() for b in batch)
    feature_dim = batch[0]["features"].shape[-1]
    
    features = []
    labels = []
    lengths = []
    utterance_ids = []
    
    for b in batch:
        feat = b["features"]
        lab = b["labels"]
        seq_len = b["length"].item()
        
        # Pad
        if seq_len < max_len:
            pad_len = max_len - seq_len
            feat = F.pad(feat, (0, 0, 0, pad_len))
            lab = F.pad(lab, (0, pad_len), value=-100)
        
        features.append(feat)
        labels.append(lab)
        lengths.append(b["length"])
        utterance_ids.append(b["utterance_id"])
    
    return {
        "features": torch.stack(features),
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths),
        "utterance_ids": utterance_ids,
    }


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
    
    def compute_loss(self, features: Tensor, lengths: Tensor, labels: Tensor) -> Tensor:
        hidden = self.encoder(features)
        return self.crf.compute_loss(hidden, lengths, labels)
    
    def decode(self, features: Tensor, lengths: Tensor):
        hidden = self.encoder(features)
        return self.crf.decode_with_traceback(hidden, lengths)


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
    
    for pred, ref in zip(predictions, references):
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
        if labels[i] != labels[i-1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_metrics(
    predictions: list[list[int]],
    references: list[list[int]],
    tolerances: list[int] = [0, 1, 2],
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    results = {f"tol_{t}": {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}
    
    for pred, ref in zip(predictions, references):
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
    
    for pred_segs, true_segs in zip(pred_segments, true_segments):
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

    def to_dict(self) -> dict:
        """Convert metrics to JSON-serializable dict."""
        return {
            "phone_error_rate": self.phone_error_rate,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "boundary_f1_tolerances": {str(k): v for k, v in self.boundary_f1_tolerances.items()},
            "segment_precision": self.segment_precision,
            "segment_recall": self.segment_recall,
            "segment_f1": self.segment_f1,
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: TIMITModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(features, lengths, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: TIMITModel,
    dataloader: DataLoader,
    device: torch.device,
) -> TIMITMetrics:
    """Evaluate model."""
    model.eval()
    
    all_predictions = []
    all_references = []
    all_pred_segments = []
    all_true_segments = []
    
    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        result = model.decode(features, lengths)
        
        for i in range(len(lengths)):
            seq_len = lengths[i].item()
            
            # Get predicted labels
            # NOTE: torch_semimarkov.Segment uses INCLUSIVE end (end=5 means position 5 included)
            # Convert to exclusive for iteration: range(start, end+1)
            pred_labels = [0] * seq_len
            for seg in result.segments[i]:
                for j in range(seg.start, min(seg.end + 1, seq_len)):
                    pred_labels[j] = seg.label
            
            ref_labels = labels[i, :seq_len].cpu().tolist()
            
            all_predictions.append(pred_labels)
            all_references.append(ref_labels)
            
            pred_segs = [SegmentAnnotation(s.start, s.end + 1, s.label) for s in result.segments[i]]
            true_segs = labels_to_segments(ref_labels)
            
            all_pred_segments.append(pred_segs)
            all_true_segments.append(true_segs)
    
    per = compute_phone_error_rate(all_predictions, all_references)
    boundary_metrics = compute_boundary_metrics(all_predictions, all_references)
    segment_metrics = compute_segment_metrics(all_pred_segments, all_true_segments)
    
    return TIMITMetrics(
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
    )


def train_model(
    data_dir: Path,
    model_type: Literal["linear", "semicrf"] = "semicrf",
    max_duration: int = 30,
    hidden_dim: int = 256,
    num_layers: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[TIMITModel, TIMITMetrics]:
    """Train a model and return it with metrics."""
    device = torch.device(device)
    
    # Load data
    train_dataset = TIMITDataset(data_dir / "train.jsonl")
    test_dataset = TIMITDataset(data_dir / "test.jsonl")
    
    # Determine feature dimension from first sample
    sample = train_dataset[0]
    input_dim = sample["features"].shape[-1]
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_timit
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_timit
    )
    
    # Build model
    encoder = BiLSTMEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    
    k = 1 if model_type == "linear" else max_duration
    model = TIMITModel(
        encoder=encoder,
        max_duration=k,
        hidden_dim=hidden_dim,
    ).to(device)
    
    logger.info(f"Model: {model_type}, K={k}, params={sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_per = float("inf")
    best_metrics = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_metrics = evaluate(model, test_loader, device)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"PER: {test_metrics.phone_error_rate:.4f} | "
                f"Boundary F1: {test_metrics.boundary_f1:.4f} | "
                f"Segment F1: {test_metrics.segment_f1:.4f}"
            )
            
            if test_metrics.phone_error_rate < best_per:
                best_per = test_metrics.phone_error_rate
                best_metrics = test_metrics
    
    return model, best_metrics


def compare_models(data_dir: Path, max_duration: int = 30, **kwargs):
    """Compare linear CRF vs semi-CRF."""
    results = {}
    
    logger.info("=" * 60)
    logger.info("Training LINEAR CRF (K=1)")
    logger.info("=" * 60)
    _, linear_metrics = train_model(data_dir, model_type="linear", **kwargs)
    results["linear_crf"] = linear_metrics
    
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF (K={max_duration})")
    logger.info("=" * 60)
    _, semicrf_metrics = train_model(data_dir, model_type="semicrf", max_duration=max_duration, **kwargs)
    results["semi_crf"] = semicrf_metrics
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Linear CRF vs Semi-CRF")
    logger.info("=" * 60)
    
    print(f"\n{'Metric':<25} {'Linear CRF':>15} {'Semi-CRF':>15} {'Δ':>12}")
    print("-" * 67)
    
    # PER (lower is better)
    l_per = results["linear_crf"].phone_error_rate
    s_per = results["semi_crf"].phone_error_rate
    print(f"{'Phone Error Rate':<25} {l_per:>15.4f} {s_per:>15.4f} {s_per - l_per:>+12.4f}")
    
    # F1 scores (higher is better)
    for metric in ["boundary_f1", "segment_f1"]:
        l_val = getattr(results["linear_crf"], metric)
        s_val = getattr(results["semi_crf"], metric)
        print(f"{metric:<25} {l_val:>15.4f} {s_val:>15.4f} {s_val - l_val:>+12.4f}")
    
    print("\nBoundary F1 at different tolerances:")
    for tol in [0, 1, 2]:
        l_val = results["linear_crf"].boundary_f1_tolerances.get(tol, 0)
        s_val = results["semi_crf"].boundary_f1_tolerances.get(tol, 0)
        print(f"  tol={tol:<2} {l_val:>15.4f} {s_val:>15.4f} {s_val - l_val:>+12.4f}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess TIMIT")
    preprocess_parser.add_argument("--timit-dir", type=Path, required=True, help="TIMIT root directory")
    preprocess_parser.add_argument("--output-dir", type=Path, required=True)
    preprocess_parser.add_argument("--feature-type", choices=["mfcc", "mel"], default="mfcc")
    preprocess_parser.add_argument("--n-mfcc", type=int, default=13)
    preprocess_parser.add_argument("--n-mels", type=int, default=80)
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--model", choices=["linear", "semicrf"], default="semicrf")
    train_parser.add_argument("--max-duration", type=int, default=30)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=3)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    
    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare linear CRF vs semi-CRF")
    compare_parser.add_argument("--data-dir", type=Path, required=True)
    compare_parser.add_argument("--max-duration", type=int, default=30)
    compare_parser.add_argument("--hidden-dim", type=int, default=256)
    compare_parser.add_argument("--num-layers", type=int, default=3)
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--output-json", type=Path, default=None,
                               help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_timit(
            args.timit_dir, args.output_dir,
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
        )
    elif args.command == "compare":
        results = compare_models(
            args.data_dir,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
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
                "linear_crf": results["linear_crf"].to_dict(),
                "semi_crf": results["semi_crf"].to_dict(),
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
