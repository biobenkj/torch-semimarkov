#!/usr/bin/env python3
"""
Exon/Intron Segmentation Benchmark using Gencode Annotations

This benchmark demonstrates where Semi-CRFs outperform linear CRFs:
- Exons and introns have dramatically different length distributions
- Exons: median ~150bp, 95th percentile ~500bp, rarely >1kb
- Introns: median ~1kb, can span >100kb
- A linear CRF cannot encode "this state tends to last N positions"
- A semi-CRF explicitly penalizes implausible durations (3bp exon, 10bp intron)

Experimental Design:
    Same encoder (BiLSTM or Mamba) with two CRF heads:
    1. Linear CRF: K=1 (standard CRF, no duration modeling)
    2. Semi-CRF: K=500 or K=1000 (explicit duration distributions)

    Both use identical transition matrices and emission projections.
    The only difference is whether duration is modeled.

Data:
    - Gencode GTF annotations (human or mouse)
    - Reference genome FASTA
    - Chromosome-based train/val/test split (e.g., train: chr1-18, val: chr19-20, test: chr21-22)

Label scheme (5 classes):
    0: intergenic
    1: 5'UTR
    2: CDS (coding exon)
    3: 3'UTR  
    4: intron

Metrics:
    1. Position-level F1 (macro and per-class)
    2. Boundary F1 (exact match and within-k tolerance)
    3. Segment-level F1 (whole segment must be correct)
    4. Duration calibration (KL divergence between predicted and true distributions)

Requirements:
    pip install pyfaidx gtfparse pandas torch lightning

Usage:
    # Preprocess data
    python gencode_exon_intron.py preprocess \
        --gtf gencode.v44.annotation.gtf.gz \
        --fasta GRCh38.primary_assembly.genome.fa \
        --output-dir data/gencode_benchmark/

    # Train and evaluate
    python gencode_exon_intron.py train \
        --data-dir data/gencode_benchmark/ \
        --model semicrf \
        --max-duration 500 \
        --epochs 50

    # Compare linear CRF vs semi-CRF
    python gencode_exon_intron.py compare \
        --data-dir data/gencode_benchmark/
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Conditional imports for preprocessing
try:
    import pyfaidx
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Label Scheme
# =============================================================================

LABEL_SCHEME = {
    "intergenic": 0,
    "5UTR": 1,
    "CDS": 2,
    "3UTR": 3,
    "intron": 4,
}
NUM_CLASSES = len(LABEL_SCHEME)
LABEL_NAMES = list(LABEL_SCHEME.keys())

# One-hot encoding for DNA (A=0, C=1, G=2, T=3, N=4)
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
DNA_DIM = 5

# Chromosome splits (human)
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 19)]
VAL_CHROMS = ["chr19", "chr20"]
TEST_CHROMS = ["chr21", "chr22"]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GeneAnnotation:
    """Annotation for a single gene."""
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    start: int  # 0-based
    end: int    # exclusive
    transcript_id: str
    exons: list[tuple[int, int]] = field(default_factory=list)  # (start, end) pairs
    cds: list[tuple[int, int]] = field(default_factory=list)
    utrs_5: list[tuple[int, int]] = field(default_factory=list)
    utrs_3: list[tuple[int, int]] = field(default_factory=list)


class SegmentAnnotation(NamedTuple):
    """A segment with label."""
    start: int
    end: int
    label: int


@dataclass
class GenomicChunk:
    """A chunk of genomic sequence with labels."""
    chrom: str
    start: int
    end: int
    sequence: str
    labels: list[int]
    segments: list[SegmentAnnotation]


# =============================================================================
# GTF Parsing and Preprocessing
# =============================================================================

def parse_gtf_line(line: str) -> dict | None:
    """Parse a single GTF line into a dictionary."""
    if line.startswith("#"):
        return None
    
    parts = line.strip().split("\t")
    if len(parts) < 9:
        return None
    
    chrom, source, feature, start, end, score, strand, frame, attributes = parts
    
    # Parse attributes
    attr_dict = {}
    for attr in attributes.split(";"):
        attr = attr.strip()
        if not attr:
            continue
        if " " in attr:
            key, value = attr.split(" ", 1)
            attr_dict[key] = value.strip('"')
    
    return {
        "chrom": chrom,
        "source": source,
        "feature": feature,
        "start": int(start) - 1,  # Convert to 0-based
        "end": int(end),          # GTF end is inclusive, we make it exclusive
        "strand": strand,
        "attributes": attr_dict,
    }


def load_gencode_annotations(gtf_path: Path, chroms: list[str] | None = None) -> dict[str, list[GeneAnnotation]]:
    """
    Load Gencode annotations from GTF file.
    
    Returns dict mapping chromosome -> list of gene annotations.
    Uses canonical transcript (longest CDS, or longest transcript if non-coding).
    """
    logger.info(f"Loading GTF from {gtf_path}")
    
    # First pass: collect all transcripts
    transcripts: dict[str, dict] = {}  # transcript_id -> info
    
    open_fn = gzip.open if str(gtf_path).endswith(".gz") else open
    mode = "rt" if str(gtf_path).endswith(".gz") else "r"
    
    with open_fn(gtf_path, mode) as f:
        for line in f:
            parsed = parse_gtf_line(line)
            if parsed is None:
                continue
            
            if chroms and parsed["chrom"] not in chroms:
                continue
            
            attrs = parsed["attributes"]
            transcript_id = attrs.get("transcript_id")
            
            if not transcript_id:
                continue
            
            if transcript_id not in transcripts:
                transcripts[transcript_id] = {
                    "gene_id": attrs.get("gene_id", ""),
                    "gene_name": attrs.get("gene_name", ""),
                    "chrom": parsed["chrom"],
                    "strand": parsed["strand"],
                    "exons": [],
                    "cds": [],
                    "start_codon": [],
                    "stop_codon": [],
                    "utrs": [],
                }
            
            t = transcripts[transcript_id]
            feature = parsed["feature"]
            coord = (parsed["start"], parsed["end"])
            
            if feature == "exon":
                t["exons"].append(coord)
            elif feature == "CDS":
                t["cds"].append(coord)
            elif feature == "UTR":
                t["utrs"].append(coord)
            elif feature == "five_prime_UTR":
                t["utrs"].append((*coord, "5"))
            elif feature == "three_prime_UTR":
                t["utrs"].append((*coord, "3"))
    
    logger.info(f"Loaded {len(transcripts)} transcripts")
    
    # Second pass: select canonical transcript per gene and build annotations
    genes_by_chrom: dict[str, list[GeneAnnotation]] = defaultdict(list)
    gene_transcripts: dict[str, list[str]] = defaultdict(list)
    
    for tid, t in transcripts.items():
        gene_transcripts[t["gene_id"]].append(tid)
    
    for gene_id, tids in gene_transcripts.items():
        # Select canonical: longest CDS, or longest transcript
        best_tid = None
        best_cds_len = -1
        best_tx_len = -1
        
        for tid in tids:
            t = transcripts[tid]
            cds_len = sum(e - s for s, e in t["cds"])
            tx_len = sum(e - s for s, e in t["exons"])
            
            if cds_len > best_cds_len or (cds_len == best_cds_len and tx_len > best_tx_len):
                best_cds_len = cds_len
                best_tx_len = tx_len
                best_tid = tid
        
        if best_tid is None:
            continue
        
        t = transcripts[best_tid]
        
        # Sort exons and CDS by position
        exons = sorted(t["exons"])
        cds = sorted(t["cds"])
        
        if not exons:
            continue
        
        # Compute UTRs from exons and CDS
        utrs_5 = []
        utrs_3 = []
        
        if cds:
            cds_start = min(s for s, e in cds)
            cds_end = max(e for s, e in cds)
            
            for ex_start, ex_end in exons:
                if ex_end <= cds_start:
                    # Entirely before CDS
                    if t["strand"] == "+":
                        utrs_5.append((ex_start, ex_end))
                    else:
                        utrs_3.append((ex_start, ex_end))
                elif ex_start >= cds_end:
                    # Entirely after CDS
                    if t["strand"] == "+":
                        utrs_3.append((ex_start, ex_end))
                    else:
                        utrs_5.append((ex_start, ex_end))
                elif ex_start < cds_start < ex_end:
                    # Partial UTR at start
                    if t["strand"] == "+":
                        utrs_5.append((ex_start, cds_start))
                    else:
                        utrs_3.append((ex_start, cds_start))
                elif ex_start < cds_end < ex_end:
                    # Partial UTR at end
                    if t["strand"] == "+":
                        utrs_3.append((cds_end, ex_end))
                    else:
                        utrs_5.append((cds_end, ex_end))
        
        gene_start = min(s for s, e in exons)
        gene_end = max(e for s, e in exons)
        
        anno = GeneAnnotation(
            gene_id=gene_id,
            gene_name=t["gene_name"],
            chrom=t["chrom"],
            strand=t["strand"],
            start=gene_start,
            end=gene_end,
            transcript_id=best_tid,
            exons=exons,
            cds=cds,
            utrs_5=utrs_5,
            utrs_3=utrs_3,
        )
        genes_by_chrom[t["chrom"]].append(anno)
    
    # Sort genes by position
    for chrom in genes_by_chrom:
        genes_by_chrom[chrom].sort(key=lambda g: g.start)
    
    total_genes = sum(len(v) for v in genes_by_chrom.values())
    logger.info(f"Built annotations for {total_genes} genes across {len(genes_by_chrom)} chromosomes")
    
    return dict(genes_by_chrom)


def build_label_track(chrom_length: int, genes: list[GeneAnnotation]) -> np.ndarray:
    """
    Build position-wise label array for a chromosome.
    
    Priority (highest to lowest): CDS > UTR > intron > intergenic
    """
    labels = np.zeros(chrom_length, dtype=np.int8)  # intergenic = 0
    
    for gene in genes:
        # First mark all exonic regions as intron (will be overwritten by exons)
        # This handles the space between exons within the gene
        for i in range(len(gene.exons) - 1):
            intron_start = gene.exons[i][1]
            intron_end = gene.exons[i + 1][0]
            labels[intron_start:intron_end] = LABEL_SCHEME["intron"]
        
        # Mark UTRs
        for start, end in gene.utrs_5:
            labels[start:end] = LABEL_SCHEME["5UTR"]
        for start, end in gene.utrs_3:
            labels[start:end] = LABEL_SCHEME["3UTR"]
        
        # Mark CDS (highest priority for exonic)
        for start, end in gene.cds:
            labels[start:end] = LABEL_SCHEME["CDS"]
    
    return labels


def extract_segments(labels: np.ndarray) -> list[SegmentAnnotation]:
    """Convert label array to list of segments."""
    if len(labels) == 0:
        return []
    
    segments = []
    current_label = labels[0]
    current_start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append(SegmentAnnotation(current_start, i, int(current_label)))
            current_label = labels[i]
            current_start = i
    
    # Final segment
    segments.append(SegmentAnnotation(current_start, len(labels), int(current_label)))
    
    return segments


def chunk_chromosome(
    chrom: str,
    sequence: str,
    labels: np.ndarray,
    chunk_size: int = 10000,
    overlap: int = 500,
    min_gene_coverage: float = 0.1,
) -> Iterator[GenomicChunk]:
    """
    Chunk a chromosome into overlapping windows.
    
    Args:
        chrom: Chromosome name
        sequence: DNA sequence
        labels: Position-wise labels
        chunk_size: Size of each chunk
        overlap: Overlap between consecutive chunks
        min_gene_coverage: Minimum fraction of non-intergenic positions to include chunk
    """
    chrom_len = len(sequence)
    stride = chunk_size - overlap
    
    for start in range(0, chrom_len, stride):
        end = min(start + chunk_size, chrom_len)
        
        chunk_labels = labels[start:end]
        chunk_seq = sequence[start:end]
        
        # Skip chunks that are mostly intergenic
        gene_coverage = np.mean(chunk_labels != LABEL_SCHEME["intergenic"])
        if gene_coverage < min_gene_coverage:
            continue
        
        segments = extract_segments(chunk_labels)
        
        yield GenomicChunk(
            chrom=chrom,
            start=start,
            end=end,
            sequence=chunk_seq,
            labels=chunk_labels.tolist(),
            segments=segments,
        )


def preprocess_gencode(
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
    chunk_size: int = 10000,
    overlap: int = 500,
):
    """
    Preprocess Gencode annotations into train/val/test chunks.
    """
    if not HAS_PYFAIDX:
        raise ImportError("pyfaidx required for preprocessing: pip install pyfaidx")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference genome
    logger.info(f"Loading reference genome from {fasta_path}")
    genome = pyfaidx.Fasta(str(fasta_path))
    
    # Load annotations
    all_chroms = TRAIN_CHROMS + VAL_CHROMS + TEST_CHROMS
    genes_by_chrom = load_gencode_annotations(gtf_path, chroms=all_chroms)
    
    # Process each split
    for split_name, chroms in [("train", TRAIN_CHROMS), ("val", VAL_CHROMS), ("test", TEST_CHROMS)]:
        logger.info(f"Processing {split_name} split: {chroms}")
        
        chunks = []
        segment_lengths = defaultdict(list)
        
        for chrom in chroms:
            if chrom not in genome.keys():
                logger.warning(f"Chromosome {chrom} not found in FASTA")
                continue
            
            if chrom not in genes_by_chrom:
                logger.warning(f"No annotations for {chrom}")
                continue
            
            sequence = str(genome[chrom]).upper()
            labels = build_label_track(len(sequence), genes_by_chrom[chrom])
            
            for chunk in chunk_chromosome(chrom, sequence, labels, chunk_size, overlap):
                chunks.append({
                    "chrom": chunk.chrom,
                    "start": chunk.start,
                    "end": chunk.end,
                    "sequence": chunk.sequence,
                    "labels": chunk.labels,
                })
                
                # Collect segment length statistics
                for seg in chunk.segments:
                    segment_lengths[seg.label].append(seg.end - seg.start)
        
        # Save chunks
        output_file = output_dir / f"{split_name}.jsonl"
        logger.info(f"Saving {len(chunks)} chunks to {output_file}")
        
        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        
        # Save segment length statistics
        stats_file = output_dir / f"{split_name}_segment_stats.json"
        stats = {}
        for label, lengths in segment_lengths.items():
            lengths = np.array(lengths)
            stats[LABEL_NAMES[label]] = {
                "count": len(lengths),
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99)),
            }
        
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Segment statistics saved to {stats_file}")
    
    logger.info("Preprocessing complete!")


# =============================================================================
# Dataset
# =============================================================================

class GencodeDataset(Dataset):
    """Dataset for Gencode exon/intron benchmark."""
    
    def __init__(self, data_path: Path, max_length: int | None = None):
        """
        Args:
            data_path: Path to JSONL file with preprocessed chunks
            max_length: Maximum sequence length (truncate longer sequences)
        """
        self.max_length = max_length
        self.chunks = []
        
        with open(data_path) as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.chunks)} chunks from {data_path}")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        chunk = self.chunks[idx]
        
        seq = chunk["sequence"]
        labels = chunk["labels"]
        
        # Truncate if needed
        if self.max_length and len(seq) > self.max_length:
            seq = seq[:self.max_length]
            labels = labels[:self.max_length]
        
        # One-hot encode sequence
        seq_encoded = torch.zeros(len(seq), DNA_DIM)
        for i, base in enumerate(seq):
            seq_encoded[i, DNA_VOCAB.get(base, 4)] = 1.0
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        length = torch.tensor(len(seq), dtype=torch.long)
        
        return {
            "sequence": seq_encoded,
            "labels": labels_tensor,
            "length": length,
        }


def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """Collate batch with padding."""
    max_len = max(b["length"].item() for b in batch)
    
    sequences = []
    labels = []
    lengths = []
    
    for b in batch:
        seq = b["sequence"]
        lab = b["labels"]
        seq_len = b["length"].item()
        
        # Pad sequence
        if seq_len < max_len:
            pad_len = max_len - seq_len
            seq = F.pad(seq, (0, 0, 0, pad_len))  # Pad time dimension
            lab = F.pad(lab, (0, pad_len), value=-100)  # -100 for ignore_index
        
        sequences.append(seq)
        labels.append(lab)
        lengths.append(b["length"])
    
    return {
        "sequence": torch.stack(sequences),
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths),
    }


# =============================================================================
# Models
# =============================================================================

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for DNA sequences."""
    
    def __init__(
        self,
        input_dim: int = DNA_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            hidden: (batch, T, hidden_dim)
        """
        output, _ = self.lstm(x)
        return output


class ExonIntronModel(nn.Module):
    """
    Combined encoder + CRF head for exon/intron segmentation.
    
    Supports both linear CRF (K=1) and semi-CRF (K>1) for comparison.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        max_duration: int = 1,  # K=1 for linear CRF
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
    
    def forward(self, sequence: Tensor, lengths: Tensor) -> dict:
        """
        Args:
            sequence: (batch, T, DNA_DIM)
            lengths: (batch,)
        Returns:
            dict with 'partition' and 'cum_scores'
        """
        hidden = self.encoder(sequence)
        return self.crf(hidden, lengths)
    
    def compute_loss(self, sequence: Tensor, lengths: Tensor, labels: Tensor) -> Tensor:
        """Compute NLL loss."""
        hidden = self.encoder(sequence)
        return self.crf.compute_loss(hidden, lengths, labels)
    
    def decode(self, sequence: Tensor, lengths: Tensor):
        """Viterbi decoding."""
        hidden = self.encoder(sequence)
        return self.crf.decode_with_traceback(hidden, lengths)


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class SegmentMetrics:
    """Metrics for segment-based evaluation."""
    position_f1: dict[str, float]
    position_f1_macro: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_f1_tolerance: dict[int, float]  # tolerance -> F1
    segment_precision: float
    segment_recall: float
    segment_f1: float
    duration_kl: dict[str, float]  # per-class KL divergence

    def to_dict(self) -> dict:
        """Convert metrics to JSON-serializable dict."""
        return {
            "position_f1": self.position_f1,
            "position_f1_macro": self.position_f1_macro,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "boundary_f1_tolerance": {str(k): v for k, v in self.boundary_f1_tolerance.items()},
            "segment_precision": self.segment_precision,
            "segment_recall": self.segment_recall,
            "segment_f1": self.segment_f1,
            "duration_kl": self.duration_kl,
        }


def compute_position_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Compute position-level F1 scores."""
    # Flatten all predictions and targets
    all_preds = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    
    # Mask out padding
    mask = all_targets != -100
    all_preds = all_preds[mask]
    all_targets = all_targets[mask]
    
    # Per-class F1
    f1_scores = {}
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[LABEL_NAMES[c]] = f1
    
    f1_scores["macro"] = np.mean(list(f1_scores.values()))
    
    return f1_scores


def extract_boundaries(labels: np.ndarray) -> set[int]:
    """Extract boundary positions from label sequence."""
    boundaries = set()
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    tolerances: list[int] = [0, 1, 2, 5, 10],
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    results = {"tolerance_" + str(t): {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}
    
    for pred, target in zip(predictions, targets):
        # Mask out padding
        mask = target != -100
        pred = pred[mask]
        target = target[mask]
        
        pred_bounds = extract_boundaries(pred)
        true_bounds = extract_boundaries(target)
        
        for tol in tolerances:
            key = f"tolerance_{tol}"
            
            # For each predicted boundary, check if there's a true boundary within tolerance
            matched_true = set()
            for pb in pred_bounds:
                for tb in true_bounds:
                    if abs(pb - tb) <= tol and tb not in matched_true:
                        results[key]["tp"] += 1
                        matched_true.add(tb)
                        break
                else:
                    results[key]["fp"] += 1
            
            results[key]["fn"] += len(true_bounds) - len(matched_true)
    
    metrics = {}
    for tol in tolerances:
        key = f"tolerance_{tol}"
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
            metrics["boundary_f1"] = f1
    
    return metrics


def compute_segment_metrics(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
) -> dict[str, float]:
    """
    Compute segment-level metrics.
    
    A predicted segment is correct if it exactly matches a true segment
    (same start, end, and label).
    """
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


def compute_duration_calibration(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
    num_classes: int = NUM_CLASSES,
    max_duration: int = 1000,
) -> dict[str, float]:
    """
    Compute KL divergence between predicted and true duration distributions.
    
    Lower is better (predicted durations match true durations).
    """
    pred_durations = {c: [] for c in range(num_classes)}
    true_durations = {c: [] for c in range(num_classes)}
    
    for pred_segs in pred_segments:
        for seg in pred_segs:
            dur = min(seg.end - seg.start, max_duration)
            pred_durations[seg.label].append(dur)
    
    for true_segs in true_segments:
        for seg in true_segs:
            dur = min(seg.end - seg.start, max_duration)
            true_durations[seg.label].append(dur)
    
    kl_divergences = {}
    
    for c in range(num_classes):
        if len(pred_durations[c]) == 0 or len(true_durations[c]) == 0:
            kl_divergences[LABEL_NAMES[c]] = float("nan")
            continue
        
        # Build histograms
        bins = np.arange(0, max_duration + 2)
        pred_hist, _ = np.histogram(pred_durations[c], bins=bins, density=True)
        true_hist, _ = np.histogram(true_durations[c], bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred_hist = pred_hist + eps
        true_hist = true_hist + eps
        
        # Normalize
        pred_hist = pred_hist / pred_hist.sum()
        true_hist = true_hist / true_hist.sum()
        
        # KL divergence: sum(true * log(true / pred))
        kl = np.sum(true_hist * np.log(true_hist / pred_hist))
        kl_divergences[LABEL_NAMES[c]] = float(kl)
    
    return kl_divergences


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: ExonIntronModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        sequence = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(sequence, lengths, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: ExonIntronModel,
    dataloader: DataLoader,
    device: torch.device,
) -> SegmentMetrics:
    """Evaluate model on a dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_pred_segments = []
    all_true_segments = []
    
    for batch in dataloader:
        sequence = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        # Decode
        result = model.decode(sequence, lengths)
        
        for i in range(len(lengths)):
            seq_len = lengths[i].item()
            
            # Get predicted labels from segments
            # NOTE: torch_semimarkov.Segment uses INCLUSIVE end (end=5 means position 5 included)
            # Convert to exclusive for numpy slicing: [start:end+1]
            pred_labels = np.zeros(seq_len, dtype=np.int64)
            for seg in result.segments[i]:
                pred_labels[seg.start:seg.end+1] = seg.label
            
            true_labels = labels[i, :seq_len].cpu().numpy()
            
            all_predictions.append(pred_labels)
            all_targets.append(true_labels)
            
            # Convert to SegmentAnnotation
            pred_segs = [SegmentAnnotation(s.start, s.end+1, s.label) for s in result.segments[i]]
            true_segs = extract_segments(true_labels)
            
            all_pred_segments.append(pred_segs)
            all_true_segments.append(true_segs)
    
    # Compute all metrics
    position_metrics = compute_position_metrics(all_predictions, all_targets)
    boundary_metrics = compute_boundary_metrics(all_predictions, all_targets)
    segment_metrics = compute_segment_metrics(all_pred_segments, all_true_segments)
    duration_kl = compute_duration_calibration(all_pred_segments, all_true_segments)
    
    return SegmentMetrics(
        position_f1={k: v for k, v in position_metrics.items() if k != "macro"},
        position_f1_macro=position_metrics["macro"],
        boundary_precision=boundary_metrics["boundary_precision"],
        boundary_recall=boundary_metrics["boundary_recall"],
        boundary_f1=boundary_metrics["boundary_f1"],
        boundary_f1_tolerance={
            int(k.split("tol")[1]): v 
            for k, v in boundary_metrics.items() 
            if k.startswith("boundary_f1_tol")
        },
        segment_precision=segment_metrics["segment_precision"],
        segment_recall=segment_metrics["segment_recall"],
        segment_f1=segment_metrics["segment_f1"],
        duration_kl=duration_kl,
    )


def train_model(
    data_dir: Path,
    model_type: Literal["linear", "semicrf"] = "semicrf",
    max_duration: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[ExonIntronModel, dict]:
    """
    Train a model and return it with metrics.
    """
    device = torch.device(device)
    
    # Load data
    train_dataset = GencodeDataset(data_dir / "train.jsonl")
    val_dataset = GencodeDataset(data_dir / "val.jsonl")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Build model
    encoder = BiLSTMEncoder(hidden_dim=hidden_dim, num_layers=num_layers)
    
    k = 1 if model_type == "linear" else max_duration
    model = ExonIntronModel(
        encoder=encoder,
        max_duration=k,
        hidden_dim=hidden_dim,
    ).to(device)
    
    logger.info(f"Model: {model_type}, K={k}, params={sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0
    best_metrics = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, val_loader, device)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"Val F1 (pos): {val_metrics.position_f1_macro:.4f} | "
                f"Val F1 (boundary): {val_metrics.boundary_f1:.4f} | "
                f"Val F1 (segment): {val_metrics.segment_f1:.4f}"
            )
            
            if val_metrics.boundary_f1 > best_val_f1:
                best_val_f1 = val_metrics.boundary_f1
                best_metrics = val_metrics
    
    return model, best_metrics


def compare_models(data_dir: Path, max_duration: int = 500, **kwargs):
    """
    Compare linear CRF vs semi-CRF on the benchmark.
    """
    results = {}
    
    # Train linear CRF
    logger.info("=" * 60)
    logger.info("Training LINEAR CRF (K=1)")
    logger.info("=" * 60)
    _, linear_metrics = train_model(data_dir, model_type="linear", **kwargs)
    results["linear_crf"] = linear_metrics
    
    # Train semi-CRF
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF (K={max_duration})")
    logger.info("=" * 60)
    _, semicrf_metrics = train_model(data_dir, model_type="semicrf", max_duration=max_duration, **kwargs)
    results["semi_crf"] = semicrf_metrics
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Linear CRF vs Semi-CRF")
    logger.info("=" * 60)
    
    print(f"\n{'Metric':<30} {'Linear CRF':>15} {'Semi-CRF':>15} {'Δ':>10}")
    print("-" * 70)
    
    for metric_name in ["position_f1_macro", "boundary_f1", "segment_f1"]:
        linear_val = getattr(results["linear_crf"], metric_name)
        semi_val = getattr(results["semi_crf"], metric_name)
        delta = semi_val - linear_val
        print(f"{metric_name:<30} {linear_val:>15.4f} {semi_val:>15.4f} {delta:>+10.4f}")
    
    # Boundary F1 at different tolerances
    print("\nBoundary F1 at different tolerances:")
    for tol in [0, 1, 2, 5, 10]:
        linear_val = results["linear_crf"].boundary_f1_tolerance.get(tol, 0)
        semi_val = results["semi_crf"].boundary_f1_tolerance.get(tol, 0)
        delta = semi_val - linear_val
        print(f"  tol={tol:<3} {linear_val:>15.4f} {semi_val:>15.4f} {delta:>+10.4f}")
    
    # Duration calibration
    print("\nDuration KL divergence (lower is better):")
    for label in LABEL_NAMES:
        linear_kl = results["linear_crf"].duration_kl.get(label, float("nan"))
        semi_kl = results["semi_crf"].duration_kl.get(label, float("nan"))
        print(f"  {label:<15} {linear_kl:>15.4f} {semi_kl:>15.4f}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess Gencode annotations")
    preprocess_parser.add_argument("--gtf", type=Path, required=True, help="Gencode GTF file")
    preprocess_parser.add_argument("--fasta", type=Path, required=True, help="Reference genome FASTA")
    preprocess_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    preprocess_parser.add_argument("--chunk-size", type=int, default=10000)
    preprocess_parser.add_argument("--overlap", type=int, default=500)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--model", choices=["linear", "semicrf"], default="semicrf")
    train_parser.add_argument("--max-duration", type=int, default=500)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare linear CRF vs semi-CRF")
    compare_parser.add_argument("--data-dir", type=Path, required=True)
    compare_parser.add_argument("--max-duration", type=int, default=500)
    compare_parser.add_argument("--hidden-dim", type=int, default=256)
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--output-json", type=Path, default=None,
                               help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_gencode(
            args.gtf, args.fasta, args.output_dir,
            chunk_size=args.chunk_size, overlap=args.overlap
        )
    elif args.command == "train":
        train_model(
            args.data_dir,
            model_type=args.model,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.command == "compare":
        results = compare_models(
            args.data_dir,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if args.output_json:
            from datetime import datetime
            output = {
                "task": "gencode",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_duration": args.max_duration,
                    "hidden_dim": args.hidden_dim,
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
