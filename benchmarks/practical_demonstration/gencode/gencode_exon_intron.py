#!/usr/bin/env python3
"""
Exon/Intron Segmentation Benchmark using Gencode Annotations

This benchmark demonstrates where Semi-CRFs outperform linear CRFs:
- Exons and introns have dramatically different length distributions
- First exons ~100bp, internal exons ~150bp, last exons ~200bp
- First introns ~2kb (longest), internal ~1kb, last ~500bp
- Start/stop codons are exactly 3bp - perfect for duration modeling
- A linear CRF cannot encode "this state tends to last N positions"
- A semi-CRF explicitly penalizes implausible durations

Experimental Design:
    Same encoder (BiLSTM or Mamba) with three CRF heads:
    1. pytorch-crf: External linear CRF baseline (optional, if installed)
    2. Linear CRF: K=1 (torch-semimarkov, no duration modeling)
    3. Semi-CRF: K=500 (explicit duration distributions via Triton kernel)

Data:
    - Gencode GTF annotations (human or mouse)
    - Reference genome FASTA
    - Chromosome-based train/val/test split (train: chr1-18, val: chr19-20, test: chr21-22)

Label scheme (11 classes with position-based exon/intron labels):
    0: intergenic
    1: first_exon      (~100bp, characteristically short)
    2: internal_exon   (~150bp, standard exon length)
    3: last_exon       (~200bp, often longer)
    4: first_intron    (~2kb, typically longest)
    5: internal_intron (~1kb, standard intron length)
    6: last_intron     (~500bp, shorter)
    7: 5UTR            (~150bp)
    8: 3UTR            (~500bp)
    9: start_codon     (exactly 3bp - great for semi-CRF)
    10: stop_codon     (exactly 3bp - great for semi-CRF)

Metrics:
    1. Position-level F1 (macro and per-class)
    2. Boundary F1 (exact match and within-k tolerance)
    3. Segment-level F1 (whole segment must be correct)
    4. Duration calibration (KL divergence between predicted and true distributions)

Requirements:
    pip install pyfaidx torch
    # Optional:
    pip install pytorch-crf  # For baseline comparison
    pip install mamba-ssm    # For Mamba encoder (GPU only)

Usage:
    # Preprocess data
    python gencode_exon_intron.py preprocess \\
        --gtf gencode.v44.annotation.gtf.gz \\
        --fasta GRCh38.primary_assembly.genome.fa \\
        --output-dir data/gencode_benchmark/

    # Train with BiLSTM + semi-CRF (default)
    python gencode_exon_intron.py train \\
        --data-dir data/gencode_benchmark/ \\
        --model semicrf --encoder bilstm

    # Train with Mamba + semi-CRF
    python gencode_exon_intron.py train \\
        --data-dir data/gencode_benchmark/ \\
        --model semicrf --encoder mamba

    # Compare all CRF types
    python gencode_exon_intron.py compare \\
        --data-dir data/gencode_benchmark/ \\
        --encoder bilstm

    # Development on CPU (no mamba-ssm required)
    python gencode_exon_intron.py train \\
        --encoder mamba_stub --device cpu
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Conditional imports for preprocessing
try:
    import pyfaidx

    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

try:
    import importlib.util

    HAS_PANDAS = importlib.util.find_spec("pandas") is not None
except ImportError:
    HAS_PANDAS = False

# Mamba SSM encoder (optional)
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None  # type: ignore

# pytorch-crf baseline (optional)
try:
    from torchcrf import CRF as TorchCRF

    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False
    TorchCRF = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Diagnostics
# =============================================================================


def print_diagnostics(batch_size: int = 32, seq_len: int = 10000, max_duration: int = 500):
    """Print system/library diagnostics for debugging OOM issues."""
    logger.info("=" * 60)
    logger.info("DIAGNOSTICS")
    logger.info("=" * 60)

    # Triton availability
    try:
        import triton

        logger.info(f"Triton: v{triton.__version__}")
    except ImportError:
        logger.warning("Triton: NOT INSTALLED (will use PyTorch streaming fallback)")

    # Mamba availability
    if HAS_MAMBA:
        logger.info("Mamba SSM: available")
    else:
        logger.info("Mamba SSM: NOT INSTALLED (use --encoder mamba_stub for CPU dev)")

    # pytorch-crf availability
    if HAS_TORCHCRF:
        logger.info("pytorch-crf: available")
    else:
        logger.info("pytorch-crf: NOT INSTALLED (--model pytorch-crf unavailable)")

    # CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1e9
            logger.info(f"GPU {i}: {props.name}, {total_gb:.1f}GB total")
    else:
        logger.info("CUDA: not available")

    # Memory estimates
    C = NUM_CLASSES
    exact_gb = batch_size * seq_len * max_duration * C * C * 4 / 1e9
    streaming_gb = batch_size * seq_len * C * 4 / 1e9  # cumsum tensor dominates

    logger.info(f"Memory estimate (B={batch_size}, T={seq_len}, K={max_duration}, C={C}):")
    logger.info(f"  Exact backend:     {exact_gb:.1f}GB (edge tensor)")
    logger.info(f"  Streaming backend: {streaming_gb:.2f}GB (cumsum tensor)")
    logger.info("=" * 60)


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed process group.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)

    Environment variables (optional):
        MASTER_ADDR: Coordinator address (default: localhost)
        MASTER_PORT: Coordinator port (default: 12355)
        NCCL_SOCKET_IFNAME: Network interface for NCCL (e.g., "ib0", "eth0")
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # For InfiniBand over Ethernet (IPoIB), NCCL uses TCP sockets automatically
    # If needed, set NCCL_SOCKET_IFNAME to specify the interface:
    #   export NCCL_SOCKET_IFNAME=ib0  (or your IPoIB interface name)

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


# =============================================================================
# Constants and Label Scheme
# =============================================================================

LABEL_SCHEME = {
    "intergenic": 0,
    "first_exon": 1,
    "internal_exon": 2,
    "last_exon": 3,
    "first_intron": 4,
    "internal_intron": 5,
    "last_intron": 6,
    "5UTR": 7,
    "3UTR": 8,
    "start_codon": 9,
    "stop_codon": 10,
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
    end: int  # exclusive
    transcript_id: str
    exons: list[tuple[int, int]] = field(default_factory=list)  # (start, end) pairs
    cds: list[tuple[int, int]] = field(default_factory=list)
    utrs_5: list[tuple[int, int]] = field(default_factory=list)
    utrs_3: list[tuple[int, int]] = field(default_factory=list)
    start_codons: list[tuple[int, int]] = field(default_factory=list)
    stop_codons: list[tuple[int, int]] = field(default_factory=list)


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
        "end": int(end),  # GTF end is inclusive, we make it exclusive
        "strand": strand,
        "attributes": attr_dict,
    }


def load_gencode_annotations(
    gtf_path: Path, chroms: list[str] | None = None
) -> dict[str, list[GeneAnnotation]]:
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
                    "start_codons": [],
                    "stop_codons": [],
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
            elif feature == "start_codon":
                t["start_codons"].append(coord)
            elif feature == "stop_codon":
                t["stop_codons"].append(coord)

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

        # Get start and stop codons
        start_codons = sorted(t.get("start_codons", []))
        stop_codons = sorted(t.get("stop_codons", []))

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
            start_codons=start_codons,
            stop_codons=stop_codons,
        )
        genes_by_chrom[t["chrom"]].append(anno)

    # Sort genes by position
    for chrom in genes_by_chrom:
        genes_by_chrom[chrom].sort(key=lambda g: g.start)

    total_genes = sum(len(v) for v in genes_by_chrom.values())
    logger.info(
        f"Built annotations for {total_genes} genes across {len(genes_by_chrom)} chromosomes"
    )

    return dict(genes_by_chrom)


def build_label_track(chrom_length: int, genes: list[GeneAnnotation]) -> np.ndarray:
    """
    Build position-wise label array for a chromosome.

    Uses position-based exon/intron labels to highlight duration differences:
    - first_exon, internal_exon, last_exon (based on CDS position)
    - first_intron, internal_intron, last_intron
    - 5UTR, 3UTR
    - start_codon, stop_codon (exactly 3bp, highest priority)

    Priority (highest to lowest):
        start_codon = stop_codon > exons > UTR > intron > intergenic
    """
    labels = np.zeros(chrom_length, dtype=np.int8)  # intergenic = 0

    for gene in genes:
        sorted_exons = sorted(gene.exons)
        num_exons = len(sorted_exons)

        # Mark introns with position-based labels
        for i in range(num_exons - 1):
            intron_start = sorted_exons[i][1]
            intron_end = sorted_exons[i + 1][0]

            if num_exons == 2:
                # Only one intron - use first_intron
                intron_label = "first_intron"
            elif i == 0:
                intron_label = "first_intron"
            elif i == num_exons - 2:
                intron_label = "last_intron"
            else:
                intron_label = "internal_intron"

            labels[intron_start:intron_end] = LABEL_SCHEME[intron_label]

        # Mark UTRs
        for start, end in gene.utrs_5:
            labels[start:end] = LABEL_SCHEME["5UTR"]
        for start, end in gene.utrs_3:
            labels[start:end] = LABEL_SCHEME["3UTR"]

        # Mark CDS regions with position-based exon labels
        sorted_cds = sorted(gene.cds)
        num_cds = len(sorted_cds)

        for i, (start, end) in enumerate(sorted_cds):
            if num_cds == 1:
                # Single CDS - use first_exon
                exon_label = "first_exon"
            elif i == 0:
                exon_label = "first_exon"
            elif i == num_cds - 1:
                exon_label = "last_exon"
            else:
                exon_label = "internal_exon"

            labels[start:end] = LABEL_SCHEME[exon_label]

        # Mark start and stop codons (highest priority, exactly 3bp)
        for start, end in gene.start_codons:
            labels[start:end] = LABEL_SCHEME["start_codon"]
        for start, end in gene.stop_codons:
            labels[start:end] = LABEL_SCHEME["stop_codon"]

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
                chunks.append(
                    {
                        "chrom": chunk.chrom,
                        "start": chunk.start,
                        "end": chunk.end,
                        "sequence": chunk.sequence,
                        "labels": chunk.labels,
                    }
                )

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
            seq = seq[: self.max_length]
            labels = labels[: self.max_length]

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


class MambaBlockStub(nn.Module):
    """Single Mamba block stub matching Mamba's API.

    Approximates compute pattern without SSM-specific ops.
    For development/testing on machines without GPU or mamba-ssm.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        # Input projection (like Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution (like Mamba's conv1d)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM approximation (real Mamba has selective scan here)
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass matching Mamba block pattern.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # In projection: split into x and z branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Convolution (transpose for conv1d)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, : x.shape[1]]  # Causal padding
        x_branch = x_branch.transpose(1, 2)

        # Activation
        x_branch = F.silu(x_branch)

        # SSM approximation
        x_branch = self.ssm_proj(x_branch)
        x_branch = F.silu(x_branch)

        # Gating
        x_branch = x_branch * F.silu(z)

        # Output projection
        out = self.out_proj(x_branch)

        return residual + out


class MambaEncoderStub(nn.Module):
    """CPU-compatible Mamba encoder stub for development/testing.

    Matches MambaEncoder API for drop-in replacement when mamba-ssm is not installed.
    Uses bidirectional processing via forward + reversed passes.
    """

    def __init__(
        self,
        input_dim: int = DNA_DIM,
        hidden_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim // 2  # Half for each direction

        # Input embedding
        self.embed = nn.Linear(input_dim, self.d_model)

        # Forward direction layers
        self.forward_layers = nn.ModuleList(
            [
                MambaBlockStub(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        # Backward direction layers
        self.backward_layers = nn.ModuleList(
            [
                MambaBlockStub(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            hidden: (batch, T, hidden_dim)
        """
        # Embed input
        h_fwd = self.embed(x)  # (batch, T, d_model)
        h_bwd = h_fwd.flip(dims=[1])  # Reverse for backward pass

        # Forward direction
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd)
            h_fwd = self.dropout(h_fwd)

        # Backward direction
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd)
            h_bwd = self.dropout(h_bwd)

        # Reverse backward output and concatenate
        h_bwd = h_bwd.flip(dims=[1])
        hidden = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, T, hidden_dim)

        return self.norm(hidden)


class MambaEncoder(nn.Module):
    """Bidirectional Mamba SSM encoder for DNA sequences.

    Uses forward + reversed Mamba layers concatenated for bidirectional output.
    Requires mamba-ssm package to be installed.
    """

    def __init__(
        self,
        input_dim: int = DNA_DIM,
        hidden_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba-ssm required for MambaEncoder. " "Install with: pip install mamba-ssm"
            )

        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim // 2  # Half for each direction

        # Input embedding
        self.embed = nn.Linear(input_dim, self.d_model)

        # Forward direction Mamba layers
        self.forward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        # Backward direction Mamba layers
        self.backward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            hidden: (batch, T, hidden_dim)
        """
        # Embed input
        h_fwd = self.embed(x)  # (batch, T, d_model)
        h_bwd = h_fwd.flip(dims=[1])  # Reverse for backward pass

        # Forward direction
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd) + h_fwd  # Residual connection
            h_fwd = self.dropout(h_fwd)

        # Backward direction
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd) + h_bwd  # Residual connection
            h_bwd = self.dropout(h_bwd)

        # Reverse backward output and concatenate
        h_bwd = h_bwd.flip(dims=[1])
        hidden = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, T, hidden_dim)

        return self.norm(hidden)


def create_encoder(
    encoder_type: Literal["bilstm", "mamba", "mamba_stub"] = "bilstm",
    input_dim: int = DNA_DIM,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    # Mamba-specific
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> nn.Module:
    """Factory function to create encoder by type.

    Args:
        encoder_type: One of "bilstm", "mamba", "mamba_stub"
        input_dim: Input dimension (DNA_DIM=5 for one-hot)
        hidden_dim: Output hidden dimension
        num_layers: Number of encoder layers
        dropout: Dropout rate
        d_state: Mamba SSM state dimension
        d_conv: Mamba local convolution width
        expand: Mamba expansion factor

    Returns:
        Encoder module with forward(x) -> (batch, T, hidden_dim)
    """
    if encoder_type == "bilstm":
        return BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "mamba":
        return MambaEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "mamba_stub":
        return MambaEncoderStub(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


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
        backend: str = "streaming",  # Default: streaming Triton kernel
        use_triton: bool = True,  # Default: use Triton (not PyTorch fallback)
    ):
        super().__init__()
        self.encoder = encoder
        self.max_duration = max_duration
        self.backend = backend
        self.use_triton = use_triton

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

    def compute_loss(
        self,
        sequence: Tensor,
        lengths: Tensor,
        labels: Tensor,
        backend: str | None = None,
        use_triton: bool | None = None,
    ) -> Tensor:
        """Compute NLL loss.

        Args:
            sequence: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            backend: Override default backend ("streaming", "exact", "auto")
            use_triton: Override default Triton usage
        """
        hidden = self.encoder(sequence)
        return self.crf.compute_loss(
            hidden,
            lengths,
            labels,
            backend=backend if backend is not None else self.backend,
            use_triton=use_triton if use_triton is not None else self.use_triton,
        )

    def decode(
        self,
        sequence: Tensor,
        lengths: Tensor,
        backend: str | None = None,
        use_triton: bool | None = None,
    ):
        """Viterbi decoding.

        Args:
            sequence: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            backend: Override default backend ("streaming", "exact", "auto")
            use_triton: Override default Triton usage
        """
        hidden = self.encoder(sequence)
        return self.crf.decode_with_traceback(
            hidden,
            lengths,
            backend=backend if backend is not None else self.backend,
            use_triton=use_triton if use_triton is not None else self.use_triton,
        )


class ExonIntronModelPytorchCRF(nn.Module):
    """
    Exon/Intron model using pytorch-crf for baseline comparison.

    Uses the same encoder interface but replaces SemiMarkovCRFHead
    with torchcrf.CRF (linear CRF, no duration modeling).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model. " "Install with: pip install pytorch-crf"
            )
        self.encoder = encoder
        self.emission_proj = nn.Linear(hidden_dim, num_classes)
        self.crf = TorchCRF(num_classes, batch_first=True)

    def forward(self, sequence: Tensor, _lengths: Tensor) -> Tensor:
        """Forward pass returning emission scores."""
        hidden = self.encoder(sequence)
        return self.emission_proj(hidden)

    def compute_loss(
        self,
        sequence: Tensor,
        lengths: Tensor,
        labels: Tensor,
        **_kwargs,
    ) -> Tensor:
        """
        Compute NLL loss using pytorch-crf.

        Args:
            sequence: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            **_kwargs: Ignored (for API compatibility with ExonIntronModel)
        """
        hidden = self.encoder(sequence)
        emissions = self.emission_proj(hidden)

        _, seq_len = sequence.shape[:2]
        mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)

        # pytorch-crf requires valid labels (no -100), replace padding with 0
        labels_clean = labels.clone()
        labels_clean[labels == -100] = 0

        # pytorch-crf.forward() returns log-likelihood, we want NLL
        log_likelihood = self.crf(emissions, labels_clean, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(self, sequence: Tensor, lengths: Tensor) -> list[list[int]]:
        """
        Viterbi decode to get best label sequences.

        Returns:
            List of label sequences (one per batch element).
        """
        hidden = self.encoder(sequence)
        emissions = self.emission_proj(hidden)

        _, seq_len = sequence.shape[:2]
        mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Returns list of lists (batch_size x variable_length)
        return self.crf.decode(emissions, mask=mask)


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
        if labels[i] != labels[i - 1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    tolerances: list[int] | None = None,
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    if tolerances is None:
        tolerances = [0, 1, 2, 5, 10]
    results = {"tolerance_" + str(t): {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}

    for pred, target in zip(predictions, targets, strict=False):
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
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sampler: DistributedSampler | None = None,
    epoch: int = 0,
    distributed: bool = False,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train (may be DDP-wrapped)
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        sampler: Optional DistributedSampler (for DDP training)
        epoch: Current epoch number (for sampler shuffling)
        distributed: Whether running in distributed mode (for loss aggregation)
    """
    # Set epoch for proper shuffling in distributed mode
    if sampler is not None:
        sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0
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

    avg_loss = total_loss / num_batches

    # Aggregate loss across all GPUs in distributed mode
    if distributed and dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        # Weighted average by number of batches per GPU
        avg_loss = loss_tensor[0].item() / dist.get_world_size()

    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_pytorch_crf: bool = False,
) -> SegmentMetrics:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate (ExonIntronModel or ExonIntronModelPytorchCRF)
        dataloader: Data loader
        device: Device
        is_pytorch_crf: If True, model is ExonIntronModelPytorchCRF (returns list[list[int]])
    """
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
            true_labels = labels[i, :seq_len].cpu().numpy()

            if is_pytorch_crf:
                # pytorch-crf returns list[list[int]]
                pred_labels = np.array(result[i][:seq_len], dtype=np.int64)
                pred_segs = extract_segments(pred_labels)
            else:
                # torch-semimarkov returns DecodeResult with segments
                # NOTE: torch_semimarkov.Segment uses INCLUSIVE end (end=5 means position 5 included)
                # Convert to exclusive for numpy slicing: [start:end+1]
                pred_labels = np.zeros(seq_len, dtype=np.int64)
                for seg in result.segments[i]:
                    pred_labels[seg.start : seg.end + 1] = seg.label
                pred_segs = [
                    SegmentAnnotation(s.start, s.end + 1, s.label) for s in result.segments[i]
                ]

            true_segs = extract_segments(true_labels)

            all_predictions.append(pred_labels)
            all_targets.append(true_labels)
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
    model_type: Literal["pytorch-crf", "linear", "semicrf"] = "semicrf",
    encoder_type: Literal["bilstm", "mamba", "mamba_stub"] = "bilstm",
    max_duration: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # Mamba-specific hyperparameters
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    # Backend selection
    backend: str = "streaming",  # Default: streaming Triton kernel
    use_triton: bool = True,  # Default: use Triton (not PyTorch fallback)
    # Logging
    log_every: int = 1,  # Evaluate and log every N epochs
    # Distributed training
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[nn.Module, SegmentMetrics | None]:
    """
    Train a model and return it with metrics.

    Args:
        data_dir: Directory containing preprocessed data
        model_type: CRF type - "pytorch-crf", "linear" (K=1), or "semicrf" (K>1)
        encoder_type: Encoder type - "bilstm", "mamba", or "mamba_stub"
        max_duration: Maximum segment duration K (for semicrf)
        hidden_dim: Hidden dimension for encoder and CRF
        num_layers: Number of encoder layers (2 for BiLSTM, 4 recommended for Mamba)
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Number of training epochs
        device: Device to train on (ignored if distributed=True)
        d_state: Mamba SSM state dimension
        d_conv: Mamba local convolution width
        expand: Mamba expansion factor
        backend: Backend for semi-CRF - "streaming" (default), "exact", or "auto"
        use_triton: Whether to use Triton kernels (default True)
        log_every: Evaluate and log every N epochs (default 1)
        distributed: Enable distributed training (DDP)
        rank: GPU rank for distributed training
        world_size: Total number of GPUs for distributed training
    """
    # Device setup - use rank for distributed, otherwise use provided device
    if distributed:
        device_obj = torch.device(f"cuda:{rank}")
    else:
        device_obj = torch.device(device)

    # Load data
    train_dataset = GencodeDataset(data_dir / "train.jsonl")
    val_dataset = GencodeDataset(data_dir / "val.jsonl")

    # Use DistributedSampler for DDP training (only for train, not val)
    train_sampler: DistributedSampler | None = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    # Validation: no distributed sampler - only rank 0 evaluates on full set
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Adjust layers for encoder type (Mamba typically uses more layers)
    encoder_layers = num_layers if encoder_type == "bilstm" else max(num_layers, 4)

    # Build encoder
    encoder = create_encoder(
        encoder_type=encoder_type,
        hidden_dim=hidden_dim,
        num_layers=encoder_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )

    # Build model based on type
    if model_type == "pytorch-crf":
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model. Install with: pip install pytorch-crf"
            )
        model: nn.Module = ExonIntronModelPytorchCRF(
            encoder=encoder,
            hidden_dim=hidden_dim,
        ).to(device_obj)
        k = 1
    elif model_type == "linear":
        k = 1
        model = ExonIntronModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
            backend=backend,
            use_triton=use_triton,
        ).to(device_obj)
    else:  # semicrf
        k = max_duration
        model = ExonIntronModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
            backend=backend,
            use_triton=use_triton,
        ).to(device_obj)

    # Wrap model in DDP for distributed training
    if distributed:
        model = DDP(model, device_ids=[rank])

    # Only log from main process
    if is_main_process():
        logger.info(
            f"Model: {model_type} + {encoder_type}, K={k}, "
            f"backend={backend}, triton={use_triton}, "
            f"params={sum(p.numel() for p in model.parameters()):,}"
            + (f", distributed={world_size} GPUs" if distributed else "")
        )

    is_pytorch_crf = model_type == "pytorch-crf"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_metrics: SegmentMetrics | None = None

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device_obj,
            sampler=train_sampler,
            epoch=epoch,
            distributed=distributed,
        )
        scheduler.step()

        # Evaluate every N epochs
        # In distributed mode: only rank 0 evaluates on full val set, others wait
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            if is_main_process():
                val_metrics = evaluate(model, val_loader, device_obj, is_pytorch_crf)

                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                    f"Val F1 (pos): {val_metrics.position_f1_macro:.4f} | "
                    f"Val F1 (boundary): {val_metrics.boundary_f1:.4f} | "
                    f"Val F1 (segment): {val_metrics.segment_f1:.4f}"
                )

                if val_metrics.boundary_f1 > best_val_f1:
                    best_val_f1 = val_metrics.boundary_f1
                    best_metrics = val_metrics

            # Synchronize all processes after evaluation
            if distributed and dist.is_initialized():
                dist.barrier()

    return model, best_metrics


def train_distributed_worker(
    rank: int,
    world_size: int,
    data_dir: Path,
    **kwargs,
):
    """Worker function for distributed training.

    Spawned by torch.multiprocessing.spawn for each GPU.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        data_dir: Path to data directory
        **kwargs: All other arguments passed to train_model
    """
    setup_distributed(rank, world_size)

    try:
        train_model(
            data_dir,
            distributed=True,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )
    finally:
        cleanup_distributed()


def compare_models(
    data_dir: Path,
    encoder_type: Literal["bilstm", "mamba", "mamba_stub"] = "bilstm",
    max_duration: int = 500,
    **kwargs,
):
    """
    Compare CRF models in a 3-way comparison.

    Models compared:
    1. pytorch-crf (optional): External linear CRF baseline
    2. K=1 torch-semimarkov: Linear CRF via streaming kernel
    3. K>1 torch-semimarkov: Full semi-CRF with duration modeling

    All use the same encoder (BiLSTM or Mamba).
    """
    results: dict[str, SegmentMetrics | None] = {}

    # 1. pytorch-crf baseline (optional)
    if HAS_TORCHCRF:
        logger.info("=" * 60)
        logger.info(f"Training PYTORCH-CRF ({encoder_type} encoder)")
        logger.info("=" * 60)
        _, pytorch_crf_metrics = train_model(
            data_dir,
            model_type="pytorch-crf",
            encoder_type=encoder_type,
            **kwargs,
        )
        results["pytorch_crf"] = pytorch_crf_metrics
    else:
        logger.warning("pytorch-crf not installed, skipping baseline")

    # 2. torch-semimarkov K=1 (linear CRF)
    logger.info("=" * 60)
    logger.info(f"Training LINEAR CRF K=1 ({encoder_type} encoder)")
    logger.info("=" * 60)
    _, linear_metrics = train_model(
        data_dir,
        model_type="linear",
        encoder_type=encoder_type,
        **kwargs,
    )
    results["linear_crf"] = linear_metrics

    # 3. Semi-CRF K>1
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF K={max_duration} ({encoder_type} encoder)")
    logger.info("=" * 60)
    _, semicrf_metrics = train_model(
        data_dir,
        model_type="semicrf",
        encoder_type=encoder_type,
        max_duration=max_duration,
        **kwargs,
    )
    results["semi_crf"] = semicrf_metrics

    # Print comparison
    _print_comparison(results)

    return results


def _print_comparison(results: dict[str, SegmentMetrics | None]) -> None:
    """Print comparison table for model results."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    # Determine which models we have
    models = [k for k, v in results.items() if v is not None]
    if len(models) < 2:
        print("Not enough models for comparison")
        return

    # Header
    header = f"{'Metric':<25}"
    for model in models:
        header += f" {model:>15}"
    if "semi_crf" in models and "linear_crf" in models:
        header += f" {'Δ(semi-lin)':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    # Main metrics
    for metric_name in ["position_f1_macro", "boundary_f1", "segment_f1"]:
        row = f"{metric_name:<25}"
        vals = {}
        for model in models:
            if results[model] is not None:
                val = getattr(results[model], metric_name)
                vals[model] = val
                row += f" {val:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        if "semi_crf" in vals and "linear_crf" in vals:
            delta = vals["semi_crf"] - vals["linear_crf"]
            row += f" {delta:>+12.4f}"
        print(row)

    # Boundary F1 at different tolerances
    print("\nBoundary F1 at different tolerances:")
    for tol in [0, 1, 2, 5, 10]:
        row = f"  tol={tol:<3}"
        vals = {}
        for model in models:
            if results[model] is not None:
                val = results[model].boundary_f1_tolerance.get(tol, 0)
                vals[model] = val
                row += f" {val:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        if "semi_crf" in vals and "linear_crf" in vals:
            delta = vals["semi_crf"] - vals["linear_crf"]
            row += f" {delta:>+12.4f}"
        print(row)

    # Duration calibration (KL divergence)
    print("\nDuration KL divergence (lower is better):")
    for label in LABEL_NAMES:
        row = f"  {label:<20}"
        for model in models:
            if results[model] is not None:
                kl = results[model].duration_kl.get(label, float("nan"))
                row += f" {kl:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        print(row)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess Gencode annotations")
    preprocess_parser.add_argument("--gtf", type=Path, required=True, help="Gencode GTF file")
    preprocess_parser.add_argument(
        "--fasta", type=Path, required=True, help="Reference genome FASTA"
    )
    preprocess_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    preprocess_parser.add_argument("--chunk-size", type=int, default=10000)
    preprocess_parser.add_argument("--overlap", type=int, default=500)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument(
        "--model",
        choices=["pytorch-crf", "linear", "semicrf"],
        default="semicrf",
        help="Model type: pytorch-crf (external lib), linear (K=1), semicrf (K>1)",
    )
    train_parser.add_argument(
        "--encoder",
        choices=["bilstm", "mamba", "mamba_stub"],
        default="bilstm",
        help="Encoder type: bilstm (default), mamba (requires mamba-ssm), mamba_stub (CPU dev)",
    )
    train_parser.add_argument("--max-duration", type=int, default=500)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of encoder layers (2 for BiLSTM, 4 recommended for Mamba)",
    )
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    # Mamba-specific
    train_parser.add_argument("--d-state", type=int, default=16, help="Mamba SSM state dimension")
    train_parser.add_argument("--d-conv", type=int, default=4, help="Mamba local convolution width")
    train_parser.add_argument("--expand", type=int, default=2, help="Mamba expansion factor")
    # Backend selection
    train_parser.add_argument(
        "--backend",
        choices=["streaming", "exact", "auto"],
        default="streaming",
        help="Backend for semi-CRF: streaming (default, memory-efficient), exact, or auto",
    )
    train_parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch fallback)",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Evaluate and log every N epochs (default: 1)",
    )
    # Distributed training
    train_parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-GPU DDP training",
    )
    train_parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of GPUs for distributed training (default: all available)",
    )
    train_parser.add_argument(
        "--nccl-ifname",
        type=str,
        default=None,
        help="Network interface for NCCL (e.g., ib0 for IPoIB, eth0 for ethernet)",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare CRF models (pytorch-crf, linear, semi-CRF)"
    )
    compare_parser.add_argument("--data-dir", type=Path, required=True)
    compare_parser.add_argument(
        "--encoder",
        choices=["bilstm", "mamba", "mamba_stub"],
        default="bilstm",
        help="Encoder type for all models",
    )
    compare_parser.add_argument("--max-duration", type=int, default=500)
    compare_parser.add_argument("--hidden-dim", type=int, default=256)
    compare_parser.add_argument("--num-layers", type=int, default=2)
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    compare_parser.add_argument(
        "--output-json", type=Path, default=None, help="Save results to JSON file"
    )
    # Mamba-specific
    compare_parser.add_argument("--d-state", type=int, default=16)
    compare_parser.add_argument("--d-conv", type=int, default=4)
    compare_parser.add_argument("--expand", type=int, default=2)
    # Backend selection
    compare_parser.add_argument(
        "--backend",
        choices=["streaming", "exact", "auto"],
        default="streaming",
        help="Backend for semi-CRF: streaming (default), exact, or auto",
    )
    compare_parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch fallback)",
    )
    compare_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Evaluate and log every N epochs (default: 1)",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_gencode(
            args.gtf, args.fasta, args.output_dir, chunk_size=args.chunk_size, overlap=args.overlap
        )
    elif args.command == "train":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Print diagnostics before training
        print_diagnostics(
            batch_size=args.batch_size,
            seq_len=10000,  # Default chunk size
            max_duration=args.max_duration,
        )

        # Common training kwargs
        train_kwargs = {
            "model_type": args.model,
            "encoder_type": args.encoder,
            "max_duration": args.max_duration,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
            "backend": args.backend,
            "use_triton": not args.no_triton,
            "log_every": args.log_every,
        }

        if args.distributed:
            import torch.multiprocessing as mp

            world_size = args.world_size or torch.cuda.device_count()
            if world_size < 2:
                logger.warning("Distributed training requires >=2 GPUs, falling back to single GPU")
                train_model(args.data_dir, device=device, **train_kwargs)
            else:
                # Set NCCL interface if specified (for IPoIB or specific network)
                if args.nccl_ifname:
                    os.environ["NCCL_SOCKET_IFNAME"] = args.nccl_ifname
                    logger.info(f"Using network interface: {args.nccl_ifname}")

                logger.info(f"Launching DDP training with {world_size} GPUs")
                mp.spawn(
                    train_distributed_worker,
                    args=(world_size, args.data_dir),
                    kwargs=train_kwargs,
                    nprocs=world_size,
                    join=True,
                )
        else:
            # Single-GPU training
            train_model(args.data_dir, device=device, **train_kwargs)
    elif args.command == "compare":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Print diagnostics before comparison
        print_diagnostics(
            batch_size=args.batch_size,
            seq_len=10000,  # Default chunk size
            max_duration=args.max_duration,
        )
        results = compare_models(
            args.data_dir,
            encoder_type=args.encoder,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            backend=args.backend,
            use_triton=not args.no_triton,
            log_every=args.log_every,
        )
        if args.output_json:
            from datetime import datetime

            output = {
                "task": "gencode",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "encoder_type": args.encoder,
                    "max_duration": args.max_duration,
                    "hidden_dim": args.hidden_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                "results": {k: v.to_dict() if v is not None else None for k, v in results.items()},
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
