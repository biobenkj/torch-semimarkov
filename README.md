# torch-semimarkov

Efficient Semi-Markov CRF Inference for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This library provides optimized implementations of Semi-Markov CRF inference algorithms, benchmarked and documented in:

> **Practical Semi-Markov CRF Inference for Genomic Sequence Annotation**
> Benjamin K. Johnson (2026)

**Key finding:** Memory, not time, is the binding constraint. Vectorized linear scan is universally applicable across all genomic parameter regimes.

## Why Semi-Markov CRFs?

Semi-Markov CRFs extend linear-chain CRFs with explicit duration modeling:

```
psi(x_{s:e}, c', c, d) = psi_emission(x_{s:e}, c) + psi_transition(c', c) + psi_duration(c, d)
```

This provides three structural advantages:
1. **Guaranteed valid segmentations** - segments tile the sequence by construction
2. **Explicit duration modeling** - incorporate biological priors (exon lengths, TE sizes)
3. **Segment-level posteriors** - amenable to calibration and uncertainty quantification

## Installation

### Basic Installation (CPU)

```bash
pip install torch-semimarkov
```

### Development Installation

```bash
git clone https://github.com/benjohnson/torch-semimarkov.git
cd torch-semimarkov
pip install -e ".[dev]"
```

### With CUDA Support

For GPU acceleration with custom CUDA kernels:

```bash
pip install -e . --config-settings="--build-option=--cuda"
```

Or using environment variable:
```bash
TORCH_SEMIMARKOV_CUDA=1 pip install -e .
```

## Quick Start

```python
import torch
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

# Parameters
batch_size = 4
seq_length = 1000   # T
max_duration = 16   # K
num_classes = 6     # C

# Create model
model = SemiMarkov(LogSemiring)

# Edge potentials: (batch, T-1, K, C, C)
edge = torch.randn(batch_size, seq_length - 1, max_duration, num_classes, num_classes)
lengths = torch.full((batch_size,), seq_length)

# Forward pass (partition function)
log_Z, _, _ = model.logpartition(
    edge,
    lengths=lengths,
    use_linear_scan=True,      # O(T) linear scan
    use_vectorized=True,       # Vectorized (2-3x speedup)
)

# Backward pass for gradients
log_Z.sum().backward()
```

## Understanding T, K, C: A Genomics Perspective

The three key parameters map directly to biological questions:

### T = Sequence Length (How much genome you decode at once)

**T is your sequence length in "positions."** In genomics that could be:
- **base pairs** (1 bp = 1 position), or
- **tokens** (e.g., 4/8/16 bp per token, k-mers, pooled stride)

Intuitively: **T is the width of context you're asking the model to produce one coherent annotation over.**

Examples:
- **Single-gene locus decoding:** T might be the span of a gene + flanks (e.g., 10k-200k bp depending on organism and gene)
- **Chunked genome scanning:** T is your chunk size (e.g., 32k bp, 128k bp, 1M bp), with overlaps
- **Transcript-level decoding:** T might be the span covering one transcript model

Why it matters:
- Larger **T** lets the model use more context (long introns, alternative exons, regulatory signals), and reduces edge effects from chunking
- Computationally, **vectorized scan time grows ~linearly with T**, while tree-style methods often run into memory cliffs

### K = Maximum Segment Duration (How long can a single segment be?)

Semi-CRF predicts segments, not per-base labels. **K is the max duration (segment length) you consider** when forming a segment that ends at position t.

In genomics, segment lengths correspond to biological pieces like:
- Exon length
- Intron length
- UTR length
- Intergenic/background length
- TE element length (or chunks thereof)

Intuition: **K sets the longest "one-piece" region your decoder can explain without splitting it.**

Key point: introns and intergenic regions can be *huge*. If you choose K to cover "everything," K will explode and the DP becomes expensive. Common strategies:
- **Cap-and-split background-like labels** (e.g., allow intron/background segments up to K=512 tokens; longer ones become multiple consecutive segments)
- **Label-specific K** (small K for exons/UTRs, larger K for introns/background)

In practice you often set K based on **a quantile** of lengths you want to model as single segments (p95/p99), not max.

### C = Number of Segment Labels (How detailed is your annotation?)

**C is the label set size.** In genomics this is your "annotation ontology" at the segment level.

Common choices:
- **Very coarse (C ~ 3):** {exon, intron, intergenic/background}
- **Gene-structure (C ~ 4-8):**
  - exon split into {CDS, UTR} or {5'UTR, CDS, 3'UTR}
  - plus intron + intergenic
  - maybe ncRNA-exon as separate
- **Richer (C ~ 10-30+):**
  - strand-split labels (+/- doubles C)
  - more biotypes / TE classes / special signals

Intuition: **C is how detailed you want the segmentation "vocabulary" to be.**

### The Decoder's Decision at Each Step

At the end of position **t**, the semi-CRF is effectively asking:

> "Did a segment end here? If so:
> (1) what label c is it? (one of C labels)
> (2) how long was it? (1...K)
> (3) what label did we come from? (another of C labels, if you model transitions)"

So:
- **T** controls how many places you make that decision
- **K** controls how far back you're allowed to "look" when proposing a segment
- **C** controls how many semantic types of segments exist

### Practical Examples

**Gene structure annotation:**
- T = locus/chunk length you decode in one shot
- K = max exon/intron/background segment you allow without splitting
- C = exon/intron/UTR/etc label set

**TE annotation:**
- T = chunk length
- K = max TE segment length you treat as one element (or chunk)
- C = TE families/superfamilies (+ background)

### Choosing Parameters (Intuition-First)

1. Pick **T** based on your inference unit: "one gene locus" or "one genome chunk."
   Bigger T = fewer boundary artifacts, more context, more compute.

2. Pick **C** based on what you want to output (coarse vs detailed).

3. Pick **K** based on what you want the model to treat as a single coherent region:
   - K_exon around "almost all exons"
   - K_intron/background capped and split for very long regions

## Algorithmic Backends

The library implements six backends for semi-CRF inference:

| Backend | Time | DP Memory | Parallel Depth | Best For |
|---------|------|-----------|----------------|----------|
| `linear_scan` | O(TKC^2) | O(TKC) | O(T) | Reference implementation |
| **`linear_scan_vectorized`** | O(TKC^2) | O(TKC) | O(T) | **General use (2-3x faster)** |
| `linear_scan_streaming` | O(TKC^2) | **O(KC)** | O(T) | Memory-constrained settings |
| `binary_tree` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Small KC only |
| `binary_tree_sharded` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Reduced peak memory |
| `block_triangular` | O(TKC^2) | O(T(KC)^2) | O(log T) | Structured sparsity |

### Recommendation

**Default to `linear_scan_vectorized`.** Despite O(T) sequential depth, it:
- Succeeds across all tested configurations
- Uses 2-4x less memory than tree-based methods
- Achieves 2-3x speedup over scalar implementation

The `linear_scan_streaming` backend has true O(KC) DP state (independent of T) using a ring buffer, ideal when memory is the binding constraint.

Tree-based methods exhaust GPU memory for state-space sizes KC > 150 due to O((KC)^3) temporaries from log-semiring matrix multiplication.

## Benchmarking

Run the included benchmarks to reproduce paper results:

```bash
# Memory analysis across backends
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --backends linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,binary_tree_sharded,block_triangular

# Multi-backend comparison
python benchmarks/benchmark_backends.py \
    --device cuda:0 \
    --T 512,1024,2048 \
    --K 12,16,20 \
    --C 3,6 \
    --backends binary_tree,linear_scan,linear_scan_vectorized \
    --csv results.csv

# Dense vs banded comparison
python benchmarks/benchmark_grid.py \
    --T 512,1024,2048 \
    --K 8,12,16 \
    --C 3,6 \
    --csv grid_results.csv
```

## Testing

Verify all backends produce equivalent results:

```bash
# Run full test suite
pytest tests/ -v

# Quick equivalence check
python tests/test_backend_equivalence.py --device cuda --quick

# All backends (smaller configs to avoid OOM)
python tests/test_backend_equivalence.py --device cuda --configs small,medium
```

## Project Structure

```
torch-semimarkov/
├── src/torch_semimarkov/
│   ├── __init__.py              # Main exports
│   ├── semimarkov.py            # SemiMarkov class with DP algorithms
│   │                            #   - _dp_standard (scalar linear scan)
│   │                            #   - _dp_standard_vectorized (vectorized)
│   │                            #   - _dp_scan_streaming (O(KC) memory)
│   │                            #   - logpartition (binary tree)
│   ├── helpers.py               # Base structured prediction class
│   ├── banded.py                # CPU BandedMatrix implementation
│   ├── banded_utils.py          # Bandwidth measurement, permutations
│   ├── blocktriangular.py       # Block-triangular matrix operations
│   ├── semirings/
│   │   ├── __init__.py          # Semiring exports
│   │   ├── semirings.py         # Log, Max, Standard, KMax, Entropy
│   │   └── checkpoint.py        # CheckpointSemiring, CheckpointShardSemiring
│   └── _genbmm/                 # CUDA extension (optional)
│       ├── __init__.py
│       ├── genmul.py            # PyTorch autograd functions
│       ├── sparse.py            # BandedMatrix with CUDA
│       └── csrc/                # CUDA kernel sources
│           ├── matmul_cuda.cpp
│           ├── matmul_cuda_kernel.cu
│           └── banded_cuda_kernel.cu
├── benchmarks/
│   ├── benchmark_backends.py    # Multi-backend timing comparison
│   ├── benchmark_grid.py        # Parameter sweep benchmarks
│   ├── benchmark_memory_analysis.py  # Memory profiling
│   └── plot_figures.py          # Paper figure generation
├── tests/
│   ├── test_backend_equivalence.py   # All-backend equivalence tests
│   └── test_partition_equivalence.py # Streaming scan tests
├── pyproject.toml
└── setup.py                     # CUDA extension build
```

## API Reference

### SemiMarkov

```python
class SemiMarkov(semiring):
    def logpartition(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,            # (batch,) sequence lengths
        use_linear_scan=True,    # Use O(T) linear scan (default: auto-select)
        use_vectorized=True,     # Vectorize inner loops
        use_banded=False,        # Use banded matrix operations
        banded_perm="auto",      # Permutation strategy
        banded_bw_ratio=0.6,     # Bandwidth threshold
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute log partition function and backward pointers.

        Returns:
            v: (batch,) log partition values
            edges: edge marginals (for gradient computation)
            charts: intermediate DP tables
        """

    def _dp_scan_streaming(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,
        force_grad=False,
    ) -> Tuple[Tensor, List[Tensor], None]:
        """
        True streaming scan with O(KC) DP state.

        Uses ring buffer with head pointer - memory independent of T.
        """
```

### Semirings

```python
from torch_semimarkov.semirings import (
    LogSemiring,      # Standard log-space (sum-product)
    MaxSemiring,      # Viterbi decoding (max-product)
    StdSemiring,      # Standard arithmetic
    KMaxSemiring,     # Top-k paths
    EntropySemiring,  # Entropy computation
)

from torch_semimarkov.semirings.checkpoint import (
    CheckpointSemiring,       # Gradient checkpointing
    CheckpointShardSemiring,  # Sharded checkpointing (reduces O((KC)^3) peak)
)
```

## Citation

If you use this library, please cite:

```bibtex
@article{johnson2026semimarkov,
  title={Practical Semi-Markov CRF Inference for Genomic Sequence Annotation},
  author={Johnson, Benjamin K.},
  journal={bioRxiv},
  year={2026}
}
```

## Acknowledgments

This library builds on:
- [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush
- [genbmm](https://github.com/harvardnlp/genbmm) for CUDA generalized batch matrix multiplication

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclosure of AI-Assisted Development

This project incorporates contributions produced with the assistance of AI-based software development tools. These tools were used for ideation, code generation, debugging, and documentation support. Final implementations were made by the authors, and all code has undergone manual review and testing. The project author(s) assumes full responsibility for the accuracy, integrity, and licensing compliance of all included code.
