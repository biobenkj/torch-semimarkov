# torch-semimarkov

Efficient Semi-Markov CRF Inference for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This library provides optimized implementations of Semi-Markov CRF inference algorithms, benchmarked and documented in:

> **Practical Semi-Markov CRF Inference for Genomic Sequence Annotation**
> Benjamin K. Johnson (2026)

**Key finding:** Memory, not time, is the binding constraint. Vectorized linear scan is universally applicable across all genomic parameter regimes.

## Why Semi-Markov CRFs?

Semi-Markov CRFs extend linear-chain CRFs with explicit duration modeling:

```
ψ(x_{s:e}, c', c, d) = ψ_emission(x_{s:e}, c) + ψ_transition(c', c) + ψ_duration(c, d)
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

## Algorithmic Backends

The library implements five backends for semi-CRF inference:

| Backend | Time | Space | Parallel Depth | GPU Memory |
|---------|------|-------|----------------|------------|
| Scalar Linear Scan | O(TKC²) | O(TC) | O(T) | Low |
| **Vectorized Linear Scan** | O(TKC²) | O(KC²) | O(T) | **Low** |
| Binary Tree | O(TKC² log T) | O(TKC²) | O(log T) | High |
| Banded | O(TKC² log T) | O(TKC·BW) | O(log T) | Medium |
| Block-Triangular | O(TKC²) | O(TKC²) | O(log T) | High |

### Recommendation

**Default to vectorized linear scan.** Despite O(T) sequential depth, it:
- Succeeds across all tested configurations
- Uses 2-4x less memory than tree-based methods
- Achieves 2-3x speedup over scalar implementation

Tree-based methods exhaust GPU memory for state-space sizes KC > 150.

## Benchmarking

Run the included benchmarks to reproduce paper results:

```bash
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

## Project Structure

```
torch-semimarkov/
├── src/torch_semimarkov/
│   ├── __init__.py           # Main exports
│   ├── semimarkov.py         # SemiMarkov class with DP algorithms
│   ├── helpers.py            # Base structured prediction class
│   ├── banded.py             # CPU BandedMatrix implementation
│   ├── banded_utils.py       # Bandwidth measurement, permutations
│   ├── blocktriangular.py    # Block-triangular matrix operations
│   ├── semirings/            # Semiring implementations
│   │   ├── semirings.py      # Log, Max, Standard semirings
│   │   └── checkpoint.py     # Gradient checkpointing
│   └── _genbmm/              # CUDA extension (optional)
│       ├── genmul.py         # PyTorch autograd functions
│       ├── sparse.py         # BandedMatrix with CUDA
│       └── csrc/             # CUDA kernel sources
├── benchmarks/               # Paper reproduction scripts
└── pyproject.toml
```

## API Reference

### SemiMarkov

```python
class SemiMarkov(semiring):
    def logpartition(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,            # (batch,) sequence lengths
        use_linear_scan=True,    # Use O(T) linear scan
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
