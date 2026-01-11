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

Highlights:
- Vectorized linear scan (2-3x speedup) for general use
- True streaming scan with O(KC) memory via ring buffer
- Optional Triton fused streaming kernel for GPU acceleration

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

### Optional Triton Kernel (GPU)

The fused streaming kernel uses [Triton](https://github.com/openai/triton) and is optional:

```bash
pip install triton
```

Triton is used automatically when available and the input is CUDA.

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

### Triton Fused Streaming Kernel

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

edge = edge.cuda()
lengths = lengths.cuda()
partition = semi_crf_triton_forward(edge, lengths, use_triton=True)
```

## Documentation

- [Parameter guide: T, K, C](docs/parameter_guide.md)
- [Backends and Triton kernel](docs/backends.md)
- [Benchmarking](docs/benchmarks.md)
- [API reference](docs/api.md)
- [AI disclosure](docs/disclosure.md)

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
├── docs/
│   ├── api.md                  # API reference
│   ├── backends.md             # Backend overview and Triton kernel
│   ├── benchmarks.md           # Benchmark recipes
│   ├── disclosure.md           # AI-assisted development disclosure
│   └── parameter_guide.md      # T/K/C parameter guide
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
│   ├── triton_scan.py           # Triton fused streaming scan (optional)
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
│   ├── test_partition_equivalence.py # Streaming scan tests
│   ├── test_triton_scan.py           # Triton scan fallback tests
│   ├── test_banded_matrix.py         # BandedMatrix unit tests
│   ├── test_banded_utils.py          # Banded utilities tests
│   ├── test_blocktriangular.py       # Block-triangular matmul tests
│   ├── test_semirings.py             # Semiring unit tests
│   └── test_semimarkov_utils.py      # SemiMarkov helpers tests
├── pyproject.toml
└── setup.py                     # CUDA extension build
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
