<div align="center">

# torch-semimarkov

Efficient Semi-Markov CRF Inference for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/biobenkj/torch-semimarkov/branch/main/graph/badge.svg)](https://codecov.io/gh/biobenkj/torch-semimarkov)

[Install](#installation) | [Quick Start](#quick-start) | [Docs](docs/) | [Examples](#quick-start) | [GitHub](https://github.com/biobenkj/torch-semimarkov)

</div>

## Overview

Optimized Semi-Markov CRF inference algorithms for genomic sequence annotation.

> **Practical Semi-Markov CRF Inference for Genomic Sequence Annotation**
> Benjamin K. Johnson (2026)

**Key finding:** Memory, not time, is the binding constraint. Streaming linear scan is universally applicable across all genomic parameter regimes.

- Streaming scan with O(KC) memory (default) - within a few percent of vectorized speed
- Optional Triton fused kernel for up to 45x GPU speedup
- Vectorized scan available when memory permits (O(TKC) memory, 2-3x faster than scalar)

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

```bash
# Basic installation
pip install torch-semimarkov

# Development installation
git clone https://github.com/benjohnson/torch-semimarkov.git
cd torch-semimarkov
pip install -e ".[dev]"

# With CUDA support (requires nvcc via CUDA_HOME)
TORCH_SEMIMARKOV_CUDA=1 pip install -e .

# Optional Triton kernel for GPU (used automatically when available)
pip install triton
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
# Uses streaming scan by default: O(KC) memory
log_Z, _, _ = model.logpartition(edge, lengths=lengths)

# Backward pass for gradients
log_Z.sum().backward()
```

### Triton Fused Kernel (up to 45x speedup)

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

edge = edge.cuda()
lengths = lengths.cuda()
partition = semi_crf_triton_forward(edge, lengths)
```

## Documentation

- [Parameter guide: T, K, C](docs/parameter_guide.md)
- [Backends and Triton kernel](docs/backends.md)
- [Benchmarking](docs/benchmarks.md)
- [API reference](docs/api.md)
- [AI disclosure](docs/disclosure.md)

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=torch_semimarkov --cov-report=term-missing
```

Tests run CPU-only by default. GPU tests require CUDA and are skipped in CI.

## Implementation Status

| Component | Status |
|-----------|--------|
| **Streaming Scan** | O(KC) memory, default backend |
| **Vectorized Scan** | O(TKC) memory, 2-3x faster |
| **Binary Tree** | O(log N) depth, high memory for large KC |
| **Block-Triangular** | Exploits duration constraint sparsity |
| **Semirings** | Log, Max, Std, KMax, Entropy, CrossEntropy |
| **Checkpoint Semiring** | Memory-efficient gradients |
| **BandedMatrix (CPU)** | Lightweight prototyping |
| **CUDA Extension** | Builds when nvcc available |
| **Triton Kernel** | ~45x speedup on GPU |

## Citation

If you use this library, please cite:

```bibtex
@article{johnson2026semimarkov,
  title={Practical Semi-Markov CRF Inference for Genomic Sequence Annotation},
  author={Johnson, Benjamin K.},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

This library builds on [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush and [genbmm](https://github.com/harvardnlp/genbmm) for CUDA generalized batch matrix multiplication.

## License

MIT License - see [LICENSE](LICENSE) for details.
