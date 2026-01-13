<div align="center">

# torch-semimarkov

Efficient Semi-Markov CRF Inference using PyTorch and Triton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/biobenkj/torch-semimarkov/branch/main/graph/badge.svg)](https://codecov.io/gh/biobenkj/torch-semimarkov)

[Install](#installation) | [Quick Start](#quick-start) | [Docs](docs/) | [Examples](#quick-start) | [GitHub](https://github.com/biobenkj/torch-semimarkov)

</div>

## Overview

Semi-Markov CRFs are powerful models for sequences with natural segment structure, such as genomic annotations. However, their inference algorithms are resource-intensive. The segment-level forward pass requires $O(TKC^2)$ time and critically $O(TKC)$ memory, where $T$ is sequence length, $K$ is maximum segment duration, and $C$ is the number of states. For chromosome-scale sequences with biologically realistic duration bounds, this memory footprint quickly exceeds GPU capacity.

Existing implementations navigate this through various tradeoffs—bounding $K$, chunked processing, or filtering heuristics. This package takes a different approach:

Streaming the linear scan collapses memory to $O(KC)$—independent of sequence length and duration.

This makes Semi-Markov CRF inference practical for genome-scale annotation without architectural compromises.

**torch-semimarkov** provides:

- **Streaming scan** — $O(KC)$ memory, universally applicable across genomic parameter regimes
- **Triton fused kernel** — optional GPU acceleration with up to 45× speedup

## Why Semi-Markov CRFs in genomics contexts?

Many biological sequences have inherent *segment* structurelike genes, exons, transcript isoforms, chromatin states, transposable elements, etc. where segment *duration* carries biological meaning. Linear-chain CRFs handle sequential dependencies well but lack explicit duration modeling, often requiring post-hoc grouping or producing biologically implausible outputs (single-base "exons," fragmentary annotations).

Semi-Markov CRFs resolve this by modeling segments directly. The potential function scores an entire segment spanning positions $s$ to $e$:

$$\psi(x_{s:e}, c', c, d) = \underbrace{\psi_{\text{emit}}(x_{s:e}, c)}_{\text{sequence content}} + \underbrace{\psi_{\text{trans}}(c', c)}_{\text{state grammar}} + \underbrace{\psi_{\text{dur}}(c, d)}_{\text{length prior}}$$

Each term encodes a distinct biological constraint: does the sequence *content* match this annotation? Is this state *transition* grammatically valid? Is this *duration* plausible for this feature type?

This formulation provides:

- **Valid segmentations by construction** — segments tile the sequence exactly, eliminating post-processing
- **Explicit duration modeling** — encode priors like "exons are typically 50–300 bp"
- **Segment-level posteriors** — enable calibration and principled uncertainty quantification over whole features, not just positions

These properties also make Semi-Markov CRFs natural structured decoders for neural sequence encoders, adding output guarantees that per-position prediction heads typically don't provide.

## Installation

```bash
# Basic installation
pip install torch-semimarkov

# Development installation
git clone https://github.com/biobenkj/torch-semimarkov.git
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
log_Z, _ = model.logpartition(edge, lengths=lengths)

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

- [Integration guide](docs/workflow_integration.md) — how to use torch-semimarkov with BERT, Mamba, CNNs, and other encoders
- [Parameter guide: T, K, C](docs/parameter_guide.md) — understanding sequence length, duration, and state dimensions
- [Semirings guide](docs/semirings.md) - context and intuition for semirings used in torch-semimarkov
- [Backends and Triton kernel](docs/backends.md) — algorithm selection and GPU acceleration
- [API reference](docs/api.md) — detailed API documentation
- [Benchmarking](docs/benchmarks.md) — performance measurement
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
| **Vectorized Scan** | O(TKC) memory, 2-3x faster than standard linear scan |
| **Binary Tree** | O(log N) depth, high memory for large KC |
| **Block-Triangular** | Exploits duration constraint sparsity |
| **Semirings** | Log, Max, Std, KMax, Entropy, CrossEntropy |
| **Checkpoint Semiring** | Memory-efficient gradients |
| **BandedMatrix (CPU)** | Prototype and not a recommended backend |
| **CUDA Extension** | Builds when nvcc available |
| **Triton Kernel** | ~45x speedup on GPU |

## Acknowledgments

This library builds on [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush and [genbmm](https://github.com/harvardnlp/genbmm) for CUDA generalized batch matrix multiplication.

## License

MIT License - see [LICENSE](LICENSE) for details.
