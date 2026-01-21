<div align="center">

# torch-semimarkov

Efficient Semi-Markov CRF Inference using PyTorch and Triton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/biobenkj/torch-semimarkov/branch/main/graph/badge.svg)](https://codecov.io/gh/biobenkj/torch-semimarkov)

[Install](#installation) | [Quick Start](#quick-start) | [Docs](docs/) | [Examples](examples/) | [GitHub](https://github.com/biobenkj/torch-semimarkov)

</div>

## Overview

Semi-Markov CRFs are powerful models for sequences with natural segment structure, such as genomic annotations. However, their inference algorithms are resource-intensive. The segment-level forward pass requires $O(TKC^2)$ time and critically $O(TKC)$ memory, where $T$ is sequence length, $K$ is maximum segment duration, and $C$ is the number of states. For chromosome-scale sequences with biologically realistic duration bounds, this memory footprint quickly exceeds GPU capacity.

Existing implementations navigate this through various tradeoffs—bounding $K$, chunked processing, or filtering heuristics. This package takes a different approach:

Streaming the linear scan collapses memory to $O(KC)$—independent of sequence length and duration.

This makes Semi-Markov CRF inference practical for genome-scale annotation without architectural compromises.

**torch-semimarkov** provides:

- **Streaming scan** — $O(KC)$ memory, universally applicable across genomic parameter regimes
- **Triton fused kernel** — optional GPU acceleration

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

# Optional Triton kernel for GPU acceleration
pip install triton
```

## Quick Start

```python
import torch
from torch_semimarkov import SemiMarkovCRFHead

# Create CRF head (integrates with any encoder)
crf = SemiMarkovCRFHead(
    num_classes=24,      # C: number of segment labels
    max_duration=100,    # K: maximum segment length
    hidden_dim=512       # matches encoder output
)

# Encoder output (from Mamba, Transformer, CNN, etc.)
batch, T = 4, 1000
hidden_states = torch.randn(batch, T, 512)
lengths = torch.full((batch,), T)

# Training: compute NLL loss
labels = torch.randint(0, 24, (batch, T))
loss = crf.compute_loss(hidden_states, lengths, labels)
loss.backward()

# Inference: partition function or Viterbi decoding
log_Z = crf(hidden_states, lengths)['partition']
viterbi_score = crf.decode(hidden_states, lengths)
```

Works with PyTorch Lightning and DDP out of the box—see [examples/lightning_integration.py](examples/lightning_integration.py).

For the low-level API with explicit edge tensors and semiring control, see the [API reference](docs/api.md).

## Tensor Conventions

**Edge tensor indexing:** `edge[batch, position, duration, c_dest, c_src]`

This library follows **destination-first** convention for edge tensors, where `edge[..., j, i]` represents the potential for transitioning **from** label `i` **to** label `j`. This differs from some other CRF libraries that use source-first ordering.

**Example:**
```python
# edge[b, t, k, j, i] represents:
#   - Batch item b
#   - Segment starting at position t
#   - Duration k (segment spans positions t to t+k-1)
#   - Transition FROM label i TO label j
```

**Transition matrix:** Similarly, `transition[c_src, c_dest]` stores the score for transitioning from `c_src` to `c_dest`.

**Duration bias indexing:** `duration_bias[k, c]` stores the log-probability bias for segments of duration `k` with label `c`.

- Index 0 is unused (no segments of duration 0)
- Valid durations: 1 to K-1 (where K = `max_duration`)
- Durations ≥ K are clamped to K-1

```python
# A segment spanning positions [t, t+2] (3 positions, duration=3)
# uses duration_bias[3, label]
```

**Special case K=1:** When `max_duration=1`, the model behaves like a standard HMM where all segments have duration 1. In this case, `duration_bias[0]` stores the bias for duration 1 (due to clamping).

### Triton Kernel

When Triton is installed, torch-semimarkov uses fused GPU kernels that significantly accelerate both forward and backward passes.

**How it works:**

The streaming API computes edge potentials on-the-fly from cumulative scores rather than materializing the full `O(T × K × C²)` edge tensor. The Triton kernel fuses this computation with the forward scan:

```
# Edge computed on-the-fly (never materialized):
content = cum_scores[t+k, c] - cum_scores[t, c]  # O(1) lookup
edge[t, k, c_dst, c_src] = content + duration_bias[k, c] + transition[c_src, c_dst]
```

Key optimizations:
- **Fused edge computation** — computes edges on-the-fly via prefix-sum, avoiding the full edge tensor
- **O(KC) memory** — ring buffer for DP state, independent of sequence length
- **Custom backward kernel** — custom Triton kernel for gradients, not autograd
- **Checkpointing** — trades compute for memory by recomputing alpha values during backward

**Usage:**

The `SemiMarkovCRFHead` uses Triton automatically when available—pass `use_triton=True` (the default on GPU) to `forward()`, `compute_loss()`, or `decode()`.

For direct access to the streaming kernel:

```python
from torch_semimarkov.streaming import semi_crf_streaming_forward

# Cumulative scores from encoder (see docs for zero-centering requirements)
partition = semi_crf_streaming_forward(
    cum_scores, transition, duration_bias, lengths, K
)
```

For performance characteristics, see [Benchmarking](docs/benchmarks.md).

## Documentation

- [Integration guide](docs/workflow_integration.md) — how to use torch-semimarkov with BERT, Mamba, CNNs, and other encoders
- [Parameter guide: T, K, C](docs/parameter_guide.md) — understanding sequence length, duration, and state dimensions
- [Semirings guide](docs/semirings.md) — context and intuition for semirings used in torch-semimarkov
- [Uncertainty and focused learning](docs/uncertainty_and_focused_learning.md) — boundary confidence, active learning, and clinical applications
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
| **Triton Kernel** | GPU acceleration, Log/Max semirings, custom forward/backward |
| **Semirings** | Log, Max, Std, KMax, Entropy, CrossEntropy, KLDivergence |

## Acknowledgments

This library builds on [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush. GPU kernels are written using [Triton](https://github.com/triton-lang/triton).

## License

MIT License - see [LICENSE](LICENSE) for details.
