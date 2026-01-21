# Backends and Triton kernel

This project provides multiple semi-CRF inference backends with different
performance and memory profiles.

## Backend summary

| Backend | Time | DP memory | Best for |
|---------|------|-----------|----------|
| `linear_scan_streaming` | O(TKC^2) | O(KC) | **Default** - PyTorch reference implementation |
| `triton` | O(TKC^2) | O(KC) | GPU inference (~45x faster than PyTorch) |

## Recommendation

**Default: `linear_scan_streaming`** - always available, O(KC) memory via ring buffer.

**Use `triton` for GPU inference** - ~45x speedup with custom Triton kernel.

## Triton fused streaming kernel (~45x speedup)

`torch_semimarkov.triton_scan.semi_crf_triton_forward` provides a fused O(T)
streaming scan that keeps the K x C frontier in fast memory. It mirrors the
streaming scan but collapses the loop into a single GPU kernel, yielding
~45x speedup compared to the vectorized PyTorch implementation.

### Hybrid inference/training approach

The Triton kernel uses a **hybrid approach** that automatically selects the
optimal execution path based on whether gradients are needed:

| Context | Path | Description |
|---------|------|-------------|
| **Inference** (`requires_grad=False`) | Custom Triton kernel | Maximum speed (~45x faster) |
| **Training** (`requires_grad=True`) | `torch.compile` | Efficient automatic backward pass |
| **CPU fallback** | PyTorch reference | Always available |

This gives you the best of both worlds:
- Blazing fast inference with the custom kernel
- Efficient training with automatic backward pass generation via `torch.compile`

### Supported semirings

The Triton kernel supports both **Log** and **Max** semirings:

- `semiring="log"` (default): Log partition function (sum-product)
- `semiring="max"`: Viterbi score (max-product)

### Parameters

```python
semi_crf_triton_forward(
    edge,              # (batch, T-1, K, C, C) potentials
    lengths,           # (batch,) sequence lengths
    use_triton=True,   # Use Triton kernel for inference when available
    validate=False,    # Use float64 PyTorch for numerical validation
    semiring="log",    # "log" for partition function, "max" for Viterbi
    use_compile=True,  # Use torch.compile for training backward pass
)
```

### Behavior

- **Inference path**: When `edge.requires_grad=False` and CUDA is available,
  uses the custom Triton kernel for maximum performance.
- **Training path**: When `edge.requires_grad=True`, uses `torch.compile` on
  the PyTorch implementation, which generates optimized Triton kernels for
  both forward AND backward passes automatically.
- **Fallback**: Falls back to PyTorch when Triton is unavailable or inputs
  are on CPU.
- **Validation**: `validate=True` runs a float64 PyTorch reference for
  numerical checks.
- **Compilation overhead**: The first training call incurs a one-time
  `torch.compile` overhead (a few seconds). Subsequent calls reuse the cached
  compiled kernel.

### Examples

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

# GPU inference: uses fast custom Triton kernel
edge = edge.cuda()
lengths = lengths.cuda()
partition = semi_crf_triton_forward(edge, lengths)

# GPU training: uses torch.compile for efficient backward
edge_train = edge.requires_grad_(True)
partition = semi_crf_triton_forward(edge_train, lengths)
partition.sum().backward()

# Viterbi score (max semiring)
viterbi_score = semi_crf_triton_forward(edge, lengths, semiring="max")

# Disable torch.compile (use gradient checkpointing instead)
partition = semi_crf_triton_forward(edge_train, lengths, use_compile=False)
```
