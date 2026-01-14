# Backends and Triton kernel

This project provides multiple semi-CRF inference backends with different
performance and memory profiles.

## Backend summary

| Backend | Time | DP memory | Parallel depth | Best for |
|---------|------|-----------|----------------|----------|
| `linear_scan_streaming` | O(TKC^2) | O(KC) | O(T) | **Default** - best memory, near-optimal speed |
| `linear_scan_vectorized` | O(TKC^2) | O(TKC) | O(T) | When memory permits (2-3x faster than scalar) |
| `linear_scan` | O(TKC^2) | O(TKC) | O(T) | Reference implementation |
| `binary_tree` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Small KC only |
| `binary_tree_sharded` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Reduced peak memory |
| `block_triangular` | O(TKC^2) | O(T(KC)^2) | O(log T) | Structured sparsity |

## Recommendation

**Default: `linear_scan_streaming`** - best for most use cases:
- O(KC) memory via ring buffer - always fits
- Within a few percent of vectorized speed
- Works across all tested configurations

Use `use_vectorized=True` when memory permits for 2-3x speedup over scalar scan.

Tree-based methods can exhaust GPU memory for KC > 150 because of O((KC)^3)
log-semiring temporaries.

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
