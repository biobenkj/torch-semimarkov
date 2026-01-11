# Backends and Triton kernel

This project provides multiple semi-CRF inference backends with different
performance and memory profiles.

## Backend summary

| Backend | Time | DP memory | Parallel depth | Best for |
|---------|------|-----------|----------------|----------|
| `linear_scan` | O(TKC^2) | O(TKC) | O(T) | Reference implementation |
| `linear_scan_vectorized` | O(TKC^2) | O(TKC) | O(T) | General use (2-3x faster) |
| `linear_scan_streaming` | O(TKC^2) | O(KC) | O(T) | Memory-constrained settings |
| `binary_tree` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Small KC only |
| `binary_tree_sharded` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Reduced peak memory |
| `block_triangular` | O(TKC^2) | O(T(KC)^2) | O(log T) | Structured sparsity |

## Recommendation

Default to `linear_scan_vectorized` for most use cases:
- Works across all tested configurations
- Uses less memory than tree-based methods
- 2-3x faster than the scalar linear scan

`linear_scan_streaming` keeps only an O(KC) DP state using a ring buffer.
It is the best option when memory is the binding constraint.

Tree-based methods can exhaust GPU memory for KC > 150 because of O((KC)^3)
log-semiring temporaries.

## Triton fused streaming kernel (optional)

`torch_semimarkov.triton_scan.semi_crf_triton_forward` provides a fused O(T)
streaming scan that keeps the K x C frontier in fast memory. It mirrors the
streaming scan but collapses the loop into a single GPU kernel.

Behavior:
- Uses Triton when available and the input is CUDA.
- Falls back to the PyTorch reference when Triton is missing or inputs are CPU.
- `validate=True` runs a float64 PyTorch reference for numerical checks.

Example:

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

partition = semi_crf_triton_forward(edge.cuda(), lengths.cuda(), use_triton=True)
```
