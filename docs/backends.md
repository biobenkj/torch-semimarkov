# Backends and Triton kernel

This project provides GPU-accelerated semi-CRF inference backends using custom Triton kernels.

## Backend summary

| Backend | Time | DP memory | Best for |
|---------|------|-----------|----------|
| `streaming` (recommended) | O(TKC²) | O(KC) | **Default** - Training and inference |
| `triton_scan` | O(TKC²) | O(TKC²) | Inference only, when edge tensor pre-exists |

## Recommendation

**Default: Streaming API** (`semi_crf_streaming_forward` or `SemiMarkovCRFHead`)

- O(KC) memory via ring buffer (edges computed on-the-fly)
- Hand-written Triton backward kernels (no torch.compile overhead)
- Faster than triton_scan even when edge tensor fits in memory

**Use triton_scan only** when you have pre-computed edge tensors from an external source
and need inference only (no gradients).

## Streaming Triton Kernel (Recommended)

The streaming API computes edge potentials on-the-fly from O(T×C) cumulative scores,
eliminating the need for the O(T×K×C²) edge tensor.

**Performance comparison (forward-only, NVIDIA L40S):**

| Configuration | triton_scan | streaming | Streaming Advantage |
|---------------|-------------|-----------|---------------------|
| K=100, batch=64 | 127ms, 14GB | 38ms, 6MB | 3.35× faster, 2,393× less memory |
| K=500, batch=32 | 330ms, 35GB | 224ms, 3MB | 1.48× faster, 11,795× less memory |

**Why streaming is faster:**
- Memory bandwidth is the bottleneck, not compute
- Computing edges on-the-fly from O(T×C) cumulative scores is faster than loading
  O(T×K×C²) pre-computed edges from memory
- Linear batch scaling: memory grows as O(batch×T×C), not O(batch×T×K×C²)

**Training advantages:**
- Hand-written Triton backward kernels (no compilation overhead)
- No torch.compile latency (which takes 20+ minutes for T=1000)
- No RecursionError from deep computational graphs
- No OOM from compiled gradient buffers

## triton_scan Module (Inference Only)

> **Warning: Do NOT use triton_scan for training.**
>
> The triton_scan module uses `torch.compile` for backward passes, which has
> critical limitations at production scales:
>
> - **RecursionError**: T > 1000 exceeds Python recursion limit in inductor
> - **OOM during backward**: Compiled graphs need 2×+ memory for gradient buffers
> - **Compilation time**: 20+ minutes for T=1000, essentially unusable
>
> **Always use the streaming API for training.**

### Execution paths

| Context | Path | Status |
|---------|------|--------|
| **Inference** (`requires_grad=False`) | Custom Triton kernel | Works, but streaming is faster |
| **Training** (`requires_grad=True`) | `torch.compile` | **Broken at scale** (see warning) |
| **CPU fallback** | PyTorch reference | Use streaming API instead |

### When to use triton_scan

The **only** valid use case for `semi_crf_triton_forward`:
- You have pre-computed edge tensors from an external source
- You need inference only (no gradients)
- The edge tensor already fits in GPU memory

For all other cases, use the streaming API.

### Supported semirings

Both APIs support **Log** and **Max** semirings:

- `semiring="log"` (default): Log partition function (sum-product)
- `semiring="max"`: Viterbi score (max-product)

### Parameters (triton_scan)

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

### Examples

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

# GPU inference with pre-computed edges (only use case)
edge = edge.cuda()  # (batch, T-1, K, C, C) - must already exist!
lengths = lengths.cuda()
partition = semi_crf_triton_forward(edge, lengths)

# Viterbi score (max semiring)
viterbi_score = semi_crf_triton_forward(edge, lengths, semiring="max")
```

For training, use the streaming API instead:

```python
from torch_semimarkov.streaming import semi_crf_streaming_forward

# Streaming: compute edges on-the-fly (recommended for both training and inference)
cum_scores = cumsum(projected, dim=1)  # O(T×C) - much smaller!
partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
partition.sum().backward()  # Hand-written Triton backward kernel
```
