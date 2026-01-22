r"""Streaming API for memory-efficient Semi-CRF inference.

This module implements on-the-fly edge computation using prefix-sum decomposition:
edge potentials are computed on-the-fly from pre-projected cumulative scores,
eliminating the need to materialize the full (batch, T-1, K, C, C) edge tensor.

.. important::
    **When to use this module vs. triton_scan:**

    Use ``streaming`` (this module) for:
        - **Training** (always) - hand-written Triton backward kernels
        - **Inference** (recommended) - faster than triton_scan even when edge fits
        - **Very long sequences** (T = 10K - 400K+) - edge tensor cannot fit

    Use ``triton_scan`` module only when:
        - Edge tensor is pre-computed from an external source
        - Inference only (no gradients needed)
        - Edge tensor already fits in GPU memory

    **Memory comparison:**

    +-----------------------+------------------+-------------------+
    | Scenario              | edge tensor size | cum_scores size   |
    +=======================+==================+===================+
    | T=1K, K=32, C=24      | 18 MB            | 96 KB             |
    +-----------------------+------------------+-------------------+
    | T=10K, K=100, C=24    | 5.5 GB           | 960 KB            |
    +-----------------------+------------------+-------------------+
    | T=400K, K=3K, C=24    | **2.76 TB**      | 38 MB             |
    +-----------------------+------------------+-------------------+

    For the T=400K case, the edge tensor cannot fit in memory. This module
    computes edges on-the-fly from O(T×C) cumulative scores instead.

    **Performance comparison (forward-only, NVIDIA L40S):**

    Streaming beats triton_scan on both speed AND memory:

    +-----------------------+---------------------+---------------------+-------------------+
    | Configuration         | triton_scan         | streaming           | Streaming wins by |
    +=======================+=====================+=====================+===================+
    | K=100, batch=64       | 127ms, 14GB         | 38ms, 6MB           | 3.35× faster      |
    +-----------------------+---------------------+---------------------+-------------------+
    | K=500, batch=32       | 330ms, 35GB         | 224ms, 3MB          | 1.48× faster      |
    +-----------------------+---------------------+---------------------+-------------------+

    Why streaming is faster:

    - **Memory bandwidth**: Loading O(T×K×C²) edges from memory is slower than
      computing O(T×C) edge blocks on-the-fly from cumulative scores
    - **Cache efficiency**: Streaming keeps working set in L1/L2 cache
    - **Linear batch scaling**: Memory grows as O(batch×T×C), not O(batch×T×K×C²)

    **Training advantages:**

    - Hand-written Triton backward kernels (no torch.compile overhead)
    - No compilation latency (torch.compile takes 20+ minutes for T=1000)
    - No RecursionError from deep computational graphs
    - No OOM from compiled gradient buffers

API Comparison
--------------
The ``triton_scan`` module takes a **pre-computed edge tensor**::

    edge = model(x)  # shape: (batch, T-1, K, C, C) - must fit in GPU memory!
    partition = semi_crf_triton_forward(edge, lengths)

This module takes **cumulative scores** and computes edges on-the-fly::

    cum_scores = cumsum(projected, dim=1)  # shape: (batch, T+1, C) - much smaller!
    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

Memory Complexity
-----------------
- Pre-computed edge API (triton_scan): O(T × K × C²) - 2.76 TB for T=400K, K=3K, C=24
- Streaming API (this module): O(T × C + K × C + C²) - ~50 MB for same dimensions

Streaming Edge Computation
--------------------------
Instead of pre-computing edges, we pre-project encoder features to label space
BEFORE the kernel (loop-invariant projection), then compute edges on-the-fly inside:

    # Outside kernel (parallel, efficient)
    projected = h @ W_content                    # (batch, T, C)
    projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
    cum_scores = cumsum(projected.float(), dim=1)  # (batch, T+1, C) in float32

    # Inside kernel (just vector ops, no matmuls)
    content_score = cum_scores[:, t+k, :] - cum_scores[:, t, :]  # (batch, C)
    segment_score = content_score + duration_bias[k]
    edge_block = segment_score.unsqueeze(-1) + transition        # (batch, C, C)

The edge potential for segment [t, t+k) with label c_dest from c_src is::

    edge[t, k, c_dest, c_src] = (cum_scores[t+k, c_dest] - cum_scores[t, c_dest])
                              + duration_bias[k, c_dest]
                              + transition[c_src, c_dest]

This structure means you **never need to materialize the full edge tensor**.

Numerical Stability
-------------------
Two critical requirements for T=400K+ sequences:

1. **Float32 cumsum**: Cumsum must be float32 to avoid precision loss.
   Float16 loses all precision at T=400K magnitudes.

2. **Zero-centering**: Without centering, cumsum drifts to ~T magnitude.
   At T=400K, float32 epsilon at that magnitude is ~0.04 - any signal
   smaller than that is completely erased. Zero-centering keeps magnitude
   at √T (~632 for T=400K), preserving signals down to ~10⁻⁴.

Usage
-----
>>> import torch
>>> from torch_semimarkov.streaming import semi_crf_streaming_forward
>>>
>>> # Pre-project features (outside kernel)
>>> h = encoder(x)  # (batch, T, hidden_dim)
>>> projected = h @ W_content
>>> projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
>>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
>>> cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
>>>
>>> # Streaming forward (edges computed on-the-fly)
>>> partition = semi_crf_streaming_forward(
...     cum_scores, transition, duration_bias, lengths, K
... )

See Also
--------
:mod:`torch_semimarkov.triton_scan` : Inference only, when edge tensor is pre-computed externally
:class:`torch_semimarkov.SemiMarkov` : High-level API with marginals and sampling
"""

from .autograd import (
    SemiCRFStreaming,
    SemiCRFStreamingTriton,
    semi_crf_streaming_forward,
)
from .constants import NEG_INF
from .pytorch_reference import (
    _compute_checkpoint_interval,
    compute_edge_block_streaming,
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)

# Re-export HAS_TRITON for external checks
try:
    from .triton_forward import HAS_TRITON
except ImportError:
    HAS_TRITON = False

# Conditionally export Triton launchers
if HAS_TRITON:
    from .triton_backward import launch_streaming_triton_backward
    from .triton_forward import launch_streaming_triton_kernel

__all__ = [
    # Main API
    "semi_crf_streaming_forward",
    # Autograd Functions
    "SemiCRFStreaming",
    "SemiCRFStreamingTriton",
    # PyTorch reference implementations
    "semi_crf_streaming_forward_pytorch",
    "semi_crf_streaming_backward_pytorch",
    "compute_edge_block_streaming",
    # Utilities
    "_compute_checkpoint_interval",
    "NEG_INF",
    "HAS_TRITON",
    # Triton launchers (conditionally available)
    "launch_streaming_triton_backward",
    "launch_streaming_triton_kernel",
]
