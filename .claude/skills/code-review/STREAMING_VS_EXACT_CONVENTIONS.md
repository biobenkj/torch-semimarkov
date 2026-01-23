# Semi-Markov CRF Implementation Reference

This document defines the tensor shapes, loop bounds, and indexing conventions for the Semi-Markov CRF implementations. **Read this before debugging any streaming/exact backend mismatch bugs.**

## Table of Contents

1. [Two Implementation Paradigms](#two-implementation-paradigms)
2. [Core Tensor Shapes](#core-tensor-shapes)
3. [Sequence Length Convention](#sequence-length-convention)
4. [Duration Indexing Convention](#duration-indexing-convention)
5. [Edge Tensor Convention](#edge-tensor-convention)
6. [Loop Bounds Reference](#loop-bounds-reference)
7. [Ring Buffer Convention](#ring-buffer-convention)
8. [Content Score Computation](#content-score-computation)
9. [Common Bug Sources](#common-bug-sources)

---

## Two Implementation Paradigms

### 1. Edge Tensor API (`semimarkov.py`)

The `SemiMarkov` class operates on **pre-materialized edge tensors**:

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

model = SemiMarkov(LogSemiring)
log_Z, _, _ = model.logpartition(edge, lengths=lengths)
```

- **Input**: Edge tensor of shape `(batch, N-1, K, C, C)`
- **Memory**: O(T × K × C²) — prohibitive for genome-scale
- **Semiring support**: Full (Log, Max, Entropy, KMax, etc.)
- **Use case**: Short sequences (T < 50K), research/flexibility

### 2. Streaming API (`streaming/`)

The streaming implementation computes edges **on-the-fly** from cumulative scores:

```python
from torch_semimarkov import semi_crf_streaming_forward

partition = semi_crf_streaming_forward(
    cum_scores,      # (batch, T+1, C)
    transition,      # (C, C)
    duration_bias,   # (K, C)
    lengths,         # (batch,)
    K,               # max duration
    semiring="log",
    use_triton=True,
)
```

- **Input**: Cumulative scores, NOT edge tensor
- **Memory**: O(K × C) ring buffer — genome-scale friendly
- **Semiring support**: Limited (Log, Max only)
- **Use case**: Long sequences (T > 10K), production

---

## Core Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `scores` | `(batch, T, C)` | Per-position label scores |
| `cum_scores` | `(batch, T+1, C)` | Cumulative scores with leading zero row |
| `edge` (SemiMarkov) | `(batch, N-1, K, C, C)` | Pre-materialized edge potentials |
| `transition` | `(C, C)` | Label transition matrix |
| `duration_bias` | `(K, C)` | Duration-specific label bias |
| `lengths` | `(batch,)` | Sequence lengths |
| `alpha_ring` | `(batch, K, C)` | Ring buffer for streaming forward |

### Dimension Definitions

| Symbol | Meaning |
|--------|---------|
| `T` | **Sequence length** (number of positions in input) |
| `N` | **SemiMarkov convention**: `N = T + 1` (see below) |
| `K` | **Maximum segment duration** |
| `C` | **Number of label classes** |

---

## Sequence Length Convention

⚠️ **CRITICAL: This is the #1 source of bugs**

### Streaming API

The streaming API uses **T** directly:

```
Input:  cum_scores has shape (batch, T+1, C)
Loop:   for t in range(1, T + 1):  # t = 1, 2, ..., T
Access: cum_scores[:, t, :]        # positions 0 to T
```

- **`lengths` tensor**: Contains actual sequence lengths (e.g., 100)
- **Valid positions**: 0, 1, 2, ..., T-1 (T positions total)
- **Loop range**: `range(1, T + 1)` iterates t = 1, 2, ..., T
- **Final position**: At `t == lengths[b]`, we capture the final alpha

### SemiMarkov (Edge Tensor) API

The SemiMarkov class uses **N = edge.shape[1] + 1**:

```python
# In _check_potentials:
batch, N_1, K, C, C2 = self._get_dimension(edge)
N = N_1 + 1  # <-- N is one more than edge's time dimension
```

- **Edge shape**: `(batch, N-1, K, C, C)` — note it's `N-1`, not `N`
- **`lengths` tensor**: Must satisfy `max(lengths) == N`
- **Valid edge positions**: 0, 1, 2, ..., N-2 (N-1 positions)

### Converting Between Conventions

When building an edge tensor with T positions for use with SemiMarkov:

```python
# Build edge with T positions (matching streaming convention)
edge = build_edge_tensor(...)  # shape (batch, T, K, C, C)

# SemiMarkov interprets this as N = T + 1
# So pass lengths + 1 to SemiMarkov:
model = SemiMarkov(LogSemiring)
result = model.logpartition(edge, lengths=lengths + 1)  # <-- CRITICAL
```

**Rule of thumb**: If your edge tensor has `T` positions in dimension 1, pass `lengths + 1` to SemiMarkov.

---

## Duration Indexing Convention

### Shape: `duration_bias[K, C]`

The duration bias tensor has shape `(K, C)` where:
- Index 0 → duration 1
- Index 1 → duration 2
- ...
- Index K-1 → duration K (or clamped for K=1)

### Duration Mapping in Code

```python
# For a segment of actual duration k (k = 1, 2, ..., K):
dur_idx = min(k, K - 1)  # Clamp to valid index range
bias = duration_bias[dur_idx, :]
```

### K=1 Special Case

When `K=1` (only duration-1 segments allowed):
- `dur_idx = min(1, 0) = 0` — always use index 0
- The loop `range(1, max(K, 2))` ensures at least one iteration

---

## Edge Tensor Convention

### Shape: `edge[batch, n, k, c_dest, c_src]`

The edge tensor encodes:
```
edge[b, n, k, c2, c1] = log P(segment starting at n with duration k+1,
                              transitioning from label c1 to label c2)
```

### Edge Computation Formula

```python
edge[n, k, c_dest, c_src] = content_score[c_dest]
                          + duration_bias[k, c_dest]
                          + transition[c_src, c_dest]

# Where content_score = cum_scores[n + k + 1] - cum_scores[n]
# for a segment spanning positions [n, n + k + 1)
```

### Transition Matrix Orientation

The transition matrix has shape `(C_src, C_dest)`:
- `transition[c_src, c_dest]` = log score for transitioning from c_src to c_dest
- In edge computation, we typically use `transition.T` to get `(C_dest, C_src)`

---

## Transition Matrix Convention

### Storage: `transition[C_src, C_dest]`

The transition parameter is stored with source-first indexing:
- `transition[i, j]` = score for transitioning FROM label `i` TO label `j`
- Shape: `(C, C)`
- This matches standard CRF literature conventions

### Usage in Edge Computation

When building edge tensors, the transition is **transposed** to match edge orientation:

```python
# Edge tensor has shape (batch, N, K, C_dest, C_src)
# transition has shape (C_src, C_dest)
# We need (C_dest, C_src) for broadcasting

edge[:, n, k] = segment_score.unsqueeze(-1) + transition.T.unsqueeze(0)
#               (batch, C, 1)               + (1, C_dest, C_src)
```

### Why Transpose?

The edge tensor indexes as `edge[..., c_dest, c_src]` because:
1. Forward pass: `alpha[t, c_dest] = logsumexp over c_src of (alpha[t-k, c_src] + edge[..., c_dest, c_src])`
2. The reduction is over `c_src`, so it should be the last dimension for efficient memory access
3. This matches the standard HMM/CRF literature orientation

### Usage in Gold Scoring

When scoring gold sequences, the transition is accessed directly without transposition:

```python
# For a segment transitioning from prev_label to curr_label:
trans_score = transition[prev_label, curr_label]
```

### Usage Across Codebase

| Location | Access Pattern | Notes |
|----------|---------------|-------|
| `_build_edge_tensor` | `transition.T` | Transposed for edge broadcasting |
| `compute_edge_block_streaming` | `transition.T` | Same transposition |
| `score_gold_vectorized` | `transition[prev, curr]` | Direct access |
| `_compute_segment_score` | `transition[prev, curr]` | Direct access |
| Triton kernels | Loaded as transposed | For edge computation |

---

## Loop Bounds Reference

### Streaming Forward (PyTorch Reference)

```python
# File: streaming/pytorch_reference.py
# semi_crf_streaming_forward_pytorch()

T = cum_scores.shape[1] - 1  # T+1 -> T

for t in range(1, T + 1):  # t = 1, 2, ..., T
    k_eff = min(K - 1, t)

    for k in range(1, max(k_eff + 1, 2)):  # k = 1, ..., min(K-1, t)
        start = t - k  # segment starts at position (t - k)

        # Edge computed for segment [start, start + k)
        # which covers positions start, start+1, ..., start+k-1
        # ending just before position t
```

### Streaming Forward (Triton Kernel)

```python
# File: streaming/triton_forward.py
# semi_crf_streaming_scan_kernel()

# Loop: t = 1, 2, ..., T
for t in tl.range(1, T + 1):

    # Loop: k = 1, 2, ..., min(K-1, t)
    # tl.maximum(K, 2) handles K=1 case
    for k in tl.range(1, tl.maximum(K, 2)):
        k_valid = (k <= t) & (k <= tl.maximum(K - 1, 1))
        start_pos = t - k
```

### SemiMarkov Linear Scan

```python
# File: semimarkov.py
# _dp_scan_streaming()

N = N_1 + 1  # edge has N-1 positions

for n in range(1, N):  # n = 1, 2, ..., N-1
    k_eff = max(1, min(K - 1, n))
    dur = torch.arange(1, k_eff + 1)  # durations 1, 2, ..., k_eff
    start = n - dur  # segment start positions

    # Access edge[:, start, dur_clamped, :, :]
```

### Building Edge Tensor (nn.py)

```python
# File: nn.py
# _build_edge_tensor()

for n in range(T):  # n = 0, 1, ..., T-1 (T positions)
    for k in range(1, max(min(K, T - n + 1), 2)):  # k = 1, ..., min(K-1, T-n)
        # Segment [n, n+k) must fit within [0, T)
        content = cum_scores[:, n + k, :] - cum_scores[:, n, :]
        dur_idx = min(k, K - 1)
        edge[:, n, dur_idx, :, :] = ...
```

---

## Ring Buffer Convention

### Structure: `alpha_ring[batch, K, C]`

The ring buffer stores the last K alpha values using modular indexing:

```python
# At iteration t:
# - alpha[t] is being computed
# - alpha[t-1] is at index (t-1) % K
# - alpha[t-k] is at index (t-k) % K for k = 1, ..., min(K-1, t)

# Write alpha[t]:
ring_idx = t % K
alpha_ring[:, ring_idx, :] = alpha_t

# Read alpha[t-k]:
ring_idx = (t - k) % K
alpha_prev = alpha_ring[:, ring_idx, :]
```

### Initialization

```python
alpha_ring = torch.full((batch, K, C), NEG_INF)
alpha_ring[:, 0, :] = 0.0  # alpha[0] = 0 for all labels (uniform init)
```

---

## Content Score Computation

### From Cumulative Scores

```python
# For segment [start, end) spanning positions start, start+1, ..., end-1:
content_score = cum_scores[:, end, :] - cum_scores[:, start, :]
```

### Cumulative Score Structure

```python
# cum_scores has shape (batch, T+1, C)
# cum_scores[:, 0, :] = 0  (leading zeros)
# cum_scores[:, t, :] = sum(scores[:, 0:t, :])  for t > 0

# Build from scores:
cum_scores = torch.zeros(batch, T + 1, C)
cum_scores[:, 1:] = torch.cumsum(scores, dim=1)
```

### Zero-Centering for Numerical Stability

For long sequences (T > 10K), zero-center scores before cumsum:

```python
scores_centered = scores - scores.mean(dim=1, keepdim=True)
cum_scores[:, 1:] = torch.cumsum(scores_centered, dim=1)
```

---

## Common Bug Sources

### 1. Length Off-by-One with SemiMarkov

❌ **Wrong**:
```python
edge = build_edge_tensor(...)  # shape (batch, T, K, C, C)
model.logpartition(edge, lengths=lengths)  # BUG: will fail assertion
```

✅ **Correct**:
```python
edge = build_edge_tensor(...)  # shape (batch, T, K, C, C)
model.logpartition(edge, lengths=lengths + 1)  # Pass lengths + 1
```

### 2. Edge Tensor Shape Mismatch

❌ **Wrong**: Building `(batch, T-1, K, C, C)` to "match SemiMarkov convention"

✅ **Correct**: Build `(batch, T, K, C, C)` and pass `lengths + 1` to SemiMarkov

### 3. Duration Index vs Actual Duration

❌ **Wrong**:
```python
# Accessing duration_bias[k] for duration k
duration_bias[k]  # Off by one!
```

✅ **Correct**:
```python
# For actual duration k (1-indexed), use index min(k, K-1)
dur_idx = min(k, K - 1)
duration_bias[dur_idx]
```

### 4. K=1 Edge Case

The loop `range(1, K)` produces empty range when K=1. Fix:

```python
for k in range(1, max(K, 2)):  # Ensures at least one iteration
    ...
```

### 5. Score Preprocessing Mismatch

The streaming API may zero-center scores. If using exact backend for comparison:

❌ **Wrong**: Using raw scores in exact backend, zero-centered in streaming

✅ **Correct**: Apply same preprocessing in both paths
```python
scores_centered = scores - scores.mean(dim=1, keepdim=True)
# Use scores_centered for both streaming and exact
```

### 6. Marginals Shape After lengths+1

When using `model.marginals(edge, lengths=lengths+1)`:
- Returns marginals with T positions (same as edge)
- Don't add padding/prepend operations that expect T-1 positions

---

## Quick Reference Card

| What | Streaming | SemiMarkov |
|------|-----------|------------|
| Main loop | `t = 1..T` | `n = 1..N-1` |
| Edge positions | Computed on-the-fly | `N-1 = T` positions |
| Lengths param | Actual lengths | `lengths + 1` |
| Memory | O(KC) | O(T×K×C²) |
| Duration loop | `k = 1..min(K-1,t)` | `k = 1..min(K-1,n)` |
| dur_idx | `min(k, K-1)` | `min(k, K-1)` |
| Ring buffer idx | `t % K` | `n % K` |

---

## File Locations

| File | Purpose |
|------|---------|
| `semimarkov.py` | Edge tensor API, SemiMarkov class |
| `streaming/pytorch_reference.py` | Pure PyTorch streaming reference |
| `streaming/triton_forward.py` | Triton GPU kernels |
| `nn.py` | SemiMarkovCRFHead with backend routing |
| `uncertainty.py` | UncertaintySemiMarkovCRFHead |

---

*Last updated: 2026-01-23*
*Created to prevent future debugging sessions on loop bounds mismatches*
