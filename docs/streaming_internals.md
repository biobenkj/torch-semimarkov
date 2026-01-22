# Streaming Semi-CRF Internals

Technical reference for developers working with the streaming inference module.

## Overview

The streaming module (`torch_semimarkov.streaming`) provides memory-efficient Semi-CRF inference by computing edge potentials on-the-fly rather than materializing the full edge tensor.

### The Memory Problem

Standard Semi-CRF inference requires an edge tensor of shape `(batch, T, K, C, C)`:
- T = sequence length (e.g., 100,000 for genomic sequences)
- K = maximum segment duration (e.g., 1,000)
- C = number of states (e.g., 24)

This requires **O(T × K × C²)** memory, which quickly exceeds GPU capacity for genome-scale sequences.

### The Streaming Solution

The streaming approach reduces memory to **O(K × C)** by:
1. Using a **ring buffer** for forward/backward messages
2. Computing edge potentials **on-the-fly** from cumulative scores
3. **Checkpointing** the ring buffer state for the backward pass

### Module Structure

| File | Purpose |
|------|---------|
| [autograd.py](../src/torch_semimarkov/streaming/autograd.py) | Public API and autograd functions |
| [triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py) | Triton forward kernels (log/max semiring) |
| [triton_backward.py](../src/torch_semimarkov/streaming/triton_backward.py) | Triton backward kernel |
| [pytorch_reference.py](../src/torch_semimarkov/streaming/pytorch_reference.py) | Pure PyTorch reference implementation |
| [constants.py](../src/torch_semimarkov/streaming/constants.py) | Shared constants (NEG_INF) |

---

## Tensor Conventions

### End-Position Indexing

The streaming module uses **end-position indexing**:
- `t` = segment **end** position (inclusive, 1-indexed in the algorithm)
- `k` = segment **duration**
- Segment covers positions `[t-k, t-1]` (0-indexed)

This differs from start-position indexing where `t` would be the start.

```
Segment with t=5, k=3:
  positions: [2, 3, 4]  (0-indexed)
  start_pos = t - k = 2
  end_pos = t - 1 = 4
```

### Destination-First Edge Convention

Edge tensors use **destination-first** ordering:
```python
edge[..., c_dst, c_src]  # Transition FROM c_src TO c_dst
```

This means:
- `transition[c_src, c_dst]` stores score for c_src → c_dst
- When computing edges, we use `transition.T` to get `[c_dst, c_src]` layout

### Cumulative Score Layout

```python
cum_scores: (batch, T+1, C)
```

- Index 0 is the boundary (all zeros)
- Index `t` contains cumulative sum through position `t-1`
- Content score for segment `[start, end]` = `cum_scores[end+1] - cum_scores[start]`

---

## Ring Buffer Architecture

### Memory Layout

```python
alpha_ring: (batch, K, C)
```

Forward messages use a ring buffer indexed by `t % K`:
- `alpha_ring[:, t % K, :]` stores α values at position `t`
- Only the most recent `K` positions are kept
- Older values are overwritten as the scan progresses

### Why O(KC) Memory

Standard forward pass stores all α values: **O(T × C)** memory.

With max segment duration K, position `t` only depends on positions `[t-K+1, t-1]`. The ring buffer exploits this by storing only the K most recent values.

```
Time:     0   1   2   3   4   5   6   7   8   ...
Ring idx: 0   1   2   0   1   2   0   1   2   ...  (K=3)
          ↑           ↑           ↑
          overwritten by t=3, 6, 9, ...
```

---

## Edge Computation On-the-Fly

### The Decomposed Potential

Edge potentials are computed on-the-fly using prefix-sum decomposition:

```
edge[c_dst, c_src] = content_score[c_dst] + duration_bias[k, c_dst] + transition[c_src, c_dst]
```

Where the content score is derived from cumulative scores:
```
content_score = cum_scores[t, c_dst] - cum_scores[t-k, c_dst]
```

### Code Location

From [triton_forward.py:209-235](torch-semimarkov/src/torch_semimarkov/streaming/triton_forward.py#L209-L235):

```python
# Content score via cumsum difference
cum_end = tl.load(cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c, ...)
cum_start = tl.load(cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c, ...)
content_score = cum_end - cum_start  # (C_PAD,)

# Add duration bias
dur_bias = tl.load(duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c, ...)
segment_score = content_score + dur_bias  # (C_PAD,)

# Build edge block
edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)
```

---

## Code-to-Math Correspondence

| Code Variable | Math Notation | Shape | Description |
|--------------|---------------|-------|-------------|
| `cum_scores[:, t, c]` | S_t,c | (batch, T+1, C) | Cumulative projected scores |
| `transition[c_src, c_dst]` | T_c',c | (C, C) | Label transitions (src → dst) |
| `duration_bias[k, c]` | B_k,c | (K, C) | Duration-specific label bias |
| `alpha_ring[:, t%K, :]` | α̃_t | (batch, C) | Log-forward messages (ring buffer) |
| `beta_ring[:, t%K, :]` | β̃_t | (batch, C) | Log-backward messages (ring buffer) |
| `ring_checkpoints[:, i, :, :]` | Ω_i | (batch, K, C) | Saved ring buffer state at checkpoint i |
| `log_Z` | log Z | (batch,) | Log partition function |

### Mathematical Notation

The edge potential (log-domain):
```
ψ̃(t, k, c, c') = (S_t,c - S_{t-k},c) + B_k,c + T_c',c
```

Forward recurrence:
```
α̃_t(c) = logsumexp_{k,c'} [ α̃_{t-k}(c') + ψ̃(t, k, c, c') ]
```

---

## Forward Pass Walkthrough

### Kernel Entry Point

From [triton_forward.py:72-343](torch-semimarkov/src/torch_semimarkov/streaming/triton_forward.py#L72-L343):

```python
@triton.jit
def semi_crf_streaming_scan_kernel(...):
```

### Initialization

```python
# Ring buffer initialized in launcher (triton_forward.py:640-641)
ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels
```

### Main Loop Structure

```python
for t in tl.range(1, T + 1):
    active = t <= seq_len
    alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)  # Accumulator

    for k in tl.range(1, tl.maximum(K, 2)):
        k_valid = (k <= t) & (k <= K - 1)
        start_pos = t - k
        ring_k_idx = start_pos % K

        # Load alpha_prev from ring buffer
        alpha_prev = tl.load(ring_base + ring_k_idx * stride_ring_k + ...)

        # Compute edge on-the-fly (see above)

        # Accumulate via logsumexp
        scores = alpha_prev[None, :] + edge_block
        score_for_k = logsumexp(scores, axis=1)  # Over c_src
        alpha_t = logsumexp_2way(alpha_t, score_for_k)  # Over k

    # Store to ring buffer
    ring_t_idx = t % K
    tl.store(ring_base + ring_t_idx * stride_ring_k + ..., alpha_t, ...)
```

### Logsumexp Reduction Pattern

The kernel uses a two-step logsumexp:
1. **Over source labels** (c_src): `logsumexp(scores, axis=1)`
2. **Over durations** (k): Accumulated incrementally via `logsumexp_2way`

```python
# Stable logsumexp accumulation
max_alpha = tl.maximum(alpha_t, score_for_k)
alpha_t = max_alpha + tl.log(
    tl.exp(alpha_t - max_alpha) + tl.exp(score_for_k - max_alpha)
)
```

---

## Checkpointing for Backward Pass

### Why Checkpointing?

The ring buffer only stores the K most recent α values. For the backward pass, we need α values at all positions. Options:
1. **Store all α**: O(TC) memory - defeats the purpose
2. **Recompute from scratch**: O(T²KC²) time - too slow
3. **Checkpoint + recompute**: O(T/S × KC) memory, O(TKC²) time

We use option 3 with checkpoint interval S = √(T×K).

### Checkpoint Storage

```python
ring_checkpoints: (batch, num_ckpts, K, C)
```

At each checkpoint position `i × S`, we save the entire ring buffer state.

### Checkpoint Interval Calculation

From [pytorch_reference.py:25-45](torch-semimarkov/src/torch_semimarkov/streaming/pytorch_reference.py#L25-L45):

```python
def _compute_checkpoint_interval(T: int, K: int) -> int:
    """Optimal interval minimizes total memory.

    Memory = (T/S) × K × C + S × C + K × C
    Taking d/dS = 0 gives S* = sqrt(T × K)
    """
    optimal = int(math.sqrt(T * K))
    return max(K, optimal)  # At least K
```

### Saving Checkpoints

From the forward kernel:

```python
should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
ckpt_idx = t // CHECKPOINT_INTERVAL
if should_checkpoint:
    for k_save in tl.range(0, K):
        ring_val = tl.load(ring_base + k_save * stride_ring_k + ...)
        tl.store(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_save * stride_ckpt_k + ...)
```

---

## Backward Pass Walkthrough

### Two-Phase Algorithm

From [triton_backward.py:42-555](torch-semimarkov/src/torch_semimarkov/streaming/triton_backward.py#L42-L555):

The backward pass processes segments in reverse order:

```python
for ckpt_idx in range(NUM_CKPTS - 1, -1, -1):
    seg_start = ckpt_idx * CHECKPOINT_INTERVAL
    seg_end = min((ckpt_idx + 1) * CHECKPOINT_INTERVAL, T)

    # Phase 1: Recompute alpha from checkpoint
    # Phase 2: Compute beta backward + accumulate gradients
```

### Phase 1: Alpha Recomputation

Load ring buffer state from checkpoint, then recompute forward through the segment:

```python
# Load from checkpoint
for k_slot in tl.range(0, K):
    alpha_val = tl.load(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_slot * stride_ckpt_k + ...)
    if k_slot == seg_start % K:
        tl.store(alpha_buf_base + 0 * stride_ab_t + ..., alpha_val)

# Recompute alpha for positions seg_start+1 to seg_end
for local_t in tl.range(1, SEGMENT_SIZE):
    t = seg_start + local_t
    if t < seg_end and t < seq_len:
        # Same forward computation as main kernel
        ...
```

### Phase 2: Beta Backward + Gradients

Compute beta backward while accumulating gradients:

```python
for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
    t = seg_end - 1 - t_offset
    if t >= seg_start and t < seq_len:
        alpha_t = tl.load(alpha_buf_base + local_t * stride_ab_t + ...)
        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        for k in tl.range(1, tl.maximum(K, 2)):
            end_pos = t + k
            if end_pos <= seq_len:
                beta_next = tl.load(beta_ring_base + (end_pos % K) * stride_br_k + ...)

                # Compute edge
                edge_block = ...

                # Compute marginal
                log_marginal = alpha_t[None, :] + edge_block + beta_next[:, None] - log_Z
                marginal = tl.exp(log_marginal)

                # Accumulate gradients
                ...

                # Update beta
                scores_for_beta = edge_block + beta_next[:, None]
                new_beta = logsumexp_2way(new_beta, logsumexp(scores_for_beta, axis=0))

        tl.store(beta_ring_base + (t % K) * stride_br_k + ..., new_beta)
```

---

## Gradient Semantics

### Per-Batch vs Shared Parameters

There are two types of parameters:

1. **Per-batch parameters** (cum_scores, proj_start, proj_end):
   - Shape includes batch dimension
   - Gradients scaled by `grad_output[batch_idx]` **inside** the kernel

2. **Shared parameters** (transition, duration_bias):
   - Shape does NOT include batch dimension
   - Accumulated per-batch **inside** kernel, scaled by `grad_output` **after** via einsum

### The Einsum Pattern

From [autograd.py:134-145](torch-semimarkov/src/torch_semimarkov/streaming/autograd.py#L134-L145):

```python
# Per-batch: scale each batch element
scale = grad_output.view(batch, 1, 1)
grad_cum_scores = grad_cum_scores * scale

# Shared: weighted sum via einsum
# grad_θ = Σ_b[grad_output[b] × grad_per_batch[b]]
grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

### Why This Matters

The difference matters when `grad_output` varies across batch elements (e.g., weighted losses, masked sequences):

```python
# Correct:
grad_θ = Σ_b[grad_output[b] × marginal_sum[b]]

# Wrong (buggy):
grad_θ = Σ_b[marginal_sum[b]] × Σ_b[grad_output[b]]
```

The einsum approach is also memory-efficient, avoiding large intermediate tensors.

---

## Numerical Considerations

### Float32 Requirement

Cumulative scores **must** be float32 for numerical stability at T > 100K:

```python
# In the kernel docstring warning:
# cum_scores MUST be float32 for numerical stability at T > 100K.
# Zero-centering before cumsum is critical to prevent precision loss.
```

### Zero-Centering

Before computing cumulative scores, zero-center the projected scores:

```python
projected = projected - projected.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
```

This prevents cumulative scores from growing too large and losing precision.

### NEG_INF Masking

Invalid positions are masked with `NEG_INF = -1e9` (not `-inf` to avoid NaN in gradients):

```python
# From constants.py
NEG_INF = -1e9

# Masking pattern in kernel
alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)
```

---

## Performance Characteristics

### When Streaming Beats Pre-computed Edges

Memory bandwidth is the bottleneck, not compute:
- **Pre-computed**: Load O(T×K×C²) edge tensor from memory
- **Streaming**: Compute edges from O(T×C) cumulative scores

For large K, streaming is faster even with the extra computation.

### Benchmark Comparison (NVIDIA L40S)

| Configuration | triton_scan (pre-computed) | streaming | Advantage |
|---------------|---------------------------|-----------|-----------|
| K=100, batch=64 | 127ms, 14GB | 38ms, 6MB | 3.35× faster, 2,393× less memory |
| K=500, batch=32 | 330ms, 35GB | 224ms, 3MB | 1.48× faster, 11,795× less memory |

### Batch Scaling

Streaming memory scales as O(batch × T × C), not O(batch × T × K × C²):
- Can process larger batches
- Linear scaling with batch size

---

## Debugging Tips

### Common Issues

1. **Shape mismatch with boundaries**: `proj_start` and `proj_end` must have shape `(batch, T, C)`, not `(batch, T+1, C)`

2. **Gradient NaN**: Usually caused by:
   - Not zero-centering before cumsum
   - Using float16 for cum_scores

3. **Wrong partition value**: Check that `cum_scores[:, 0, :]` is all zeros (the boundary condition)

### Verification Against Reference

Use the PyTorch reference for debugging:

```python
from torch_semimarkov.streaming.pytorch_reference import semi_crf_streaming_forward_pytorch

# Compare results
partition_triton, _, _ = launch_streaming_triton_kernel(...)
partition_ref, _, _ = semi_crf_streaming_forward_pytorch(...)

assert torch.allclose(partition_triton, partition_ref, atol=1e-4)
```
