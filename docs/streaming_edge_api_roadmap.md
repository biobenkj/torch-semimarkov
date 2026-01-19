# Streaming Edge API Roadmap

## Overview

This document outlines the implementation plan for a memory-efficient Semi-CRF that computes edge potentials on-the-fly from encoder features, enabling processing of very long sequences (T=400K+) with large segment durations (K=3K+) on a single GPU.

**Target Architecture**: Mamba SSM Encoder → Semi-CRF Decoder for genomic sequence annotation

## Problem Statement

### Current Memory Requirements

The current API requires a pre-computed edge tensor:

```python
edge = torch.Tensor  # shape: (batch, T-1, K, C, C)
partition = semi_crf_triton_forward(edge, lengths)
```

For target dimensions (T=400K, K=3K, C=24, batch=1):

| Component | Size |
|-----------|------|
| Edge tensor | (400K × 3K × 24 × 24 × 4 bytes) = **2.76 TB** |
| Single L40S GPU | 48 GB |

**The edge tensor alone is 57× larger than GPU memory.**

### Why This Happens

The edge tensor represents all possible segment transitions:
- `edge[b, t, k, c_dest, c_src]` = score for transitioning from label `c_src` at position `t` to label `c_dest` at position `t+k`
- This is O(T × K × C²) values

## Solution: Pre-Projected Streaming (Golden Rule)

Instead of pre-computing the full edge tensor OR doing matmuls inside the kernel,
we **pre-project features to label space** before the scan.

### Critical Insight: The "Golden Rule"

**Original streaming idea (FLAWED):**
```python
# Inside kernel - BAD: matmul in sequential scan = latency disaster
segment_feat = cum_features[t+k] - cum_features[t]  # (D,)
segment_score = segment_feat @ label_weights         # D×C matmul per (t,k)!
```

**Golden Rule (CORRECT):**
```python
# Outside kernel - parallel, fast
projected = encoder_output @ label_weights  # (B, T, D) @ (D, C) → (B, T, C)
cum_projected = cumsum(projected)           # (B, T+1, C), MUST be float32!

# Inside kernel - just vector subtraction, no matmuls
segment_score = cum_projected[t+k] - cum_projected[t]  # (C,) subtract only
```

This removes all matrix multiplications from the sequential scan.

### Architectural Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mamba SSM Encoder                         │
│  Input: (batch, T, 4) one-hot DNA                                │
│  Output: (batch, T, D) hidden states                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Pre-Projection (Outside Kernel, Parallel)           │
│                                                                  │
│  Option A (Basic): Content-only scoring                          │
│    projected = h @ W_content           # (B, T, C)               │
│    cum_scores = cumsum(projected)      # (B, T+1, C) in float32! │
│                                                                  │
│  Option B (Genomics): Content + Boundary scoring                 │
│    proj_start = h @ W_start            # (B, T, C)               │
│    proj_end = h @ W_end                # (B, T, C)               │
│    cum_content = cumsum(h @ W_content) # (B, T+1, C) in float32! │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            Semi-CRF Streaming Forward (Triton Kernel)            │
│  For each position t, for each duration k:                       │
│                                                                  │
│    Option A: segment_score = cum[t+k] - cum[t]                   │
│                                                                  │
│    Option B: segment_score = proj_start[t]                       │
│                            + proj_end[t+k-1]                     │
│                            + (cum_content[t+k] - cum_content[t]) │
│                                                                  │
│    edge[c_dest, c_src] = segment_score[c_dest]                   │
│                        + transition[c_src, c_dest]               │
│    Continue DP scan...                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Comparison

| Component | Pre-computed Edge | Golden Rule (Basic) | Golden Rule (Boundaries) |
|-----------|-------------------|---------------------|--------------------------|
| Edge tensor | 2.76 TB | **0** | **0** |
| Encoder hidden states | - | (not stored) | (not stored) |
| Projected cumsum (T+1, C) | - | 38 MB | 38 MB |
| Boundary projections (T, C) | - | 0 | 77 MB |
| Transition matrix | - | 2 KB | 2 KB |
| Ring buffer (α) | 288 KB | 288 KB | 288 KB |
| Checkpoints | ~7 MB | ~7 MB | ~7 MB |
| **Total** | **~2.76 TB** | **~50 MB** | **~125 MB** |

**Memory reduction: 55,000× (basic) or 22,000× (with boundaries)**

### Compute Comparison

| Metric | Pre-computed Edge | Original Streaming | Golden Rule |
|--------|-------------------|-------------------|-------------|
| Inner loop per (t,k) | 1 scalar load | 2×D loads + D×C matmul | 2-4×C loads + C ops |
| FLOPs per (t,k) | ~10 | 6,144 | ~50 |
| Viable at scale? | ❌ (memory) | ❌ (latency) | ✅ |

### Numerical Stability: Two Critical Issues

#### Issue 1: Catastrophic Cancellation (dtype)

**CRITICAL**: Cumulative sums MUST use float32 to avoid precision loss.

```python
# BAD: cumsum in float16/bfloat16
cum = cumsum(projected.half())  # cum[400K] ≈ 400K, precision lost!

# GOOD: cumsum in float32, cast result after subtraction
cum = cumsum(projected.float())                    # (T+1, C) in float32
segment_score = (cum[t+k] - cum[t]).to(dtype)     # Subtract first, then cast
```

After T=400K positions, cumulative values may reach ~10⁵ magnitude. Subtracting
two such values in float16 (5 significant digits) loses all information about
small differences. Float32 (7 significant digits) maintains precision.

#### Issue 2: Floating Point Erasure (magnitude) - CRITICAL FOR T=400K

**Even float32 is insufficient** if projected scores are consistently positive/negative!

**The Problem:**
- float32 has 23 bits of significand (precision ~10⁻⁷)
- If projected scores average +1.0 per position, after T=400K: `cum ≈ 400,000`
- Machine epsilon at magnitude 400K: `400K × 10⁻⁷ ≈ 0.04`
- **Any signal smaller than 0.04 is completely erased!**

**The Fix: Zero-Centering Before Cumsum**

```python
# In MambaSemiCRF.forward, BEFORE cumsum:
content_projected = h @ self.W_content  # (B, T, C)

# CRITICAL: Center to keep cumsum magnitude bounded (random walk scaling)
content_projected = content_projected - content_projected.mean(dim=1, keepdim=True)

# Now cumsum grows as √T (random walk) instead of T
cum_scores = torch.cumsum(content_projected.float(), dim=1)
# At T=400K: magnitude ≈ √400K ≈ 632, epsilon ≈ 6e-5 (1000× better precision!)
```

**Why this works:**
- Centered scores form a random walk with expected magnitude √T
- At T=400K: √T ≈ 632, machine epsilon ≈ 6×10⁻⁵
- Preserves signals down to ~10⁻⁴ magnitude (vs 0.04 without centering)

**Test case to add:**
```python
def test_long_sequence_drift():
    """Verify gradient reaches 'needle in haystack' feature at T=350K."""
    T = 400_000
    # Set positive weights (triggers drift without centering)
    # Inject distinctive feature at t=350_000
    # Assert gradient at t=350K is non-zero
```

---

## API Design

### New Pre-Projected Streaming API

```python
def semi_crf_streaming_forward(
    # Pre-projected inputs (Golden Rule: projection done OUTSIDE kernel)
    cum_scores: torch.Tensor,           # (batch, T+1, C) - cumsum of ZERO-CENTERED projected features, FLOAT32
    transition: torch.Tensor,           # (C, C)
    duration_bias: torch.Tensor,        # (K, C) - REQUIRED: compensates for sum-pooling length bias

    # Existing inputs
    lengths: torch.Tensor,              # (batch,)
    K: int,                             # max segment duration

    # Options
    semiring: Semiring = LogSemiring,   # Any semiring from torch_semimarkov.semirings

    # Optional: boundary features for genomics
    proj_start: Optional[torch.Tensor] = None,  # (batch, T, C) - start boundary scores
    proj_end: Optional[torch.Tensor] = None,    # (batch, T, C) - end boundary scores
) -> torch.Tensor:
    """
    Compute Semi-CRF partition function with pre-projected streaming.

    This uses the "Golden Rule" optimization: features are projected to label
    space BEFORE the kernel, eliminating matmuls from the sequential scan.

    Edge potentials are computed on-the-fly as:
        content_score = cum_scores[t+k] - cum_scores[t]  # (C,) - just subtraction!
        segment_score = content_score + duration_bias[k] (+ boundaries if provided)
        edge[c_dest, c_src] = segment_score[c_dest] + transition[c_src, c_dest]

    Args:
        cum_scores: Cumulative sum of ZERO-CENTERED projected encoder outputs.
            Computed as: cumsum((h @ W - mean) .float())
            MUST be float32 and zero-centered to avoid precision loss at T=400K!
            Shape: (batch, T+1, C)
        transition: Label transition scores. Shape: (C, C)
        duration_bias: Duration-specific label bias. REQUIRED because sum-pooling
            creates implicit length bias (variance scales with k). Without this,
            the model cannot properly compare short-strong vs long-weak segments.
            Shape: (K, C)
        lengths: Sequence lengths. Shape: (batch,)
        K: Maximum segment duration.
        semiring: Any semiring from torch_semimarkov.semirings.
            LogSemiring for partition/marginals, MaxSemiring for Viterbi, etc.
        proj_start: Start boundary scores (genomics). Shape: (batch, T, C)
        proj_end: End boundary scores (genomics). Shape: (batch, T, C)

    Returns:
        Result depends on semiring (partition, entropy, etc). Shape: (batch,) or (ssize, batch).
    """
```

### Integration with Mamba Encoder

```python
class MambaSemiCRF(nn.Module):
    """End-to-end Mamba encoder + Semi-CRF decoder with Golden Rule optimization.

    Incorporates all critical numerical stability fixes for T=400K+ sequences:
    1. Zero-centering before cumsum (prevents floating-point erasure)
    2. Float32 cumsum (prevents catastrophic cancellation)
    3. Duration bias (compensates for sum-pooling length variance)
    """

    def __init__(self, vocab_size, hidden_dim, num_labels, max_duration,
                 use_boundaries=True):
        super().__init__()
        self.encoder = MambaEncoder(vocab_size, hidden_dim)
        self.max_duration = max_duration
        self.use_boundaries = use_boundaries
        self.num_labels = num_labels

        # Projection weights (applied OUTSIDE the kernel)
        self.W_content = nn.Parameter(torch.randn(hidden_dim, num_labels) * 0.01)
        self.transition = nn.Parameter(torch.zeros(num_labels, num_labels))

        # Duration bias: REQUIRED to compensate for sum-pooling variance
        # Initialize to slightly prefer shorter segments (common in genomics)
        self.duration_bias = nn.Parameter(torch.zeros(max_duration, num_labels))

        if use_boundaries:
            self.W_start = nn.Parameter(torch.randn(hidden_dim, num_labels) * 0.01)
            self.W_end = nn.Parameter(torch.randn(hidden_dim, num_labels) * 0.01)

    def forward(self, x, lengths, semiring=LogSemiring):
        # Encode: (batch, T, vocab) -> (batch, T, hidden)
        h = self.encoder(x)  # (B, T, D)
        batch, T, D = h.shape

        # === Golden Rule: Project OUTSIDE the kernel ===
        content_projected = h @ self.W_content  # (B, T, C)

        # CRITICAL: Zero-center before cumsum to prevent floating-point erasure
        # Without this, signals at t > T/2 can be completely lost at T=400K!
        content_centered = content_projected - content_projected.mean(dim=1, keepdim=True)

        # Cumsum in float32 (prevents catastrophic cancellation)
        cum_scores = torch.zeros(batch, T + 1, self.num_labels,
                                 device=h.device, dtype=torch.float32)
        cum_scores[:, 1:, :] = torch.cumsum(content_centered.float(), dim=1)

        # Boundary scores (if using)
        proj_start = h @ self.W_start if self.use_boundaries else None
        proj_end = h @ self.W_end if self.use_boundaries else None

        # Semi-CRF forward - NO MATMULS inside kernel!
        result = semi_crf_streaming_forward(
            cum_scores=cum_scores,
            transition=self.transition,
            duration_bias=self.duration_bias,
            lengths=lengths,
            K=self.max_duration,
            semiring=semiring,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        return result

    def neg_log_likelihood(self, x, lengths, gold_segments):
        """Compute NLL = -log P(gold | x) = log_partition - gold_score."""
        partition = self.forward(x, lengths, semiring=LogSemiring)
        gold_score = self._score_gold(x, gold_segments)
        return (partition - gold_score).mean()

    def decode(self, x, lengths):
        """Viterbi decoding using MaxSemiring."""
        return self.forward(x, lengths, semiring=MaxSemiring)

    def entropy(self, x, lengths):
        """Compute entropy of segmentation distribution for uncertainty."""
        return self.forward(x, lengths, semiring=EntropySemiring)
```

### Backward Pass (Gradient Flow)

With the Golden Rule, gradients flow through the pre-projection:

```python
# Gradient flow (conceptual):
#
# 1. Kernel computes marginals: marginal[t, k, c_dest, c_src]
#
# 2. Gradient to cum_scores (efficient: no atomics needed!)
#    For each t: grad_cum_scores[t] receives contributions from:
#      - Segments STARTING at t: negative contribution (subtracted)
#      - Segments ENDING at t: positive contribution (added)
#    This can be computed as a reverse cumsum of marginal sums.
#
# 3. Gradient to W_content:
#    grad_W_content = h.T @ grad_cum_scores  # (D, C)
#
# 4. Gradient to boundary weights (if used):
#    grad_W_start = h.T @ (marginals summed over segments starting at each t)
#    grad_W_end = h.T @ (marginals summed over segments ending at each t)
#
# 5. Gradient to encoder (h):
#    grad_h = grad_cum_scores @ W_content.T  # Flows back through projection
#           + grad_start @ W_start.T         # (if using boundaries)
#           + grad_end @ W_end.T

# Key insight from reviewer: Use reverse cumsum, NOT atomic adds!
# grad_h[t] = reverse_cumsum(grad_cum_scores)[t]
```

### Aggregated Marginals for Clinical Uncertainty

The existing semiring framework (see `torch_semimarkov.semirings`) already provides:
- `LogSemiring`: Gradients w.r.t. edge give full marginal probabilities
- `EntropySemiring`: Entropy of segmentation distribution
- `KLDivergenceSemiring`: Compare two model distributions

For clinical applications, we need **aggregated** marginals that are easy to visualize
and interpret. These are computed from the full edge marginals (which are ephemeral
during backward) with minimal memory overhead:

```python
def compute_aggregated_marginals(edge_marginals, lengths):
    """
    Aggregate edge marginals into position-level uncertainty metrics.

    Args:
        edge_marginals: Full marginals from LogSemiring backward.
            Shape: (batch, T-1, K, C, C)
        lengths: Sequence lengths. Shape: (batch,)

    Returns:
        marginal_start: P(segment of label c starts at t). Shape: (batch, T, C)
        marginal_end: P(segment of label c ends at t). Shape: (batch, T, C)
        marginal_label: P(position t has label c). Shape: (batch, T, C)
    """
    batch, T_minus_1, K, C, _ = edge_marginals.shape
    T = T_minus_1 + 1

    # marginal_start[t, c] = Σ_{k, c'} marginal[t, k, c, c']
    # "Sum over all segments of label c that START at position t"
    marginal_start = edge_marginals.sum(dim=(2, 4))  # (batch, T-1, C)

    # marginal_end[t, c] = Σ_{k, c'} marginal[t-k, k, c, c']
    # "Sum over all segments of label c that END at position t"
    # This requires a scatter-add or careful indexing
    marginal_end = compute_end_marginals(edge_marginals, lengths)  # (batch, T, C)

    # marginal_label[t, c] = Σ_{s≤t<s+k} marginal[s, k, c, :]
    # "Sum over all segments that COVER position t with label c"
    # Efficient: marginal_label = cumsum(marginal_start) - cumsum(marginal_end shifted)
    marginal_label = (
        torch.cumsum(marginal_start, dim=1) -
        torch.cumsum(marginal_end[:, :-1, :], dim=1)
    )

    return marginal_start, marginal_end, marginal_label
```

**Memory-efficient approach for streaming backward:**

During the backward pass, we compute marginals ephemerally for gradient updates.
We can aggregate on-the-fly without storing the full (T, K, C, C) tensor:

```python
# Inside backward kernel, when computing marginal[t, k, c_dest, c_src]:
p = exp(alpha[t, c_src] + edge[t,k,c_dest,c_src] + beta[t+k, c_dest] - log_Z)

# Aggregate immediately (atomic add to small buffers):
marginal_start[t, c_dest] += p.sum(over c_src)      # (T, C) buffer
marginal_end[t + k, c_dest] += p.sum(over c_src)    # (T, C) buffer
# Skip marginal_label in kernel - compute outside via cumsum of start/end
```

**Clinical visualization:**

```
Position:      0    100   200   300   400   500 ...
              ─────────────────────────────────────
Exon prob:    ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓░░░░
Intron prob:  ▓▓▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓
              ─────────────────────────────────────
Start conf:        ▲ (0.95)              ▲ (0.72)
End conf:                   ▲ (0.98)          ▲ (0.85)

Legend: ▓ = high probability, ░ = low probability
        ▲ = boundary with confidence score
```

This enables clinicians to see:
1. **Boundary confidence**: "Is the exon start at position 105 or 107?"
2. **Label uncertainty**: "How sure are we this region is an exon vs UTR?"
3. **Fuzzy boundaries**: Gradual transitions indicate uncertain boundaries

---

## Implementation Phases

### Phase 0: Micro-Benchmark (Recommended First Step)

**Goal**: Validate that the Golden Rule kernel will be fast enough before full implementation.

**Rationale**: Your colleague correctly suggests benchmarking the inner loop before committing
to the full implementation. This takes ~1 hour and prevents wasted effort.

**Benchmark**:
```python
@triton.jit
def inner_loop_microbenchmark(
    cum_scores_ptr,    # (T, C)
    transition_ptr,    # (C, C)
    T: tl.constexpr,
    C: tl.constexpr,
):
    """Measure: 2 vector loads + subtract + add transition."""
    c_idx = tl.arange(0, C)
    transition = tl.load(transition_ptr + ...)  # (C, C) - load once

    for t in tl.range(1, T):
        for k in tl.range(1, 32):  # Sample K range
            # This is what runs T × K times
            cum_end = tl.load(cum_scores_ptr + t * C + c_idx)
            cum_start = tl.load(cum_scores_ptr + (t - k) * C + c_idx)
            segment_score = cum_end - cum_start  # (C,)
            edge_block = segment_score[:, None] + transition  # (C, C)
```

**Success criteria**:
- Inner loop latency < 50 cycles per (t, k) iteration → viable
- Inner loop latency > 200 cycles → need Block-K tiling optimization

**Estimated effort**: 2-4 hours

---

### Phase 1: PyTorch Reference Implementation

**Goal**: Working Golden Rule forward/backward in pure PyTorch for correctness verification.

**Files**:
- `src/torch_semimarkov/streaming.py` (new)

**Tasks**:
- [ ] Implement `semi_crf_streaming_forward_pytorch()` with pre-projected inputs
- [ ] Implement `semi_crf_streaming_backward_pytorch()` (compute marginals + gradients)
- [ ] Create `SemiCRFStreaming` autograd Function
- [ ] Add gradient tests with `torch.autograd.gradcheck`
- [ ] Test numerical stability with T=100K+ in float32

**Golden Rule edge computation (inside forward loop)**:
```python
def compute_edge_block_golden_rule(cum_scores, transition, t, k,
                                    proj_start=None, proj_end=None):
    """Compute edge[t, k, :, :] - NO MATMULS, just vector ops."""
    # Content score via cumsum difference: (batch, C)
    content_score = cum_scores[:, t + k, :] - cum_scores[:, t, :]

    # Add boundary scores if provided
    segment_score = content_score
    if proj_start is not None:
        segment_score = segment_score + proj_start[:, t, :]
    if proj_end is not None:
        segment_score = segment_score + proj_end[:, t + k - 1, :]

    # Add transition to get full edge block: (batch, C, C)
    edge_block = segment_score.unsqueeze(-1) + transition.unsqueeze(0)

    return edge_block  # (batch, C, C)
```

**Estimated effort**: 2-3 days

---

### Phase 2: Triton Forward Kernel

**Goal**: GPU-accelerated streaming forward pass.

**Files**:
- `src/torch_semimarkov/triton_streaming.py` (new)

**Tasks**:
- [ ] Port `semi_crf_scan_kernel` to use streaming edge computation
- [ ] Handle cumulative feature memory access patterns
- [ ] Optimize for memory coalescing (features are accessed with stride K)
- [ ] Add ring buffer checkpointing for backward pass
- [ ] Benchmark against PyTorch reference

**Kernel modifications**:
```python
@triton.jit
def semi_crf_streaming_scan_kernel(
    # Feature inputs (replaces edge_ptr)
    cum_features_ptr,   # (batch, T+1, D)
    label_weights_ptr,  # (D, C) or (K, D, C)
    transition_ptr,     # (C, C) or (K, C, C)

    # Outputs
    ring_ptr,           # ring buffer
    out_ptr,            # partition

    # ... rest similar to current kernel
):
    # Inside the main loop, instead of loading edge:
    for k in tl.range(1, K):
        # Load segment features
        feat_start = tl.load(cum_features_ptr + (t - k) * stride_fd + ...)
        feat_end = tl.load(cum_features_ptr + t * stride_fd + ...)
        segment_feat = feat_end - feat_start  # (D,)

        # Compute segment scores
        segment_scores = tl.dot(segment_feat, label_weights)  # (C,)

        # Build edge block
        edge_block = segment_scores[:, None] + transition  # (C, C)

        # Continue with existing DP logic...
```

**Challenges**:
- Memory access pattern: cumulative features indexed by (t-k) and (t)
- May need to tile over hidden dimension D if D > shared memory
- Label weights matrix multiply inside the kernel

**Estimated effort**: 3-4 days

---

### Phase 3: Triton Backward Kernel with Gradient Accumulation

**Goal**: GPU-accelerated backward pass with gradients to features.

**Files**:
- `src/torch_semimarkov/triton_streaming.py` (extend)

**Tasks**:
- [ ] Modify checkpointed backward to compute gradients w.r.t. features
- [ ] Accumulate gradients to `cumulative_features` efficiently
- [ ] Accumulate gradients to `label_weights` and `transition`
- [ ] Handle gradient checkpointing for memory efficiency
- [ ] Verify gradients match PyTorch reference

**Gradient accumulation strategy**:
```python
# For each (t, k, c_dest, c_src) with marginal probability p:
#
# grad_cum_features[t] -= p * label_weights[:, c_dest]
# grad_cum_features[t+k] += p * label_weights[:, c_dest]
# grad_label_weights[:, c_dest] += p * segment_feat[t, k]
# grad_transition[c_src, c_dest] += p

# Efficient approach: accumulate in chunks, use atomics or reduction
```

**Estimated effort**: 4-5 days

---

### Phase 4: Enhanced Scoring (Duration + Boundary Contrast)

**Goal**: Improve segment scoring for genomics applications with:
1. Duration-dependent scoring (exons have typical length distributions)
2. Boundary contrast features (splice sites are defined by *transitions*, not positions)

#### 4A: Duration-Dependent Scoring

**Motivation**: Different labels have different typical durations. Exons: 50-300bp, Introns: 100-10,000bp.

**Tasks**:
- [ ] Extend API to accept `(K, C, C)` duration-dependent transitions
- [ ] Add low-rank factorization option to prevent overfitting: `transition[k] = base + U[k] @ V[k].T`
- [ ] Update kernels to use duration-indexed transitions

**Note**: `duration_bias` (K, C) is now in Phase 2 (core) since it's required for sum-pooling.

#### 4B: Boundary Contrast Features

**Motivation**: A segment boundary is defined by a *change* in signal (Intron→Exon), not
just the absolute signal at a position. Current `proj_start = h[t] @ W_start` relies on
Mamba to encode "this is a transition point" into a single vector h[t].

**Enhancement**: Explicitly model the transition signal:

```python
# Instead of just h[t], use the contrast between h[t-1] and h[t]
# This gives the model direct access to the "derivative" of the encoder signal

class MambaSemiCRF(nn.Module):
    def __init__(self, ...):
        ...
        # Boundary contrast weights (Phase 4 enhancement)
        self.use_contrast = True
        if self.use_contrast:
            # Input: [h[t-1], h[t]] concatenated → 2*D dimensions
            self.W_start_contrast = nn.Parameter(
                torch.randn(2 * hidden_dim, num_labels) * 0.01
            )
            self.W_end_contrast = nn.Parameter(
                torch.randn(2 * hidden_dim, num_labels) * 0.01
            )

    def forward(self, x, lengths, semiring=LogSemiring):
        h = self.encoder(x)  # (B, T, D)

        if self.use_contrast:
            # Boundary features include contrast with neighboring position
            # h_contrast[t] = [h[t-1], h[t]] for start boundaries
            h_padded = F.pad(h, (0, 0, 1, 0))  # Pad with zeros at start
            h_contrast_start = torch.cat([h_padded[:, :-1, :], h], dim=-1)  # (B, T, 2D)
            proj_start = h_contrast_start @ self.W_start_contrast

            # For end boundaries: [h[t], h[t+1]]
            h_padded_end = F.pad(h, (0, 0, 0, 1))
            h_contrast_end = torch.cat([h, h_padded_end[:, 1:, :]], dim=-1)
            proj_end = h_contrast_end @ self.W_end_contrast
        else:
            proj_start = h @ self.W_start
            proj_end = h @ self.W_end

        # ... rest of forward pass
```

**Why this matters for splice sites:**
- Splice donor: GT dinucleotide at exon/intron boundary
- Splice acceptor: AG dinucleotide at intron/exon boundary
- These are *transitions* - the contrast feature captures exactly this

**Estimated effort**: 2-3 days

---

### Phase 5: Integration Layer

**Goal**: Easy-to-use module for Mamba → Semi-CRF pipeline.

**Files**:
- `src/torch_semimarkov/modules.py` (new or extend)

**Tasks**:
- [ ] Create `SemiCRFDecoder` nn.Module
- [ ] Create `MambaSemiCRF` end-to-end module (optional)
- [ ] Add Viterbi decoding for inference
- [ ] Add segment sampling
- [ ] Documentation and examples

**Module design**:
```python
class SemiCRFDecoder(nn.Module):
    """Semi-CRF decoder layer for sequence labeling with segments."""

    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        max_duration: int,
        duration_dependent: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.max_duration = max_duration

        if duration_dependent:
            self.label_weights = nn.Parameter(
                torch.randn(max_duration, hidden_dim, num_labels) * 0.01
            )
            self.transition = nn.Parameter(
                torch.zeros(max_duration, num_labels, num_labels)
            )
        else:
            self.label_weights = nn.Parameter(
                torch.randn(hidden_dim, num_labels) * 0.01
            )
            self.transition = nn.Parameter(
                torch.zeros(num_labels, num_labels)
            )

    def forward(
        self,
        encoder_output: torch.Tensor,  # (batch, T, hidden_dim)
        lengths: torch.Tensor,          # (batch,)
    ) -> torch.Tensor:
        """Compute log partition function."""
        # Compute cumulative features
        batch, T, D = encoder_output.shape
        cum_features = torch.zeros(batch, T + 1, D, device=encoder_output.device)
        cum_features[:, 1:, :] = torch.cumsum(encoder_output, dim=1)

        return semi_crf_streaming_forward(
            cumulative_features=cum_features,
            label_weights=self.label_weights,
            transition=self.transition,
            lengths=lengths,
        )

    def neg_log_likelihood(
        self,
        encoder_output: torch.Tensor,
        lengths: torch.Tensor,
        gold_segments: List[List[Tuple[int, int, int]]],  # [(start, end, label), ...]
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        partition = self.forward(encoder_output, lengths)
        gold_score = self._score_gold(encoder_output, gold_segments)
        return (partition - gold_score).mean()

    def decode(
        self,
        encoder_output: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[List[Tuple[int, int, int]]]:
        """Viterbi decoding to find best segmentation."""
        # Use max semiring forward + backtracking
        ...
```

**Estimated effort**: 2-3 days

---

### Phase 6: Testing and Validation

**Goal**: Comprehensive testing at scale.

**Tasks**:
- [ ] Unit tests for streaming forward/backward
- [ ] Gradient correctness tests (`gradcheck` with float64)
- [ ] Numerical stability tests (extreme values, long sequences)
- [ ] Memory profiling at target dimensions (T=400K, K=3K, C=24)
- [ ] Performance benchmarks vs. theoretical compute
- [ ] Integration test with actual Mamba encoder

**Test configurations**:
```python
SMALL_CONFIGS = [
    {"T": 100, "K": 10, "C": 4, "D": 32},
    {"T": 500, "K": 50, "C": 8, "D": 64},
]

MEDIUM_CONFIGS = [
    {"T": 10_000, "K": 500, "C": 16, "D": 128},
    {"T": 50_000, "K": 1000, "C": 20, "D": 256},
]

LARGE_CONFIGS = [
    {"T": 100_000, "K": 1000, "C": 24, "D": 256},
    {"T": 400_000, "K": 3000, "C": 24, "D": 256},  # Target
]
```

**Estimated effort**: 2-3 days

---

## Technical Details

### Cumulative Feature Computation

The cumulative sum enables O(1) segment feature extraction:

```python
# Encoder output: h[t] for t in 0..T-1
encoder_output = mamba(x)  # (batch, T, D)

# Cumulative sum with leading zeros
# cum[0] = 0
# cum[t] = Σᵢ₌₀^(t-1) h[i] for t in 1..T
cum_features = F.pad(encoder_output.cumsum(dim=1), (0, 0, 1, 0))  # (batch, T+1, D)

# Segment feature for [t, t+k):
segment_feat = cum_features[:, t+k, :] - cum_features[:, t, :]
# = Σᵢ₌ₜ^(t+k-1) h[i]
```

### Numerical Stability

The streaming computation must maintain numerical stability:

1. **Segment scores**: May need layer normalization on segment features
2. **Transition scaling**: Initialize transition near zero
3. **Log-space arithmetic**: Use logsumexp for all reductions
4. **Gradient clipping**: May need for very long sequences

```python
# Stable segment score computation
segment_feat = cum_features[:, t+k, :] - cum_features[:, t, :]
segment_feat = F.layer_norm(segment_feat, [hidden_dim])  # Optional normalization
segment_scores = segment_feat @ label_weights
```

### Memory Access Patterns in Triton

The streaming kernel has different access patterns than the pre-computed edge kernel:

**Pre-computed edge**:
- Access: `edge[t, k, :, :]` - contiguous (C, C) blocks
- Stride: edge is (T, K, C, C), accessing along T dimension

**Streaming features**:
- Access: `cum_features[t]` and `cum_features[t+k]`
- Two loads per (t, k) pair instead of one
- But D << K×C×C typically, so net memory reduction

**Optimization strategies**:
1. Prefetch cumulative features for next few positions
2. Cache label_weights in shared memory (small: D×C)
3. Cache transition in shared memory (small: C×C)

### Gradient Checkpointing for Backward

The backward pass needs both α and β values to compute marginals. With streaming:

1. **Forward pass**:
   - Compute α using streaming edge
   - Save ring buffer checkpoints (same as current)
   - Save cumulative features (already in memory from encoder)

2. **Backward pass**:
   - Recompute α within segments using streaming edge
   - Compute β backward using streaming edge
   - Compute marginals and accumulate gradients

The gradient targets:
```
∂L/∂cum_features[t] = Σ_{k} (marginal[t-k:t, k, :, :].sum() × (-W))
                    + Σ_{k} (marginal[t:t+k, k, :, :].sum() × W)
```

This requires careful accumulation to avoid race conditions in parallel processing.

---

## File Structure After Implementation

```
src/torch_semimarkov/
├── __init__.py
├── triton_scan.py          # Existing: forward kernel (pre-computed edge)
├── backward.py             # Existing: PyTorch backward reference
├── checkpointed.py         # Existing: checkpointed backward
├── streaming.py            # NEW: PyTorch streaming reference
├── triton_streaming.py     # NEW: Triton streaming kernels
├── modules.py              # NEW: nn.Module wrappers
└── triton_backward.py      # Existing: re-exports

tests/
├── test_streaming.py       # NEW: streaming API tests
├── test_streaming_gradients.py  # NEW: gradient correctness
└── ...

benchmarks/
├── test_scale_streaming.py # NEW: scale tests for streaming
├── benchmark_mamba_crf.py  # NEW: end-to-end benchmark
└── ...
```

---

## Timeline Summary

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | PyTorch reference streaming | 2-3 days | None |
| 2 | Triton forward kernel | 3-4 days | Phase 1 |
| 3 | Triton backward kernel | 4-5 days | Phase 2 |
| 4 | Duration-dependent scoring | 2 days | Phase 3 |
| 5 | Integration layer | 2-3 days | Phase 3 |
| 6 | Testing and validation | 2-3 days | All phases |

**Total estimated effort: 15-20 days**

---

## Open Questions

1. **Hidden dimension size**: What D will the Mamba encoder output? Larger D = more expressive but slower kernel.

2. **Duration-dependent vs. independent**: Do you need different scores for different segment lengths, or is a single projection sufficient?

3. **Segment feature pooling**: Is sum-pooling (via cumulative sum) sufficient, or do you need attention-based pooling over segments?

4. **Viterbi decoding**: Do you need the best segmentation path, or just the partition function for training?

5. **Mixed precision**: Would float16 features with float32 accumulation be acceptable for training?

---

## Next Steps

1. Review this roadmap and provide feedback
2. Answer open questions above
3. Start Phase 1: PyTorch reference implementation
4. Iterate based on testing results

---

*Created: 2025-01-18*
*Target: torch-semimarkov v2.0 with streaming edge API*
