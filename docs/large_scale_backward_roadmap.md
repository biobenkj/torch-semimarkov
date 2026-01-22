# Large-Scale Semi-CRF Backward Pass Roadmap

Target dimensions: T=400,000+, K=3,000+, C=16-24

## Problem Statement

The current backward implementations have memory issues at scale:

| Current Implementation | Memory for T=400K, K=3K, C=24 |
|------------------------|-------------------------------|
| `SemiCRFTritonBackward` (full α) | 38 MB/batch |
| `SemiCRFOptimizedCheckpointedBackward` (S=√T) | **182 MB/batch** (worse!) |
| Ring buffer only (forward) | 288 KB/batch |

The "optimized" checkpointing is actually a **pessimization** for large K because:
- Checkpoint interval S = √T = 632 → 633 checkpoints
- Each checkpoint stores K×C = 72K floats
- Total: 633 × 72K = 45.6M floats = 182 MB

## Solution: Adaptive Checkpoint Interval

### Mathematical Derivation

Memory M = (checkpoints) + (segment buffer) + (β ring)
        = (T/S)×K×C + S×C + K×C

Minimizing over S:
- dM/dS = -TKC/S² + C = 0
- S* = √(T×K)

Optimal memory: M* ≈ 2×√(T×K)×C + K×C

For T=400K, K=3K, C=24:
- S* = √(400K × 3K) = 34,641
- Checkpoints: 400K/35K ≈ 11.5 checkpoints × 72K floats = 3.3 MB
- Segment buffer: 35K × 24 = 3.4 MB
- β ring: 72K floats = 288 KB
- **Total: ~7 MB/batch** (vs 182 MB with current formula)

## Implementation Phases

### Phase 1: Fix Checkpoint Interval Formula ✅ QUICK WIN
**Goal**: Change interval from √T to √(T×K)

File: `checkpointed.py`
```python
def _compute_checkpoint_interval(T: int, K: int = 1) -> int:
    # OLD: return max(K, int(math.sqrt(T)))
    # NEW: Optimize for both T and K
    return max(K, int(math.sqrt(T * K)))
```

Expected improvement: 182 MB → 7 MB for target dimensions.

### Phase 2: Pure Recomputation Kernel (Minimal Memory)
**Goal**: O(K×C) memory with O(2T) compute

Memory budget: 2 × K × C = 576 KB/batch (just two ring buffers)

Algorithm:
```
Forward pass:
  - Stream α with ring buffer
  - Save ONLY log_Z (partition value)

Backward pass (fused recomputation):
  - Maintain α ring buffer (recomputing forward)
  - Maintain β ring buffer (computing backward)
  - Challenge: α and β go opposite directions!

Solution: Two-pass backward
  Pass 1: Compute all β values backward, store in ring buffer checkpoints
          at interval S = √(T×K), memory = √(T/K) × K × C
  Pass 2: Recompute α forward, compute gradients using stored β checkpoints
```

Actually, for truly minimal memory, we need a different approach...

### Phase 2 (Alternative): Segment-Synchronized Recomputation
**Goal**: O(K×C) memory, O(T) compute per segment

For each segment of size K (going backward):
1. We have β ring buffer from previous segments
2. Recompute α for this segment (need α from K positions before segment start)
3. Compute gradients for this segment
4. Update β ring buffer

Challenge: To recompute α for segment [iK, (i+1)K), we need α[(i-1)K : iK].
This requires either:
- Storing α at segment boundaries: T/K × C = 133 × 24 = 3.2K floats = 13 KB ✓
- Or recomputing from start: O(T × T/K) = O(T²/K) compute ✗

**Hybrid approach**:
- Store α (not full ring buffer, just α values) at every K positions
- Checkpoint memory: (T/K) × C = 133 × 24 × 4 = 13 KB
- Working memory: 2 × K × C = 576 KB
- **Total: ~600 KB/batch**

Wait, this doesn't work because to compute α[t] we need α[t-1], α[t-2], ..., α[t-K+1].
So we need the full ring buffer state, not just one α value.

### Phase 2 (Revised): Optimal Checkpointing is the Answer

After analysis, the optimal approach IS checkpointing with the correct interval:
- Checkpoint interval: S = √(T×K)
- Store full ring buffer (K×C) at each checkpoint
- Memory: √(T/K) × K × C + √(T×K) × C ≈ 2×√(T×K)×C

For target dimensions: ~7 MB/batch with O(T) compute.

The "pure recomputation" (2× compute) approach would save memory but:
- Still needs ~3 MB for β storage during the backward-then-forward passes
- Doubles compute time
- Complexity not worth the marginal memory savings

**Recommendation**: Implement Phase 1 (fix interval formula) and test.
This alone should reduce memory from 182 MB to 7 MB.

### Phase 3: Triton Kernel Optimization
**Goal**: Fuse operations, optimize memory access patterns

- Optimize the segment forward/backward kernels for large K
- Consider block-parallel over C dimension if C > 32
- Profile and optimize memory bandwidth

### Phase 4: Testing at Scale
**Goal**: Verify correctness and performance at target dimensions

Tests:
- [ ] Gradient correctness: `torch.autograd.gradcheck` with float64
- [ ] Memory profiling: Verify actual memory usage matches predictions
- [ ] Compute benchmarks: Compare approaches
- [ ] Numerical stability: Test with extreme values

Test configurations:
- Small: T=1000, K=100, C=8 (quick validation)
- Medium: T=10000, K=500, C=16 (intermediate)
- Large: T=100000, K=1000, C=24 (stress test)
- Target: T=400000, K=3000, C=24 (production)

### Phase 5: Code Cleanup
**Goal**: Remove redundant implementations, simplify module structure

Current files:
- `backward.py` - PyTorch reference + non-checkpointed Triton
- `checkpointed.py` - Multiple checkpointing approaches
- `triton_backward.py` - Re-export module

After cleanup:
- `backward.py` - PyTorch reference (for testing only)
- `triton_backward.py` - Single production Triton implementation
- Remove: Basic checkpointing (O(T^1.5)), non-checkpointed Triton (O(T×C) memory)

Keep only:
1. PyTorch reference (for gradient verification)
2. Optimized checkpointed Triton (production)

## Memory Budget Summary

| Component | Formula | T=400K, K=3K, C=24 |
|-----------|---------|-------------------|
| Ring buffer checkpoints | √(T/K) × K × C | 3.3 MB |
| Segment α buffer | √(T×K) × C | 3.4 MB |
| β ring buffer | K × C | 288 KB |
| **Total** | **~2×√(T×K)×C + K×C** | **~7 MB** |

Compare to:
- Full α storage: T × C = 38 MB
- Current "optimized": √T × K × C = 182 MB

## Timeline

1. **Phase 1** (immediate): Fix checkpoint interval - 1 hour
2. **Phase 2** (if needed): Further optimization - 1 day
3. **Phase 3** (optimization): Triton tuning - 2-3 days
4. **Phase 4** (validation): Scale testing - 1 day
5. **Phase 5** (cleanup): Code consolidation - 1 day

## Open Questions

1. **Batch size**: What batch sizes are typical? Memory scales linearly with batch.

2. **Mixed precision**: Would float16 be acceptable? Would halve memory.

3. **Gradient accumulation**: For very large batches, could accumulate gradients
   over micro-batches to reduce peak memory.

4. **Sequence length variation**: If T varies significantly within a batch,
   should we compute optimal S per-sequence or use max T?

---

*Created: 2025-01-18*
