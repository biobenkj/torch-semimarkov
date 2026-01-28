# Sentinel: CRF Heads (nn.py)

**Verified against:** `src/torch_semimarkov/nn.py` @ commit `40fe66b`

**Linked tests:** `tests/test_semimarkov.py`, `tests/test_streaming_triton.py::TestTritonBasic`

## Summary

The `SemiMarkovCRFHead` class provides the user-facing API for semi-Markov CRF
sequence labeling. It wraps streaming/exact backends with:

1. **Automatic backend selection** based on memory heuristics
2. **Numerical stability** via zero-centering and float32 conversion
3. **NaN validation** at projection and cumsum stages
4. **Unified interface** for training (`compute_loss`) and inference (`decode`)

## Active Assumptions

### Mechanically Verified

These are verified automatically via `python3 verify-assumptions.py crf-heads`.

| ID | Assumption | Verification |
|----|------------|--------------|
| N1 | Zero-centering applied before cumsum | anchor: ZERO_CENTER |
| N2 | Float32 conversion for numerical stability | anchor: FLOAT32_CONVERT |
| N3 | NaN check after projection exists | anchor: NAN_CHECK_PROJECTION |
| N4 | NaN check after cumsum exists | anchor: NAN_CHECK_CUMSUM |
| N5 | Streaming forward called for partition | anchor: STREAMING_FORWARD_CALL |

### Agent-Verified (on trace load)

These require human/agent judgment when loading the trace.

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| N6 | Backend selection matches dispatch-overview.md | Compare `_select_backend` logic with dispatch-overview decision tree |
| N7 | T=1 skip for zero-centering documented | Check `if T > 1:` guard before zero-centering (~line 341) |
| N8 | Duration bias indexing uses k-1 | Verify `dur_idx = k - 1` in `_build_edge_tensor` (~line 215) |

## Algorithm Flow: forward()

1. **Input validation** (lines 300-302)
   - `validate_hidden_states()`, `validate_lengths()`, `validate_device_consistency()`

2. **Projection** (lines 307-309)
   ```python
   if self.projection is not None:
       scores = self.projection(hidden_states)  # (batch, T, C)
   ```

3. **NaN check after projection** (lines 314-319)
   - Detects gradient corruption from corrupted model parameters

4. **Backend selection** (lines 322-334)
   - `auto` → streaming vs exact based on edge tensor size
   - `streaming` → force streaming backend
   - `exact` → force exact backend via `SemiMarkov.logpartition`
   - `binary_tree_sharded` → memory-efficient reference implementation

5. **Cumulative scores** (lines 340-346)
   ```python
   scores_float = scores.float()  # Float32 for stability
   if T > 1:
       scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)  # Zero-center
   cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)
   ```

6. **NaN check after cumsum** (lines 349-353)
   - Detects extreme input values causing overflow

7. **Partition computation** (lines 355-372)
   - streaming → `semi_crf_streaming_forward()`
   - exact → `_forward_exact()` via `SemiMarkov.logpartition`
   - binary_tree_sharded → `_forward_binary_tree_sharded()`

## Backend Selection (_select_backend)

**Location:** lines 170-187

| Condition | Backend | Triton |
|-----------|---------|--------|
| Edge tensor > 8GB | streaming | use_triton param |
| Edge tensor <= 8GB | exact | False |
| Semiring not log/max | exact | False (error if OOM) |

**Edge tensor size formula:** `T * K * C * C * 4` bytes

```python
def _should_use_streaming(self, T: int) -> bool:
    edge_tensor_bytes = T * K * C * C * 4
    return edge_tensor_bytes > self.edge_memory_threshold  # default 8GB
```

## compute_loss() Flow

**Location:** lines 376-430

1. Validate labels via `validate_labels()`
2. Call `forward()` to get partition and cum_scores
3. Score gold segmentation via `_score_gold()` → `score_gold_vectorized()`
4. Return `partition - gold_score` with optional reduction

## decode() vs decode_with_traceback()

| Method | Line | Returns | Memory |
|--------|------|---------|--------|
| `decode()` | 461 | Viterbi score only | O(KC) |
| `decode_with_traceback()` | 539 | Score + segment list | O(TC) for backpointers |

**decode_with_traceback** uses:
- `semi_crf_streaming_viterbi_with_backpointers()` for backpointer computation
- `_traceback_from_backpointers()` for O(T) segment reconstruction

## Critical Invariants

- [ ] Zero-centering MUST be skipped for T=1 (single position has no variance)
- [ ] Float32 REQUIRED for T > 100K sequences (prevents cumsum overflow)
- [ ] NaN checks MUST come after projection and cumsum (early detection)
- [ ] Duration bias shape: (K, C) where index 0 = duration 1
- [ ] Transition matrix convention: `transition[i, j]` = FROM label i TO label j

## Numerical Guards

| Location | Guard | Purpose |
|----------|-------|---------|
| line 314 | `torch.isnan(scores).any()` | Detect corrupted projection weights |
| line 349 | `torch.isnan(cum_scores).any()` | Detect extreme input values |

## Entry Points

| Method | Line | Purpose | Returns |
|--------|------|---------|---------|
| `forward()` | 272 | Partition function | dict with `partition`, `cum_scores` |
| `compute_loss()` | 376 | NLL training loss | Tensor |
| `decode()` | 461 | Viterbi score | Tensor |
| `decode_with_traceback()` | 539 | Viterbi + segments | `ViterbiResult` |
| `_select_backend()` | 170 | Backend selection | `(backend_type, use_triton)` |
| `_score_gold()` | 445 | Gold segmentation score | Tensor |

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| T=1 mean zeros content | Medium | Skip zero-centering with `if T > 1:` guard |
| Triton backpointer disabled | Low | Uses PyTorch reference due to memory corruption issue |

## Version History

- **2026-01-28**: Initial trace @ commit `40fe66b`
