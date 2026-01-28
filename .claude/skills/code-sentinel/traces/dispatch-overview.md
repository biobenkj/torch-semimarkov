# Sentinel: Dispatch Overview

**Verified against:**
- `src/torch_semimarkov/streaming/autograd.py` @ commit `40fe66b`
- `src/torch_semimarkov/semimarkov.py` @ commit `40fe66b`

**Linked tests:** `tests/test_streaming_triton.py::TestDispatch`, `tests/test_semimarkov.py`

## Summary

The torch-semimarkov library provides two API families:

1. **Streaming API** (`semi_crf_streaming_forward`) - Computes edges on-the-fly from cumulative scores. Memory-efficient for long sequences. Entry point: `autograd.py:474`

2. **Non-Streaming API** (`SemiMarkov.logpartition`) - Operates on pre-computed edge tensors. Multiple algorithm variants. Entry point: `semimarkov.py:35`

**Note on Duration Indexing:** The two APIs currently use different indexing conventions by design (benchmarking phase):

| API               | Convention | Index Range | Example                      |
|-------------------|------------|-------------|------------------------------|
| **Streaming**     | 0-based    | 0 to K-1    | Duration k uses `edge[k-1]`  |
| **Non-streaming** | 1-based    | 1 to K-1    | Duration k uses `edge[k]`    |

The streaming (0-based) convention is the production target and will be adopted universally post-benchmarking.

## Active Assumptions

### Mechanically Verified

These are verified automatically via `python3 verify-assumptions.py dispatch-overview`.

| ID | Assumption | Verification |
|----|------------|--------------|
| D1 | K=1 dispatches to LinearCRFStreaming | anchor: K1_DISPATCH |
| D2 | K=2 dispatches to SemiCRFK2Streaming | anchor: K2_DISPATCH |
| D3 | K>=3 GPU checks can_use_triton | anchor: CAN_USE_TRITON |
| D4 | Triton path uses SemiCRFStreamingTriton | anchor: TRITON_DISPATCH |

### Agent-Verified (on trace load)

These require human/agent judgment when loading the trace.

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| D5 | K boundary conditions use correct comparisons | Inspect K==1, K==2, K>=3 conditionals in dispatch function (~lines 580-650) |

## Decision Tree: Streaming API

```mermaid
graph TD
    A[semi_crf_streaming_forward] --> B{K value?}
    B -->|K=1| C{boundary projections?}
    C -->|yes| D[Fall to K>=3 path]
    C -->|no| E{needs_grad?}
    E -->|yes| F[LinearCRFStreaming.apply]
    E -->|no| G{semiring?}
    G -->|max| H[linear_crf_viterbi_pytorch]
    G -->|log| I[linear_crf_forward_pytorch]

    B -->|K=2| J{boundary projections?}
    J -->|yes| D
    J -->|no| K{needs_grad?}
    K -->|yes| L[SemiCRFK2Streaming.apply]
    K -->|no| M{semiring?}
    M -->|max| N[semi_crf_k2_viterbi_pytorch]
    M -->|log| O[semi_crf_k2_forward_pytorch]

    B -->|K>=3| P{can_use_triton?}
    P -->|yes| Q{needs_grad?}
    Q -->|yes| R[SemiCRFStreamingTriton.apply]
    Q -->|no| S[launch_streaming_triton_kernel]
    P -->|no| T{needs_grad?}
    T -->|yes| U[SemiCRFStreaming.apply]
    T -->|no| V[semi_crf_streaming_forward_pytorch]

    F --> W[k1-linear-crf.md]
    H --> W
    I --> W
    L --> X[k2-fast-path.md]
    N --> X
    O --> X
    R --> Y[triton-forward-k3plus.md]
    S --> Y
    U --> Z[pytorch-forward-k3plus.md]
    V --> Z
```

## Decision Tree: Non-Streaming API

**Entry:** `SemiMarkov(semiring).logpartition()` at `semimarkov.py:35`

```mermaid
graph TD
    A[logpartition] --> B{use_banded?}
    B -->|yes| C[_dp_banded]
    B -->|no| D{use_linear_scan?}
    D -->|None auto| E{K*C > 200?}
    E -->|yes| F[linear_scan=True]
    E -->|no| G[linear_scan=False]
    D -->|explicit| H{value}
    F --> I{use_vectorized?}
    G --> J[_dp_binary_tree]
    H -->|True| I
    H -->|False| J
    I -->|yes| K[_dp_standard_vectorized]
    I -->|no| L[_dp_scan_streaming]

    C --> M[non-streaming-backends.md]
    J --> M
    K --> M
    L --> M
```

## Non-Streaming Algorithm Lookup

| Condition | Algorithm | Method | Trace |
|-----------|-----------|--------|-------|
| `use_banded=True` | Banded binary tree | `_dp_banded` | non-streaming-backends.md |
| `K*C <= 200` (auto) | Binary tree | `_dp_binary_tree` | non-streaming-backends.md |
| `K*C > 200`, vectorized | Vectorized scan | `_dp_standard_vectorized` | non-streaming-backends.md |
| `K*C > 200`, default | Ring buffer scan | `_dp_scan_streaming` | non-streaming-backends.md |

**Auto-selection:** `semimarkov.py:62-65`

```python
K_C_product = K * C
use_linear_scan = K_C_product > 200
```

## Streaming Dispatch Conditions

### can_use_triton (line 636)

```python
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda
```

- `HAS_TRITON`: Triton import succeeded (line 20-26)
- `use_triton`: User parameter (default True)
- `cum_scores.is_cuda`: Input is on GPU

### needs_grad (lines 573-579)

```python
needs_grad = (
    cum_scores.requires_grad
    or transition.requires_grad
    or duration_bias.requires_grad
    or (proj_start is not None and proj_start.requires_grad)
    or (proj_end is not None and proj_end.requires_grad)
)
```

## Streaming Lookup Table

| K | Device | Triton | needs_grad | Boundaries | Path | Trace |
|---|--------|--------|------------|------------|------|-------|
| 1 | any | any | yes | no | LinearCRFStreaming | k1-linear-crf.md |
| 1 | any | any | no | no | linear_crf_*_pytorch | k1-linear-crf.md |
| 1 | any | any | any | yes | K>=3 path | (see below) |
| 2 | any | any | yes | no | SemiCRFK2Streaming | k2-fast-path.md |
| 2 | any | any | no | no | semi_crf_k2_*_pytorch | k2-fast-path.md |
| 2 | any | any | any | yes | K>=3 path | (see below) |
| >=3 | GPU | yes | yes | any | SemiCRFStreamingTriton | triton-forward-k3plus.md |
| >=3 | GPU | yes | no | any | launch_streaming_triton_kernel | triton-forward-k3plus.md |
| >=3 | CPU | any | yes | any | SemiCRFStreaming | pytorch-forward-k3plus.md |
| >=3 | CPU | any | no | any | semi_crf_streaming_forward_pytorch | pytorch-forward-k3plus.md |

## Failure Mode Routing

### Streaming API Issues

| Symptom | Primary Trace | Secondary Trace | Check First |
|---------|---------------|-----------------|-------------|
| **NaN in loss** | triton-forward-k3plus.md | dispatch-overview.md | NEG_INF guards |
| **NaN in backward** | triton-backward-k3plus.md | pytorch-backward-k3plus.md | Partition validation (lines 88, 224) |
| **Wrong gradients** | triton-backward-k3plus.md | pytorch-backward-k3plus.md | Cross-reference outputs |
| **OOM on GPU** | triton-backward-k3plus.md | - | Recomputation Logic section |
| **K=1/K=2 mismatch** | k1-linear-crf.md / k2-fast-path.md | dispatch-overview.md | Dispatch conditions |
| **Triton vs PyTorch diff** | triton-forward-k3plus.md | pytorch-forward-k3plus.md | Ring buffer indexing |

### Non-Streaming API Issues

| Symptom | Primary Trace | Check First |
|---------|---------------|-------------|
| **OOM with large edge tensor** | non-streaming-backends.md | Use streaming API instead |
| **Slow binary tree** | non-streaming-backends.md | Check K*C > 200 threshold |
| **Algorithm mismatch** | non-streaming-backends.md | Verify all algorithms produce same result |
| **Duration index off-by-one** | non-streaming-backends.md | Verify `dur_idx = k - 1` |

## Entry Points

### Streaming API (autograd.py)

| Function | Line | Called When |
|----------|------|-------------|
| `semi_crf_streaming_forward()` | 474 | Public API entry point |
| `LinearCRFStreaming.apply()` | 590 | K=1, needs_grad, no boundaries |
| `SemiCRFK2Streaming.apply()` | 616 | K=2, needs_grad, no boundaries |
| `SemiCRFStreamingTriton.apply()` | 642 | K>=3, GPU, Triton, needs_grad |
| `SemiCRFStreaming.apply()` | 655 | K>=3, no Triton, needs_grad |
| `launch_streaming_triton_kernel()` | 669 | K>=3, GPU, Triton, inference |
| `semi_crf_streaming_forward_pytorch()` | 683 | K>=3, CPU, inference |

### Non-Streaming API (semimarkov.py)

| Function | Line | Called When |
|----------|------|-------------|
| `SemiMarkov.logpartition()` | 35 | Public API entry point |
| `_dp_binary_tree()` | 85 | K*C <= 200 (auto) or explicit |
| `_dp_scan_streaming()` | 150 | K*C > 200, default |
| `_dp_standard()` | 230 | Legacy reference implementation |
| `_dp_standard_vectorized()` | 277 | K*C > 200, use_vectorized=True |
| `_dp_blocktriangular()` | 326 | Block-triangular variant |
| `_dp_banded()` | 503 | use_banded=True |

## Critical Invariants

### Duration Indexing (divergent during benchmarking)

- [ ] **Streaming**: `dur_idx = k - 1` for duration value `k` (0-based, production target)
- [ ] **Non-streaming**: `dur_idx = k` for duration value `k` (1-based, legacy)
- [ ] `duration_bias` shape: `(K, C)` - index semantics differ by API
- [ ] Edge tensor last dims: `(c_dst, c_src)` - destination first (consistent)

### Streaming API

- [ ] K>=3 required for ring buffer architecture (Triton kernel constraint)
- [ ] K=1/K=2 fast paths do NOT support boundary projections
- [ ] Boundary projections force fallback to K>=3 path even for K=1/K=2
- [ ] `cum_scores` MUST be float32 for numerical stability

### Non-Streaming API

- [ ] Auto-selection threshold: `K*C > 200` triggers linear scan
- [ ] All algorithms produce identical partition values (numerical precision aside)
- [ ] Ring buffer in `_dp_scan_streaming` uses `(head - (dur - 1)) % K`

## Numerical Guards

| Location | Guard | Purpose |
|----------|-------|---------|
| autograd.py:88-95 | `torch.isfinite(partition)` | Validate partition before PyTorch backward |
| autograd.py:224-231 | `torch.isfinite(partition)` | Validate partition before Triton backward |
| autograd.py:339-345 | `torch.isfinite(partition)` | Validate partition before K=1 backward |
| autograd.py:431-437 | `torch.isfinite(partition)` | Validate partition before K=2 backward |
| autograd.py:115-121 | `torch.isfinite(grad_cum_scores)` | Validate PyTorch backward output |
| autograd.py:254-265 | `torch.isfinite(grad_*)` | Validate Triton backward outputs |

## Known Issues

| Issue | Severity | Frequency | Resolution | Commit |
|-------|----------|-----------|------------|--------|
| K=1/K=2 ring buffer aliasing | Critical | Always with K<3 | Dispatch to specialized paths | `870bd1f` |
| Boundaries force K>=3 path | Medium | When boundaries used with K<3 | Document; specialized paths don't support boundaries | - |

## Version History

- **2026-01-28**: Documented dual indexing scheme divergence (streaming 0-based vs non-streaming 1-based)
- **2026-01-28**: Added non-streaming backends (`semimarkov.py`) decision tree, algorithm lookup, entry points, and failure routing @ commit `40fe66b`
- **2026-01-27**: Initial trace (streaming API only) @ commit `40fe66b`
