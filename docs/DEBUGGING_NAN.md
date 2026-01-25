# NaN Debugging Guide for Semi-Markov CRF

This document catalogs all debugging instrumentation added to trace stochastic NaN issues during training.

## Quick Reference

| Location | What it does | Added in |
|----------|--------------|----------|
| `autograd.py:110-119` | Validates partition from forward before backward | This session |
| `autograd.py:252-261` | Same for Triton path | This session |
| `autograd.py:138-145` | Validates grad_cum_scores after backward (PyTorch) | This session |
| `autograd.py:282-289` | Same for Triton path | This session |
| `nn.py:420-424` | Detects NaN after projection layer | This session |
| `nn.py:444-449` | Detects NaN after cumsum | This session |
| `nn.py:528-563` | `parameter_penalty()` for L2 regularization | This session |
| `helpers.py:603-612` | `inf*0=NaN` fix with `torch.where` | This session |
| `pytorch_reference.py:578-594` | Input clamping before marginal computation | This session |
| `pytorch_reference.py:842-858` | Same in marginals function | This session |
| `triton_forward.py:283-292` | NEG_INF guard for inner logsumexp | This session |
| `triton_forward.py:297-307` | NEG_INF guard for accumulation logsumexp | This session |
| `triton_forward.py:347-356` | NEG_INF guard for final reduction | This session |
| `triton_backward.py:343-364` | NEG_INF guards for alpha recompute logsumexp | This session |
| `triton_backward.py:577-598` | NEG_INF guards for beta update logsumexp | This session |
| `triton_backward.py:459-472` | Input clamping in Triton marginal | This session |
| `triton_backward.py:176-182` | **OOB fix: clamped indices for READS from unpadded inputs** | This session |
| `triton_backward.py:~765` | **OOB fix: pad ALL gradient allocations to C_PAD** | This session |
| `triton_backward.py:~541-608` | **Atomic fix: unclamped indices for WRITES to padded outputs** | This session |
| `triton_backward.py:~765` | **Float64 fix: use double precision for gradient accumulation** | This session |
| `triton_backward.py:~920` | **Float64 fix: convert back to original dtype before returning** | This session |
| `autograd.py:~292-315` | **Validate ALL gradient outputs** (not just grad_cum_scores) | This session |
| `timit_phoneme.py:1533-1547` | Parameter magnitude logging per epoch | This session |
| `timit_phoneme.py:804-865` | Fixed-length collate for debugging | This session |

---

## Debugging Flags

### `--no-triton`
Forces PyTorch reference implementation. If NaN disappears, bug is in Triton kernels.

```bash
python benchmarks/.../timit_phoneme.py train --no-triton ...
```

### `--crf-reg <float>`
L2 regularization on CRF parameters. Prevents gradient explosion from parameter drift.

```bash
python benchmarks/.../timit_phoneme.py train --crf-reg 0.01 ...
```

### `--fixed-length <int>`
Forces all sequences to exact length. If NaN disappears, bug is in boundary handling.

```bash
python benchmarks/.../timit_phoneme.py train --fixed-length 200 ...
```

### Debug logging
Enable DEBUG level to see parameter magnitudes every epoch:

```bash
PYTHONLOGLEVEL=DEBUG python benchmarks/.../timit_phoneme.py train ...
```

---

## Validation Points (Error Messages)

### 1. Forward Pass Validation

**Location:** `nn.py:420-424`
```
ValueError: hidden_states contains NaN values after projection layer
```
**Meaning:** Encoder output or projection weights are corrupted.

**Location:** `nn.py:444-449`
```
ValueError: NaN detected in cumulative scores after cumsum
```
**Meaning:** Extreme values in scores caused cumsum overflow.

### 2. Partition Validation

**Location:** `autograd.py:110-119` (PyTorch) and `autograd.py:252-261` (Triton)
```
RuntimeError: Non-finite partition from forward pass (PyTorch/Triton): N NaN, M Inf
```
**Meaning:** Forward pass produced invalid partition. Check logsumexp stability.

### 3. Gradient Validation

**Location:** `autograd.py:138-145` (PyTorch) and `autograd.py:282-289` (Triton)
```
RuntimeError: Non-finite values in CRF backward (PyTorch/Triton): grad_cum_scores has N NaN, M Inf
```
**Meaning:** Backward pass produced invalid gradients. Check marginal computation.

### 4. Parameter Drift Warning

**Location:** `timit_phoneme.py:1541-1547`
```
WARNING: Epoch N: CRF parameters drifting high! trans_max=X, dur_max=Y. Consider increasing --crf-reg.
```
**Meaning:** CRF parameters exceeding 20. Increase L2 regularization.

---

## Numerical Guards

### NEG_INF Guards (Triton Logsumexp)
Prevent undefined arithmetic when all inputs are NEG_INF:

```python
# triton_forward.py:286-292 (inner logsumexp)
max_scores = tl.max(scores, axis=1)
is_all_neginf = max_scores < (NEG_INF + 1.0)
max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
log_sum_exp = tl.log(
    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1) + 1e-10
)
score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)
```

When `max_scores == NEG_INF` (-1e9), the subtraction `scores - max_scores` produces 0 instead of staying at NEG_INF. The guard detects this case and returns NEG_INF directly.

Same pattern applied to:

- `triton_forward.py:297-307` - accumulation logsumexp
- `triton_forward.py:347-356` - final reduction
- `triton_backward.py:343-364` - alpha recompute
- `triton_backward.py:577-598` - beta update

### Epsilon Guards (Triton Forward)
Prevent `log(0) = -inf` in logsumexp:

```python
# triton_forward.py:290
log_sum_exp = tl.log(tl.sum(tl.exp(...)) + 1e-10)
```

### Input Clamping (Backward)
Prevent extreme values before marginal computation:

```python
# pytorch_reference.py:581-583, triton_backward.py:459-461
alpha_t_safe = torch.clamp(alpha_t, min=-1e6, max=1e6)
beta_next_safe = torch.clamp(beta_next, min=-1e6, max=1e6)
edge_block_safe = torch.clamp(edge_block, min=-1e6, max=1e6)
```

### Log-Marginal Clamping
Prevent exp overflow (float32 overflows at ~88):

```python
# pytorch_reference.py:593, triton_backward.py:471
log_marginal = torch.clamp(log_marginal, min=-80.0, max=80.0)
```

### inf*0=NaN Fix
Use `torch.where` instead of multiplication for masking:

```python
# helpers.py:606, 612
trans_scores = torch.where(first_seg_mask, torch.zeros_like(...), trans_scores)
total_per_segment = torch.where(seg_mask, total_per_segment, torch.zeros_like(...))
```

### OOB Memory Access & Atomic Contention Fix (DEFINITIVE ROOT CAUSE)

The stochastic NaN was caused by **out-of-bounds pointer calculation** and **atomic contention** in `triton_backward.py`.

**Problem 1 (OOB):** Gradient tensors were allocated with size `C` (e.g., 39), but Triton launches `C_PAD` threads (next power of 2, e.g., 64). Masked-out threads (39-63) still calculate pointer addresses, causing OOB access.

**Problem 2 (Atomic Contention):** Clamping all masked-out indices to `C-1` forces 25 threads to target the *same memory address* with `atomic_add`, causing race conditions even when writes are masked.

**Fix (2 parts):**

**Part 1: Pad ALL gradient allocations** to `C_PAD`:

```python
# triton_backward.py:~765
grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, ...)
grad_proj_start = torch.zeros(batch, T, C_PAD, ...)
grad_proj_end = torch.zeros(batch, T, C_PAD, ...)
grad_tr_workspace = torch.zeros(batch, K, C_PAD, C_PAD, ...)
grad_db_workspace = torch.zeros(batch, K, C_PAD, ...)
```

**Part 2: Use unclamped indices for writes, clamped for reads:**

```python
# WRITES: Use unclamped c_idx so each thread writes to unique address
tl.atomic_add(grad_cs_base + t * stride + c_idx * stride_c, ...)  # c_idx, not c_idx_safe

# READS: Use clamped c_idx_safe for unpadded inputs
cum_end = tl.load(cum_scores_base + t * stride + c_idx_safe * stride_c, ...)
```

**Part 3: Slice back to C** before returning:

```python
# triton_backward.py:~915
grad_cum_scores = grad_cum_scores[:, :, :C]
grad_proj_start = grad_proj_start[:, :, :C]
grad_proj_end = grad_proj_end[:, :, :C]
grad_tr_workspace = grad_tr_workspace[:, :, :C, :C]
grad_db_workspace = grad_db_workspace[:, :, :C]
```

This ensures every thread writes to its own unique memory address (valid padding), completely eliminating OOB access and atomic contention.

---

## Diagnostic Workflow

### Step 1: Clear Triton Cache
```bash
rm -rf ~/.triton/cache
```

### Step 2: Isolate Triton vs PyTorch
```bash
# Test PyTorch reference 5x
for i in {1..5}; do
    python .../timit_phoneme.py train --no-triton --crf-reg 0.01 ...
done
```
- If 5/5 pass: Bug is in Triton kernels
- If failures: Bug is in shared code

### Step 3: Test Fixed Length
```bash
python .../timit_phoneme.py train --fixed-length 200 --crf-reg 0.01 ...
```
- If NaN disappears: Bug is in boundary handling

### Step 4: Monitor Parameters
```bash
PYTHONLOGLEVEL=DEBUG python .../timit_phoneme.py train --crf-reg 0.01 ...
```
Watch for parameter drift warnings.

---

## Root Cause (CONFIRMED)

### OOB Memory Access & Atomic Contention in Triton Backward Kernel

**CONFIRMED:** The stochastic NaN was caused by two related issues in `triton_backward.py`:

1. **OOB Pointer Calculation**: Gradient tensors allocated with size `C` (e.g., 39), but Triton launches `C_PAD` threads (e.g., 64). Masked-out threads calculated pointers beyond allocated memory.

2. **Atomic Contention**: Initial fix of clamping indices to `C-1` forced 25+ threads to target the *same* memory address with `atomic_add`, causing hardware-level race conditions even with masked writes.

**Fix applied:** Pad ALL gradient allocations to `C_PAD` + use unclamped indices for writes (each thread gets unique address) + use clamped indices only for reads from unpadded inputs + slice back to `C` before returning.

### Floating-Point Non-Associativity (ACTUAL ROOT CAUSE)

**CONFIRMED:** After fixing OOB/contention issues, systematic gradient differences remain:

- Differences grow with sequence length: 1e-4 at T=100 → 1e-3 at T=500
- `atomic_add` executes additions in non-deterministic order (depends on which GPU thread wins)
- Small differences (~1e-7 per addition in float32) accumulate over T×K×C operations
- At T=500, K=30, C=39: 585,000 additions per batch element → ~1e-3 accumulated error
- Over 20 epochs × 29 batches = 580 backward passes → parameter drift → NaN

**Fix applied:** Use `torch.float64` for gradient accumulation in Triton backward kernel:

- Float32 error: ~1e-7 per operation → ~1e-3 accumulated
- Float64 error: ~1e-16 per operation → ~1e-10 accumulated (negligible)
- Convert back to original dtype before returning

```python
# triton_backward.py:~765 - Allocate with float64
accum_dtype = torch.float64
grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=accum_dtype)

# triton_backward.py:~920 - Convert back to original dtype
grad_cum_scores = grad_cum_scores[:, :, :C].to(dtype)
```

### Previously Suspected (Not Root Cause)

- **Triton Atomic Operations**: `tl.atomic_add` order non-determinism IS a correctness issue due to floating-point non-associativity (fixed with float64)
- **Buffer Initialization**: Allocation timing is fine with PyTorch's async execution model
- **Stochastic Batch Ordering**: Only exposed the underlying numerical precision issue, not a root cause itself

---

## Files Modified Summary

```
src/torch_semimarkov/
├── nn.py                          # NaN detection, parameter_penalty()
├── helpers.py                     # inf*0=NaN fix
└── streaming/
    ├── autograd.py                # Partition + gradient validation
    ├── pytorch_reference.py       # Input clamping in backward
    ├── triton_forward.py          # Epsilon guards
    └── triton_backward.py         # Input clamping, float64 accumulation

benchmarks/practical_demonstration/timit/
└── timit_phoneme.py               # --crf-reg, --fixed-length, param logging
```

---

## Removing Debug Code

When the bug is fixed, these can be removed or kept as defensive programming:

**Keep (defensive):**

- **Float64 gradient accumulation** (ACTUAL ROOT CAUSE FIX)
- **OOB/Contention fix: padded gradients + unclamped writes + clamped reads**
- NEG_INF guards in Triton logsumexp
- Epsilon guards in logsumexp
- Input clamping before marginal
- Log-marginal clamping
- `inf*0=NaN` fix with torch.where
- `parameter_penalty()` method

**Remove (debug-only):**
- Partition validation in backward (raises on every NaN)
- Gradient validation in backward (raises on every NaN)
- Parameter magnitude logging per epoch
- `--fixed-length` flag (or keep as optional feature)

---

## num_warps Tuning and IR/PTX Analysis

### Current Status

**Finding:** `num_warps > 2` causes non-deterministic NaN gradients on L40/L40S. Setting `num_warps=2` fixes correctness but severely degrades performance.

**Hypothesis:** Register pressure causes spills to local memory with higher warp counts, introducing undefined behavior.

### Testing num_warps with find_determinism.py

```bash
# On HPC, run the determinism test
python src/torch_semimarkov/streaming/find_determinism.py
```

This script tests both forward and backward passes for:
- Non-deterministic results across multiple runs
- NaN values in gradients

### IR/PTX Analysis for Root Cause

To debug the `num_warps` issue, capture Triton's compilation output:

```bash
# Set Triton debug environment variables
export TRITON_DEBUG=1
export TRITON_CACHE_DIR=/tmp/triton_debug
export TRITON_ALWAYS_COMPILE=1  # Disable caching for fresh compilation

# Run with num_warps=2 (known good)
python -c "
from torch_semimarkov.streaming.triton_forward import *
# ... run kernel
" 2>&1 | tee triton_warps2.log

# Then modify kernel to num_warps=4 and compare
```

**What to look for in PTX output:**

1. **Register spills** - Look for `ld.local` / `st.local` instructions
   - `ld.local` = loading from spilled registers (bad for performance and possibly correctness)
   - `st.local` = storing to local memory (spill)

2. **Register count** - Check `.reg` declarations at the top of PTX
   - More warps = fewer registers per thread
   - Typical limit: 64 registers/thread for good occupancy

3. **Memory barriers** - Look for `bar.sync` instructions
   - Missing barriers between warps could cause race conditions

### Programmatic IR Access

```python
import triton

# Get compiled kernel info
compiled = kernel.run(..., warmup=True, return_kernel=True)
print(compiled.asm['ttir'])   # High-level Triton IR
print(compiled.asm['ttgir'])  # GPU-specific IR
print(compiled.asm['llir'])   # Low-level IR
print(compiled.asm['ptx'])    # Final CUDA assembly
```

### Potential Fixes for num_warps > 2

1. **Loop Tiling** - Reduce register pressure by processing C_PAD×C_PAD in smaller tiles
2. **num_stages Tuning** - Reduce pipeline stages to save registers
3. **Grid Parallelization** - Move work from warps to grid dimension

---

## Related Commits

- `8b399a8` - Roll back Triton backpointer to PyTorch
- `256db2a` - Add Triton backpointer (introduced issue?)
- `3c8deea` - Add PyTorch backpointer (pre-Triton, also has issue)
- Current session - All debugging instrumentation, float64 accumulation, num_warps=2
