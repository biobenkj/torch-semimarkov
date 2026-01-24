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

## Suspected Root Causes

### 1. Triton Atomic Operations
`triton_backward.py:504-544` uses `tl.atomic_add` for gradient accumulation. Within-batch iterations are sequential, but timing could cause issues.

### 2. Buffer Initialization
Ring buffers and checkpoints are allocated before kernel launch. No explicit `torch.cuda.synchronize()` between allocation and launch.

### 3. Unmasked Memory Access
Loading from invalid positions before mask is applied could read garbage.

### 4. Stochastic Batch Ordering
Different batch compositions expose different numerical edge cases.

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
    └── triton_backward.py         # Input clamping in Triton

benchmarks/practical_demonstration/timit/
└── timit_phoneme.py               # --crf-reg, --fixed-length, param logging
```

---

## Removing Debug Code

When the bug is fixed, these can be removed or kept as defensive programming:

**Keep (defensive):**

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

## Related Commits

- `8b399a8` - Roll back Triton backpointer to PyTorch
- `256db2a` - Add Triton backpointer (introduced issue?)
- `3c8deea` - Add PyTorch backpointer (pre-Triton, also has issue)
- Current session - All debugging instrumentation
