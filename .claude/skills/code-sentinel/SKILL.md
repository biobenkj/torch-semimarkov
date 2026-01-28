---
name: code-sentinel
description: >
  Expert system for torch-semimarkov. Manages persistent execution traces ("Sentinels")
  for Triton/PyTorch backends.

  MANDATORY: Invoke this skill BEFORE debugging NaN/Inf, shape mismatches,
  modifying autograd logic, or making commits.

  Actions:
  1. Run `./sentinel.py status` to check overall health
  2. Run `./sentinel.py verify --trace <name>` before providing debugging advice
  3. Run `./sentinel.py pipeline` before committing changes
  4. Locate active backend via `dispatch-overview.md`
  5. Load specific trace into context and verify against current git HEAD

  Triggers: "debug semi-markov", "trace execution", "explain backend dispatch",
  "Triton kernel debug", "NaN", "shape mismatch", "gradient flow", "sentinel",
  "verify code", "check consistency", "pre-commit", "retrace"
allowed-tools: Read, Grep, Glob, Bash
---

# Code Sentinel

Persistent execution traces ("Sentinels") for torch-semimarkov backends. Prevents hallucinations about code execution paths by maintaining verified baseline documentation.

## Quick Start

```bash
# Check overall health
./sentinel.py status

# Scaffold a new trace from source
./sentinel.py init src/torch_semimarkov/streaming/new_module.py

# Verify before debugging
./sentinel.py verify --trace triton-forward-k3plus

# Auto-detect and fix anchor drift
./sentinel.py retrace triton-forward-k3plus --auto --apply

# Run full pre-commit pipeline
./sentinel.py pipeline

# Install git hooks
./sentinel.py install-hooks
```

## Verification Protocol (ALWAYS EXECUTE FIRST)

### Step 1: Run Verification

```bash
./sentinel.py verify --trace <trace-name>
```

This performs:
- Commit staleness check (verified hash vs current HEAD)
- Uncommitted changes detection
- Anchor verification (pattern -> line number)
- Assumption verification (mechanical checks)

### Step 2: Interpret Output

**On success (all verified):**
```
Grounded: triton-forward-k3plus @ 40fe66b ✓
```

**On failure (drift detected):**
```
Grounded: triton-forward-k3plus @ 40fe66b
  Commit: STALE (COMMITTED_CHANGES)
    Verified: 40fe66b | Current: abc1234
  Anchors: 5/7 verified, 2 failed
    ✗ RING_BUFFER_WRITE: ANCHOR_DRIFT: Expected ~320, found 345 (drift 25 > 20)
  Assumptions: A1 ✓, A2 ✗, A3 ✓

⚠️  Cannot provide advice until sentinel is updated.
```

### Step 3: Remediate if Needed

If stale, run diff to synchronize understanding:
```bash
./sentinel.py retrace triton-forward-k3plus --diff-only
```

If anchors drifted, auto-update line numbers:
```bash
./sentinel.py retrace triton-forward-k3plus --anchors-only
```

## Pre-Commit Pipeline

Run before every commit:

```bash
./sentinel.py pipeline

# Or with automatic test execution
SENTINEL_RUN_TESTS=1 ./sentinel.py pipeline
```

Pipeline steps:
1. **Consistency check** - meta ↔ anchors ↔ traces alignment
2. **Trace verification** - all sentinels verified
3. **Test advisory** - suggest tests for changed files

## Backend Selection Tree

```
semi_crf_streaming_forward() [autograd.py:474]
|
+-- K == 1 (no boundary projections)
|   +-- needs_grad: LinearCRFStreaming.apply() [line 590]
|   +-- inference: linear_crf_forward_pytorch() [line 605]
|   --> Trace: k1-linear-crf.md
|
+-- K == 2 (no boundary projections)
|   +-- needs_grad: SemiCRFK2Streaming.apply() [line 616]
|   +-- inference: semi_crf_k2_forward_pytorch() [line 631]
|   --> Trace: k2-fast-path.md
|
+-- K >= 3
    +-- GPU + HAS_TRITON + use_triton
    |   +-- needs_grad: SemiCRFStreamingTriton.apply() [line 642]
    |   +-- inference: launch_streaming_triton_kernel() [line 669]
    |   --> Trace: triton-forward-k3plus.md
    |
    +-- CPU or no Triton
        +-- needs_grad: SemiCRFStreaming.apply() [line 655]
        +-- inference: semi_crf_streaming_forward_pytorch() [line 683]
        --> Trace: pytorch-forward-k3plus.md
```

## Failure Mode Routing

| Symptom | Primary Trace | Check First |
|---------|---------------|-------------|
| **NaN in loss** | triton-forward-k3plus.md | NEG_INF guards |
| **NaN in backward** | triton-backward-k3plus.md | Partition validation |
| **Wrong gradients** | triton-backward-k3plus.md | Cross-reference outputs |
| **OOM on GPU** | triton-backward-k3plus.md | Recomputation Logic section |
| **K=1/K=2 mismatch** | k1-linear-crf.md | Dispatch conditions |
| **Triton vs PyTorch diff** | triton-forward-k3plus.md | Ring buffer indexing |

## Available Traces

| Trace | Purpose | Source File |
|-------|---------|-------------|
| [dispatch-overview.md](traces/dispatch-overview.md) | Backend selection | autograd.py |
| [triton-forward-k3plus.md](traces/triton-forward-k3plus.md) | Triton forward (K>=3) | triton_forward.py |
| [triton-backward-k3plus.md](traces/triton-backward-k3plus.md) | Triton backward | triton_backward.py |
| [pytorch-forward-k3plus.md](traces/pytorch-forward-k3plus.md) | PyTorch forward | pytorch_reference.py |
| [pytorch-backward-k3plus.md](traces/pytorch-backward-k3plus.md) | PyTorch backward | pytorch_reference.py |
| [k1-linear-crf.md](traces/k1-linear-crf.md) | K=1 fast path | pytorch_reference.py |
| [k2-fast-path.md](traces/k2-fast-path.md) | K=2 fast path | pytorch_reference.py |
| [non-streaming-backends.md](traces/non-streaming-backends.md) | Non-streaming API | semimarkov.py |
| [autograd-kernel-interface.md](traces/autograd-kernel-interface.md) | Autograd contract | autograd.py |
| [crf-heads.md](traces/crf-heads.md) | User-facing CRF API | nn.py |

## CLI Reference

| Command | Description |
|---------|-------------|
| `./sentinel.py status` | Show overall sentinel health |
| `./sentinel.py init SOURCE` | Scaffold a new trace from source file |
| `./sentinel.py init SOURCE --name NAME` | Scaffold with custom trace name |

**Note on `init` scaffolding:** The `init` command intentionally provides only ~17% of a completed trace (structure + line refs). The ~83% gap represents **irreducible domain knowledge**—explaining *why* code works, *what* invariants matter, and *which* patterns are numerically critical. This is by design: sentinels derive value from encoding understanding that source code alone does not express. Do NOT attempt to auto-generate this content.
| `./sentinel.py verify --trace NAME` | Verify specific trace |
| `./sentinel.py verify --all` | Verify all traces |
| `./sentinel.py verify --all --check-consistency` | Verify all with consistency check |
| `./sentinel.py pipeline` | Run full pre-commit pipeline |
| `./sentinel.py retrace NAME --auto` | Auto-analyze anchor impacts |
| `./sentinel.py retrace NAME --auto --apply` | Analyze and apply safe updates |
| `./sentinel.py retrace NAME --diff-only` | Show diff without updating |
| `./sentinel.py retrace NAME --anchors-only` | Force update anchor line numbers |
| `./sentinel.py install-hooks` | Install git pre-commit hooks |
| `./sentinel.py report --format json` | Generate JSON report |
| `./sentinel.py report --format markdown` | Generate Markdown report |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All verification passed |
| 1 | One or more anchors missing |
| 2 | One or more anchors drifted beyond tolerance |
| 3 | One or more anchor patterns are ambiguous |
| 4 | Consistency check failed |
| 5 | Assumption verification failed |

## Updating Sentinels

When code changes require sentinel updates:

1. **Update trace markdown first**
   - Update "Verified against" commit hash
   - Update Algorithm Steps line references
   - Preserve Critical Invariants, Known Issues, Linked Tests

2. **Update anchors.yaml** (if patterns changed)
   ```bash
   ./sentinel.py retrace <name> --anchors-only
   ```

3. **Update .sentinel-meta.yaml**
   - Bump `verified_commit`
   - Update `last_global_verification`

4. **Commit together**
   ```bash
   git add traces/ anchors/ .sentinel-meta.yaml sentinel.py
   git commit -m "sentinel: update <trace-name> for <change>"
   ```

## Symbolic Shape Legend

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `T` | Sequence length |
| `K` | Maximum segment length |
| `C` | Number of classes |
| `C_PAD` | Padded class count (power of 2) |

## Domain-Specific Invariants

| Invariant | Check |
|-----------|-------|
| Log-partition bounds | `partition >= viterbi_score` |
| Padding sentinel | `alpha[t] == NEG_INF for t > T` |
| Prefix-sum init | `cum_scores[:, 0, :] == 0` |
| Ring buffer aliasing | `alpha[t]` overwrites `alpha[t-K]` |

## Example Invocations

```bash
# Check health before debugging
./sentinel.py status

# Scaffold a new trace
./sentinel.py init src/torch_semimarkov/nn.py --name crf-heads

# Trace the Triton forward kernel
./sentinel.py verify --trace triton-forward-k3plus

# Debug NaN in backward pass
./sentinel.py verify --trace triton-backward-k3plus

# Auto-analyze and fix drifted anchors
./sentinel.py retrace triton-forward-k3plus --auto --apply

# Pre-commit check
./sentinel.py pipeline

# Generate CI report
./sentinel.py report --format json > sentinel-report.json
```

## Files Reference

| File | Purpose |
|------|---------|
| `sentinel.py` | Main CLI orchestrator |
| `.sentinel-meta.yaml` | Machine-readable sentinel state |
| `anchors/anchors.yaml` | Pattern -> line number mappings |
| `anchors/verify-all.py` | Batch anchor verification |
| `anchors/verify-anchor.sh` | Single anchor verification |
| `verify-assumptions.py` | Assumption verification |
| `hooks/pre-commit-*.sh` | Git hook scripts |
