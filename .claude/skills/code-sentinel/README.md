# Code Sentinel

**Persistent execution traces ("Sentinels") for torch-semimarkov development.**

Code Sentinel prevents hallucinations about code execution paths by maintaining verified baseline documentation that can be mechanically checked against source code. It transforms the traditional "trust the model's understanding" approach into a "verify then trust" methodology.

## Philosophy

When using LLMs for code assistance, the model constructs a mental map of your codebase from context. This map can drift from reality through:

1. **Stale context**: Code changed since the model last read it
2. **Agentic assumptions**: Model "fills in gaps" with plausible but incorrect details
3. **Hallucinated connections**: Inventing function calls or data flows that don't exist

Code Sentinel addresses this by:
- **Grounding** model understanding in verified trace documents
- **Detecting drift** through mechanical anchor verification
- **Forcing synchronization** before providing advice

## Quick Start

```bash
# Check all sentinels for drift
./sentinel.py status

# Scaffold a new trace from source file
./sentinel.py init src/torch_semimarkov/streaming/new_module.py

# Full verification before making changes
./sentinel.py verify --trace triton-forward-k3plus

# Auto-detect anchor changes after code modifications
./sentinel.py retrace triton-forward-k3plus --auto

# Run full pipeline (recommended before commits)
./sentinel.py pipeline

# Install git hooks
./sentinel.py install-hooks
```

## Core Concepts

### Sentinels (Trace Documents)

A sentinel is a Markdown document in `traces/` that captures:
- **Verified commit**: The exact git commit against which the trace was written
- **Algorithm steps**: Line-by-line explanation of execution flow
- **Data shapes**: Input/output tensor dimensions and types
- **Critical invariants**: Properties that must hold for correctness
- **Numerical guards**: Safety checks for NaN/Inf handling
- **Known issues**: Previously encountered bugs and their resolutions

Example trace header:
```markdown
# Sentinel: Triton Forward Kernel (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/triton_forward.py` @ commit `40fe66b`
**Linked tests:** `tests/test_streaming_triton.py::TestTritonBasic`
```

### Anchors

Anchors are pattern-based references to specific code locations. They enable mechanical verification that trace line references haven't drifted beyond acceptable bounds.

```yaml
# anchors/anchors.yaml
triton-forward-k3plus:
  KERNEL_ENTRY:
    file: src/torch_semimarkov/streaming/triton_forward.py
    pattern: "def semi_crf_streaming_scan_kernel("
    expected_line: 84
    drift_tolerance: 20
```

Anchor verification returns:
- `ANCHOR_VERIFIED`: Pattern found within drift tolerance
- `ANCHOR_MISSING`: Pattern not found in file
- `ANCHOR_DRIFT`: Pattern found but beyond tolerance
- `ANCHOR_AMBIGUOUS`: Pattern matches multiple lines

### Assumptions

Assumptions are verifiable claims embedded in trace documents. Two types:

**Mechanically Verified** (automated):
```markdown
| ID | Assumption | Verification |
|----|------------|--------------|
| A1 | Ring buffer uses `t % K` write indexing | anchor: RING_BUFFER_WRITE |
```

**Agent-Verified** (manual check on trace load):
```markdown
| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| A4 | C_PAD is power of 2 | Check C_PAD assignment uses `2 ** math.ceil(...)` |
```

## Directory Structure

```
.claude/skills/code-sentinel/
├── SKILL.md                    # Claude Code skill definition
├── README.md                   # This file
├── sentinel.py                 # Main CLI orchestrator
├── verify-assumptions.py       # Assumption verification
├── .sentinel-meta.yaml         # Machine-readable sentinel state
├── anchors/
│   ├── anchors.yaml           # Pattern → line number mappings
│   ├── verify-all.py          # Batch anchor verification
│   └── verify-anchor.sh       # Single anchor verification
├── traces/
│   ├── dispatch-overview.md   # Backend selection decision tree
│   ├── triton-forward-k3plus.md
│   ├── triton-backward-k3plus.md
│   ├── pytorch-forward-k3plus.md
│   ├── pytorch-backward-k3plus.md
│   ├── k1-linear-crf.md
│   ├── k2-fast-path.md
│   ├── non-streaming-backends.md
│   ├── autograd-kernel-interface.md
│   └── concerns/              # Cross-cutting concerns
├── hooks/
│   ├── pre-commit-anchors.sh  # Verify anchors on commit
│   └── pre-commit-tests.sh    # Suggest tests for changed files
└── diffs/                      # Working directory for drift analysis
```

## CLI Reference

### `sentinel.py status`

Show overall sentinel health without running verification.

```bash
./sentinel.py status
```

Output:
```
Code Sentinel Status
========================================
Last global verification: 2026-01-28T00:00:00Z

Sentinels:
  ✓ triton-forward-k3plus: VERIFIED
  ✓ dispatch-overview: VERIFIED
  ...
```

### `sentinel.py init`

Scaffold a new trace from a source file. Analyzes the source to extract functions and classes, assigns importance levels based on domain-specific patterns, and generates a draft trace document with suggested anchors.

```bash
# Basic usage
./sentinel.py init src/torch_semimarkov/streaming/new_module.py

# Custom trace name
./sentinel.py init src/torch_semimarkov/streaming/new_module.py --name my-trace

# Overwrite existing trace
./sentinel.py init src/torch_semimarkov/streaming/new_module.py --force
```

Output:
```
Analyzing src/torch_semimarkov/streaming/new_module.py...
  Found 15 functions, 2 classes
  Critical: 3, High: 5

Created trace: traces/new_module.md

=== Suggested Anchors ===
Add to anchors/anchors.yaml:
  FORWARD:
    file: src/torch_semimarkov/streaming/new_module.py
    pattern: "def forward("
    expected_line: 42
    drift_tolerance: 30
...

=== Suggested Meta Entry ===
...

Next steps:
  1. Edit the trace file to complete TODO sections
  2. Add anchors to anchors/anchors.yaml
  3. Add entry to .sentinel-meta.yaml
  4. Update verified_commit and status when ready
  5. Run: ./sentinel.py verify --trace new_module
```

The init command uses domain-specific pattern matching to identify critical functions:
- **Critical patterns**: `forward`, `backward`, `apply`, `@triton.jit`, `semi_crf`, `launch_`
- **High patterns**: `__init__`, `logsumexp`, `NEG_INF`, `torch.isfinite`, `torch.isnan`, `decode`

#### Design Philosophy: Intentional Scaffolding Limits

The `init` command intentionally provides only ~17% of a completed trace (structural scaffolding and accurate line references). The remaining ~83% represents **irreducible domain knowledge** that must come from human or LLM understanding:

| Category | % of Final Trace | Nature |
|----------|------------------|--------|
| Domain understanding | ~37% | *Why* code does what it does |
| Numerical patterns | ~19% | Which guards are critical and why |
| Assumption formulation | ~15% | Mapping invariants to verifiable checks |
| Anchor pattern discovery | ~11% | Non-obvious patterns (guards, conversions) |
| Known issues | ~8% | Historical context and edge cases |

**This gap is intentional and should NOT be optimized away.** Sentinels derive their value precisely from encoding understanding that source code alone does not express. Attempting to auto-generate the "why" would defeat the purpose of grounded verification.

The `init` command's role is deliberately limited to:

- Eliminating structural/formatting decisions
- Providing accurate line number references
- Identifying key entry points by name

It explicitly does NOT attempt to explain code behavior, identify critical invariants, or document numerical stability patterns—these require reading and understanding the code.

### `sentinel.py verify`

Run full verification for a specific trace or all traces.

```bash
# Verify specific trace
./sentinel.py verify --trace triton-forward-k3plus

# Verify all traces
./sentinel.py verify --all

# Include consistency check
./sentinel.py verify --all --check-consistency
```

Output on success:
```
Grounded: triton-forward-k3plus @ 40fe66b ✓
```

Output on failure:
```
Grounded: triton-forward-k3plus @ 40fe66b
  Commit: STALE (COMMITTED_CHANGES)
    Verified: 40fe66b | Current: abc1234
  Anchors: 5/7 verified, 2 failed
    ✗ RING_BUFFER_WRITE: ANCHOR_DRIFT: Expected ~320, found 345 (drift 25 > 20)
  Assumptions: A1 ✓, A2 ✗, A3 ✓

⚠️  Cannot provide advice until sentinel is updated.
```

### `sentinel.py pipeline`

Run full pre-commit pipeline:
1. Consistency check (meta ↔ anchors ↔ traces)
2. Trace verification (all traces)
3. Test advisory (suggest tests for changed files)

```bash
# Full pipeline
./sentinel.py pipeline

# Pipeline with automatic test execution
SENTINEL_RUN_TESTS=1 ./sentinel.py pipeline

# Pipeline for specific files
./sentinel.py pipeline --files src/torch_semimarkov/streaming/triton_forward.py
```

Exit codes:
- `0`: All checks passed
- `1`: One or more anchors missing
- `2`: One or more anchors drifted
- `3`: Anchor pattern is ambiguous
- `4`: Consistency check failed
- `5`: Assumption verification failed

### `sentinel.py retrace`

Regenerate a sentinel when code has changed. Offers multiple modes from fully automatic to manual.

```bash
# Auto-analyze anchor impacts (dry run)
./sentinel.py retrace triton-forward-k3plus --auto

# Auto-analyze and apply safe updates (shifted anchors only)
./sentinel.py retrace triton-forward-k3plus --auto --apply

# Force update all anchor line numbers
./sentinel.py retrace triton-forward-k3plus --anchors-only

# Show current state and diff
./sentinel.py retrace triton-forward-k3plus --diff-only
```

The `--auto` mode analyzes changes since the verified commit and classifies anchor impacts:

- **Unchanged**: Anchor still at expected location (within tolerance)
- **Shifted** (auto-fixable): Pattern found but line number changed
- **Modified** (needs review): Pattern matches multiple lines
- **Deleted** (manual fix): Pattern no longer found

Example output:
```
=== Auto-Retrace Analysis ===

Anchor Impact Summary:
  Unchanged: 5
  Shifted (auto-fixable): 2
  Modified (needs review): 0
  Deleted (manual fix): 0

Shifted anchors (safe to auto-update):
  RING_BUFFER_WRITE: 320 -> 325
  CHECKPOINT_SAVE: 329 -> 334

Run with --apply to update 2 shifted anchor(s)
```

### `sentinel.py install-hooks`

Install git hooks for automatic verification on commit.

```bash
./sentinel.py install-hooks
```

### `sentinel.py report`

Generate machine-readable verification report.

```bash
# JSON report
./sentinel.py report --format json > sentinel-report.json

# Markdown report
./sentinel.py report --format markdown > SENTINEL_STATUS.md
```

## Verification Protocol

When loading a sentinel for debugging assistance:

### Step 1: Run Verification

```bash
./sentinel.py verify --trace <trace-name>
```

This performs:
- Commit staleness check (verified hash vs current HEAD)
- Uncommitted changes detection
- Anchor verification (pattern → line number)
- Assumption verification (mechanical checks)

### Step 2: Interpret Output

**On success:** Proceed with debugging
```
Grounded: triton-forward-k3plus @ 40fe66b ✓
```

**On failure:** Remediate before advising
```
⚠️  Cannot provide advice until sentinel is updated.
```

### Step 3: Remediate if Needed

If anchors drifted, auto-update line numbers:
```bash
./sentinel.py retrace triton-forward-k3plus --anchors-only
```

If code semantics changed, manually update trace.

## Updating Sentinels

When code changes require sentinel updates:

### Anchors Drifted (line numbers changed)

```bash
# Auto-update anchor line numbers
./sentinel.py retrace <trace-name> --anchors-only

# Verify update worked
./sentinel.py verify --trace <trace-name>
```

### Code Semantics Changed

1. Update trace markdown (source of truth for narrative)
   - Update "Verified against" commit hash
   - Update Algorithm Steps
   - Preserve Critical Invariants, Known Issues, Linked Tests

2. Update anchors.yaml if patterns changed

3. Update .sentinel-meta.yaml
   - Bump `verified_commit`
   - Update `last_global_verification`

4. Commit together:
   ```bash
   git add traces/ anchors/ .sentinel-meta.yaml
   git commit -m "sentinel: update triton-forward-k3plus for <change>"
   ```

## Workflow Examples

### Before Making Changes

```bash
# Load relevant sentinel(s)
./sentinel.py verify --trace triton-forward-k3plus

# If verified, proceed with changes
# If failed, retrace first
./sentinel.py retrace triton-forward-k3plus --diff-only
```

### Before Committing

```bash
# Run full pipeline
./sentinel.py pipeline

# Or with automatic tests
SENTINEL_RUN_TESTS=1 ./sentinel.py pipeline
```

### Debugging a NaN

```bash
# 1. Identify backend (check dispatch-overview first)
./sentinel.py verify --trace dispatch-overview

# 2. Load specific forward trace
./sentinel.py verify --trace triton-forward-k3plus

# 3. Check NEG_INF guards section in trace
# 4. Follow Failure Mode Routing table in SKILL.md
```

## Configuration Reference

### .sentinel-meta.yaml

```yaml
version: 2.0
last_global_verification: 2026-01-28T00:00:00Z

sentinels:
  trace-name:
    verified_commit: abc1234
    source_files:
      - path/to/file.py
    assumptions_mechanical: [A1, A2, A3]
    assumptions_agent: [A4]
    anchors:
      - ANCHOR_NAME_1
      - ANCHOR_NAME_2
    linked_tests:
      - tests/test_file.py::TestClass::test_method
    status: VERIFIED  # or STALE, DEGRADED
    depends_on:
      - other-trace-name

test_bindings:
  path/to/source.py:
    - tests/test_file.py::TestClass
```

### anchors.yaml

```yaml
trace-name:
  ANCHOR_NAME:
    file: path/to/file.py
    pattern: "def function_name("
    expected_line: 100
    drift_tolerance: 20
    after: "class ClassName"  # Optional: disambiguate repeated patterns
```

## Best Practices

1. **Always verify before debugging**: Run staleness check before providing advice
2. **Commit sentinels with code**: Keep trace updates atomic with code changes
3. **Link tests**: Every sentinel should reference tests that validate its assumptions
4. **Use drift tolerance wisely**: Larger tolerances for stable code, smaller for volatile
5. **Prefer mechanical verification**: Convert agent-verified assumptions to anchors when possible
6. **Document known issues**: Track resolved bugs to prevent regression in advice

## Troubleshooting

### "ANCHOR_AMBIGUOUS" errors

The pattern matches multiple lines. Add an `after:` field to disambiguate:

```yaml
ANCHOR_NAME:
  pattern: "if condition:"
  after: "def specific_function("  # Search only after this pattern
```

### Sentinel always shows STALE

Check that the verified_commit in both `.sentinel-meta.yaml` and the trace header match the actual commit hash in git history:

```bash
git log --oneline -10 -- path/to/file.py
```

### Tests not being suggested

Verify test_bindings in `.sentinel-meta.yaml` use path patterns that match your changed files:

```yaml
test_bindings:
  src/torch_semimarkov/streaming/triton_forward.py:  # Exact path
    - tests/test_streaming_triton.py::TestTritonBasic
```
