# Benchmarking

Run the included benchmarks to reproduce paper results.

## Basic Usage

```bash
# Memory analysis across backends (default: forward+backward combined)
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --backends triton,triton_pytorch,linear_scan_streaming
```

## Triton-Accelerated Scan

Compare Triton kernel vs PyTorch reference implementations:

```bash
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 256,512,1024 \
    --K 8,12,16 \
    --C 6,9,12 \
    --backends triton,triton_pytorch,linear_scan_streaming
```

Note: Triton backends support `Log` and `Max` semirings.

## Phase-Separated Timing

Measure forward and backward passes separately:

```bash
# Forward pass only
python benchmarks/benchmark_memory_analysis.py \
    --phases forward \
    --backends linear_scan_streaming,triton

# Backward pass only
python benchmarks/benchmark_memory_analysis.py \
    --phases backward \
    --backends linear_scan_streaming,triton

# All phases for comparison
python benchmarks/benchmark_memory_analysis.py \
    --phases forward,backward,both \
    --backends linear_scan_streaming,triton
```

## Semiring Comparison

Compare different semirings (Log, Max, Entropy):

```bash
python benchmarks/benchmark_memory_analysis.py \
    --semirings Log,Max,Entropy \
    --backends triton,linear_scan_streaming \
    --T 128,256,512 \
    --K 4,8,12 \
    --C 3,6
```

Note: `triton` and `triton_pytorch` backends support `Log` and `Max` semirings.

## Compile-Friendly Benchmarking (Recommended for Triton)

When benchmarking triton backends with `torch.compile`, use compile-friendly sampling to avoid
excessive compilation overhead. This groups configurations by canonical shapes and pre-compiles
kernels before timing:

```bash
# Fast benchmark with compile-friendly sampling (recommended)
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16 \
    --C 3,6,9,12 \
    --backends triton,triton_pytorch,linear_scan_streaming \
    --compile-friendly \
    --max-canonical-shapes 8 \
    --samples-per-shape 2 \
    --output-dir results/
```

This reduces compilation from ~96 unique shapes to ~8 canonical shapes, cutting benchmark
time from hours to minutes while maintaining good coverage of the parameter space.

### How Compile-Friendly Sampling Works

1. **Shape Bucketing**: Configurations are grouped into canonical shapes based on T (sequence length)
   and K×C (state-space size) buckets
2. **Pre-compilation**: All canonical shapes are compiled once before timing begins
3. **Persistent Cache**: Compiled kernels are cached in `--output-dir/.torch_compile_cache/`
4. **Representative Sampling**: A subset of actual configs are benchmarked per canonical shape

### Canonical Shape Buckets

| Dimension | Buckets |
|-----------|---------|
| T (sequence) | 64, 128, 256, 512, 1024, 2048 |
| K×C (states) | 12, 24, 48, 72, 96, 144, 192, 288 |

## Full Benchmark Suite

Run comprehensive benchmarks across all dimensions:

```bash
# Full grid (use for non-triton backends or when compile time isn't a concern)
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --B 4 \
    --repeats 5 \
    --phases forward,backward,both \
    --semirings Log,Max \
    --backends linear_scan_streaming \
    --output-dir results/

# With triton backends, use compile-friendly mode
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --B 4 \
    --repeats 5 \
    --phases forward,backward,both \
    --semirings Log,Max \
    --backends linear_scan_streaming,triton \
    --compile-friendly \
    --max-canonical-shapes 12 \
    --output-dir results/
```

## Available Options

### Backends

| Backend | Description | Memory | Semirings |
|---------|-------------|--------|-----------|
| `linear_scan_streaming` | Streaming O(N) scan (PyTorch reference) | O(KC) | All |
| `triton` | Fused Triton GPU kernel (torch.compile for training) | O(KC) | Log, Max |
| `triton_pytorch` | PyTorch reference for Triton | O(KC) | Log, Max |
| `triton_checkpointing` | Triton with gradient checkpointing | O(KC) | Log, Max |

### Semirings

| Semiring | Description | Use Case |
|----------|-------------|----------|
| `Log` | Log-space (logsumexp, +) | Partition function, marginals |
| `Max` | Max-plus (max, +) | Viterbi decoding |
| `Entropy` | Entropy computation | Uncertainty quantification |

### Phases

| Phase | Description |
|-------|-------------|
| `forward` | Time forward pass only |
| `backward` | Time backward pass only |
| `both` | Time forward + backward together (default) |

### Triton Options

| Option | Description |
|--------|-------------|
| `--use-compile` | Use torch.compile for triton training (default) |
| `--no-use-compile` | Use gradient checkpointing instead of torch.compile |

### Compile-Friendly Sampling Options

| Option | Default | Description |
|--------|---------|-------------|
| `--compile-friendly` | off | Enable compile-aware sampling to minimize torch.compile overhead |
| `--max-canonical-shapes` | 8 | Maximum unique shapes to compile (more = better coverage, slower) |
| `--samples-per-shape` | 2 | Actual configs to benchmark per canonical shape |
| `--skip-precompile` | off | Skip pre-compilation phase (use if cache is warm) |
| `--compile-cache-dir` | `<output-dir>/.torch_compile_cache` | Directory for compile cache (use local scratch on HPC) |
| `--compile-cache-size-gb` | 10.0 | Maximum cache size in GB before eviction |
| `--compile-threads` | all | Number of CPU threads for compilation (match HPC allocation) |

The `triton` backend uses a hybrid approach:
- **Inference** (`--phases forward`): Custom Triton kernel (~45x faster)
- **Training** (`--phases backward,both`): `torch.compile` for efficient backward by default

Use `triton_checkpointing` backend or `--no-use-compile` to compare against the older gradient checkpointing approach.

## Output Files

The benchmark produces three output files in `--output-dir`:

- `benchmark_full.csv`: Complete results with all metrics
- `heatmap_data.json`: Data for OOM feasibility heatmaps
- `memory_breakdown.csv`: Memory breakdown by category

## Analysis and Plotting

Use the analysis script to generate plots and derived metrics from benchmark results:

```bash
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --format pdf

# Compare all backends against a specific baseline
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --baseline linear_scan_streaming
```

### Generated Plots

| Plot | Description |
|------|-------------|
| `scalability_T_*.pdf` | Time vs sequence length (log-log) - reveals O(N) vs O(log N) |
| `scalability_KC_*.pdf` | Time vs state-space size |
| `throughput_*.pdf` | Positions/sec vs KC |
| `memory_efficiency_*.pdf` | Memory per state-position |
| `backward_forward_ratio_*.pdf` | Cost of backward pass vs forward |
| `semiring_overhead_*.pdf` | Overhead of Max/Entropy vs Log |
| `time_ratio_baseline_*.pdf` | Time ratio vs baseline backend |
| `memory_ratio_baseline_*.pdf` | Memory ratio vs baseline backend |
| `time_ratio_heatmap_*.pdf` | Heatmap of time ratios (backend × KC) |
| `memory_ratio_heatmap_*.pdf` | Heatmap of memory ratios (backend × KC) |

### Generated Tables

| Table | Description |
|-------|-------------|
| `summary_stats.csv` | Aggregate statistics by backend/semiring/phase |
| `backward_forward_ratios.csv` | Backward/forward time ratios per config |
| `semiring_overhead.csv` | Overhead relative to LogSemiring |
| `crossover_points.csv` | KC thresholds where streaming beats other backends |
| `baseline_ratios.csv` | Time and memory ratios vs baseline backend |

### Example: Generate all analysis

```bash
# Run benchmarks (compile-friendly mode for triton)
python benchmarks/benchmark_memory_analysis.py \
    --phases forward,backward \
    --semirings Log,Max \
    --backends linear_scan_streaming,triton \
    --T 128,256,512,1024 \
    --K 4,8,12 \
    --C 3,6,9 \
    --compile-friendly \
    --max-canonical-shapes 8 \
    --output-dir results/

# Analyze results
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --format pdf
```

### Example: Quick Triton vs Streaming Comparison

```bash
# Minimal compile-friendly run for rapid iteration
python benchmarks/benchmark_memory_analysis.py \
    --backends triton,linear_scan_streaming \
    --T 256,512,1024 \
    --K 8,16 \
    --C 6,12 \
    --compile-friendly \
    --max-canonical-shapes 4 \
    --samples-per-shape 1 \
    --phases forward \
    --output-dir results/quick/
```

This runs ~8 configs with only 4 compiled shapes, completing in under a minute on warm cache.

### Example: HPC with Local Scratch

On HPC systems, use node-local storage for the compile cache to avoid network filesystem overhead:

```bash
# Use $TMPDIR (typically node-local scratch) with larger cache
# Match --compile-threads to your SLURM/PBS CPU allocation
python benchmarks/benchmark_memory_analysis.py \
    --backends triton,linear_scan_streaming \
    --T 128,256,512,1024 \
    --K 4,8,12,16 \
    --C 3,6,9,12 \
    --compile-friendly \
    --compile-cache-dir "${TMPDIR:-/tmp}/torch_compile_cache" \
    --compile-cache-size-gb 50.0 \
    --compile-threads "${SLURM_CPUS_PER_TASK:-8}" \
    --output-dir results/

# Or use a persistent cache on fast local NVMe (with warm cache)
python benchmarks/benchmark_memory_analysis.py \
    --backends triton,linear_scan_streaming \
    --compile-friendly \
    --compile-cache-dir /local/scratch/$USER/torch_cache \
    --compile-cache-size-gb 100.0 \
    --skip-precompile \
    --output-dir results/
```

**HPC Tips:**
- Use `$TMPDIR` or node-local NVMe instead of network filesystems (Lustre, GPFS)
- Set `--compile-threads` to match your CPU allocation (`$SLURM_CPUS_PER_TASK`, `$PBS_NCPUS`)
- Set cache size based on available scratch space (compiled kernels can be 100MB+ each)
- Use `--skip-precompile` on subsequent runs if cache persists between jobs
- The cache stores both torch.compile artifacts and Triton kernels
