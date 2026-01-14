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
    --backends linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,binary_tree_sharded,block_triangular
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

Note: Triton backends only support `Log` semiring (hardcoded logsumexp operations).

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
    --backends linear_scan,linear_scan_streaming,binary_tree \
    --T 128,256,512 \
    --K 4,8,12 \
    --C 3,6
```

Note: `triton` and `triton_pytorch` backends only support `Log` semiring.

## Full Benchmark Suite

Run comprehensive benchmarks across all dimensions:

```bash
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --B 4 \
    --repeats 5 \
    --phases forward,backward,both \
    --semirings Log,Max \
    --backends linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,triton \
    --output-dir results/
```

## Available Options

### Backends

| Backend | Description | Memory | Semirings |
|---------|-------------|--------|-----------|
| `linear_scan` | Reference O(N) scan | O(TKC) | All |
| `linear_scan_vectorized` | Vectorized O(N) scan | O(TKC) | All |
| `linear_scan_streaming` | Streaming O(N) scan | O(KC) | All |
| `binary_tree` | O(log N) parallel tree | O((KC)^3) | All |
| `binary_tree_sharded` | Sharded tree (checkpointed) | Reduced | All |
| `block_triangular` | Block-sparse tree | O(N*KC^2) | All |
| `triton` | Fused Triton GPU kernel | O(KC) | Log only |
| `triton_pytorch` | PyTorch reference for Triton | O(KC) | Log only |

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
# Run benchmarks
python benchmarks/benchmark_memory_analysis.py \
    --phases forward,backward \
    --semirings Log,Max \
    --backends linear_scan_streaming,triton,binary_tree \
    --T 128,256,512,1024 \
    --K 4,8,12 \
    --C 3,6,9 \
    --output-dir results/

# Analyze results
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --format pdf
```
