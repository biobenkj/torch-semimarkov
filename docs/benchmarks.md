# Benchmarking

Run the included benchmarks to reproduce paper results.

```bash
# Memory analysis across backends
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --backends linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,binary_tree_sharded,block_triangular

# Multi-backend comparison
python benchmarks/benchmark_backends.py \
    --device cuda:0 \
    --T 512,1024,2048 \
    --K 12,16,20 \
    --C 3,6 \
    --backends binary_tree,linear_scan,linear_scan_vectorized \
    --csv results.csv

# Dense vs banded comparison
python benchmarks/benchmark_grid.py \
    --T 512,1024,2048 \
    --K 8,12,16 \
    --C 3,6 \
    --csv grid_results.csv
```

Triton vs streaming scan comparisons:

```bash
python -m torch_semimarkov.triton_scan
```
