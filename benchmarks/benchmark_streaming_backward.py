#!/usr/bin/env python3
"""
Benchmark for streaming Semi-CRF backward kernel performance.

This benchmark measures the backward pass performance at various scales
to establish a baseline before Phase 3 optimizations.

Usage:
    python3 benchmarks/benchmark_streaming_backward.py

Requires CUDA and Triton.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch: int
    T: int
    K: int
    C: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"B{self.batch}_T{self.T}_K{self.K}_C{self.C}"


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    config: BenchmarkConfig
    forward_ms: float
    backward_ms: float
    total_ms: float
    peak_memory_gb: float
    status: str = "success"
    error: Optional[str] = None


# Benchmark configurations from small to large scale
CONFIGS = [
    # Small scale - quick iteration
    BenchmarkConfig(batch=4, T=100, K=8, C=4, name="tiny"),
    BenchmarkConfig(batch=4, T=500, K=16, C=8, name="small"),

    # Medium scale - typical training
    BenchmarkConfig(batch=4, T=1000, K=32, C=24, name="medium"),
    BenchmarkConfig(batch=4, T=5000, K=64, C=24, name="medium_long"),

    # Large scale - target genomics
    BenchmarkConfig(batch=2, T=10000, K=100, C=24, name="large"),
    BenchmarkConfig(batch=2, T=50000, K=500, C=24, name="very_large"),

    # Scale target (may OOM on smaller GPUs)
    BenchmarkConfig(batch=1, T=100000, K=1000, C=24, name="target_scale"),
]


def create_inputs(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype = torch.float32):
    """Create test inputs for the streaming API."""
    torch.manual_seed(42)

    batch, T, K, C = config.batch, config.T, config.K, config.C

    # Simulate projected encoder features (zero-centered)
    projected = torch.randn(batch, T, C, device=device, dtype=dtype)
    projected = projected - projected.mean(dim=1, keepdim=True)

    # Cumulative scores: (batch, T+1, C)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    # Lengths
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    return cum_scores, transition, duration_bias, lengths


def benchmark_triton_backward(
    config: BenchmarkConfig,
    device: torch.device,
    warmup: int = 3,
    repeats: int = 10,
) -> BenchmarkResult:
    """Benchmark the Triton streaming backward kernel."""
    from torch_semimarkov.streaming import semi_crf_streaming_forward, HAS_TRITON

    if not HAS_TRITON:
        return BenchmarkResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            status="error",
            error="Triton not available",
        )

    try:
        # Create inputs with gradients
        cum_scores, transition, duration_bias, lengths = create_inputs(config, device)
        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # Warmup runs
        for _ in range(warmup):
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=True
            )
            partition.sum().backward()
            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed runs
        forward_times = []
        backward_times = []
        total_times = []

        for _ in range(repeats):
            torch.cuda.synchronize()

            # Forward timing
            start_fwd = time.perf_counter()
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=True
            )
            torch.cuda.synchronize()
            end_fwd = time.perf_counter()

            # Backward timing
            start_bwd = time.perf_counter()
            partition.sum().backward()
            torch.cuda.synchronize()
            end_bwd = time.perf_counter()

            forward_times.append((end_fwd - start_fwd) * 1000)
            backward_times.append((end_bwd - start_bwd) * 1000)
            total_times.append((end_bwd - start_fwd) * 1000)

            # Clear gradients for next iteration
            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        return BenchmarkResult(
            config=config,
            forward_ms=sum(forward_times) / len(forward_times),
            backward_ms=sum(backward_times) / len(backward_times),
            total_ms=sum(total_times) / len(total_times),
            peak_memory_gb=peak_memory,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchmarkResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            status="oom",
            error="CUDA out of memory",
        )
    except Exception as e:
        return BenchmarkResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            status="error",
            error=str(e),
        )


def benchmark_pytorch_backward(
    config: BenchmarkConfig,
    device: torch.device,
    warmup: int = 3,
    repeats: int = 10,
) -> BenchmarkResult:
    """Benchmark the PyTorch reference backward pass."""
    from torch_semimarkov.streaming import semi_crf_streaming_forward

    try:
        # Create inputs with gradients
        cum_scores, transition, duration_bias, lengths = create_inputs(config, device)
        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # Warmup runs
        for _ in range(warmup):
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=False
            )
            partition.sum().backward()
            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed runs
        forward_times = []
        backward_times = []
        total_times = []

        for _ in range(repeats):
            torch.cuda.synchronize()

            # Forward timing
            start_fwd = time.perf_counter()
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=False
            )
            torch.cuda.synchronize()
            end_fwd = time.perf_counter()

            # Backward timing
            start_bwd = time.perf_counter()
            partition.sum().backward()
            torch.cuda.synchronize()
            end_bwd = time.perf_counter()

            forward_times.append((end_fwd - start_fwd) * 1000)
            backward_times.append((end_bwd - start_bwd) * 1000)
            total_times.append((end_bwd - start_fwd) * 1000)

            # Clear gradients for next iteration
            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        return BenchmarkResult(
            config=config,
            forward_ms=sum(forward_times) / len(forward_times),
            backward_ms=sum(backward_times) / len(backward_times),
            total_ms=sum(total_times) / len(total_times),
            peak_memory_gb=peak_memory,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchmarkResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            status="oom",
            error="CUDA out of memory",
        )
    except Exception as e:
        return BenchmarkResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            status="error",
            error=str(e),
        )


def print_results(results_triton: list[BenchmarkResult], results_pytorch: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("STREAMING BACKWARD KERNEL BENCHMARK RESULTS")
    print("=" * 100)

    print(f"\n{'Config':<20} | {'Backend':<10} | {'Forward':>10} | {'Backward':>10} | {'Total':>10} | {'Memory':>8} | {'Status':<10}")
    print("-" * 100)

    for rt, rp in zip(results_triton, results_pytorch):
        # Triton row
        if rt.status == "success":
            print(f"{rt.config.name:<20} | {'Triton':<10} | {rt.forward_ms:>8.2f}ms | {rt.backward_ms:>8.2f}ms | {rt.total_ms:>8.2f}ms | {rt.peak_memory_gb:>6.2f}GB | {rt.status:<10}")
        else:
            print(f"{rt.config.name:<20} | {'Triton':<10} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8} | {rt.status:<10}")

        # PyTorch row
        if rp.status == "success":
            print(f"{'':<20} | {'PyTorch':<10} | {rp.forward_ms:>8.2f}ms | {rp.backward_ms:>8.2f}ms | {rp.total_ms:>8.2f}ms | {rp.peak_memory_gb:>6.2f}GB | {rp.status:<10}")
        else:
            print(f"{'':<20} | {'PyTorch':<10} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8} | {rp.status:<10}")

        # Speedup
        if rt.status == "success" and rp.status == "success" and rp.backward_ms > 0:
            speedup = rp.backward_ms / rt.backward_ms
            print(f"{'':<20} | {'Speedup':<10} | {'':>10} | {speedup:>8.2f}x  | {'':>10} | {'':>8} |")

        print("-" * 100)

    # Summary statistics
    print("\nSUMMARY: Backward Pass Speedups (Triton vs PyTorch)")
    print("-" * 50)
    speedups = []
    for rt, rp in zip(results_triton, results_pytorch):
        if rt.status == "success" and rp.status == "success" and rp.backward_ms > 0:
            speedup = rp.backward_ms / rt.backward_ms
            speedups.append((rt.config.name, speedup))
            print(f"  {rt.config.name:<20}: {speedup:.2f}x")

    if speedups:
        avg_speedup = sum(s for _, s in speedups) / len(speedups)
        print(f"\n  Average speedup: {avg_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to benchmark on")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config names or 'all'")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    device = torch.device(args.device)
    print(f"Benchmarking on device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.repeats}")

    # Select configs
    if args.configs == "all":
        configs = CONFIGS
    else:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = [c for c in CONFIGS if c.name in config_names]
        if not configs:
            print(f"ERROR: No matching configs found. Available: {[c.name for c in CONFIGS]}")
            return

    print(f"\nRunning {len(configs)} configurations...")

    results_triton = []
    results_pytorch = []

    for config in configs:
        print(f"\n--- {config.name}: batch={config.batch}, T={config.T}, K={config.K}, C={config.C} ---")

        # Clear cache between configs
        torch.cuda.empty_cache()

        print("  Running Triton backward...", end=" ", flush=True)
        result_triton = benchmark_triton_backward(config, device, args.warmup, args.repeats)
        if result_triton.status == "success":
            print(f"OK ({result_triton.backward_ms:.2f}ms)")
        else:
            print(f"{result_triton.status}: {result_triton.error}")
        results_triton.append(result_triton)

        torch.cuda.empty_cache()

        print("  Running PyTorch backward...", end=" ", flush=True)
        result_pytorch = benchmark_pytorch_backward(config, device, args.warmup, args.repeats)
        if result_pytorch.status == "success":
            print(f"OK ({result_pytorch.backward_ms:.2f}ms)")
        else:
            print(f"{result_pytorch.status}: {result_pytorch.error}")
        results_pytorch.append(result_pytorch)

    print_results(results_triton, results_pytorch)


if __name__ == "__main__":
    main()
