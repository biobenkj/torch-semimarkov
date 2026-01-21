#!/usr/bin/env python3
"""
Scale benchmark for streaming Semi-CRF at genomics-scale dimensions.

This benchmark tests performance at the target dimensions for the
streaming edge API: T=100K+, K=1000+, C=24.

Usage:
    python3 benchmarks/benchmark_streaming_scale.py

Requires CUDA with sufficient memory (recommended: 40GB+ GPU like A100/L40S).
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ScaleConfig:
    """Configuration for scale testing."""

    batch: int
    T: int
    K: int
    C: int
    name: str
    expected_memory_gb: float  # Estimated peak memory


@dataclass
class ScaleResult:
    """Result from a scale benchmark run."""

    config: ScaleConfig
    forward_ms: float
    backward_ms: float
    total_ms: float
    peak_memory_gb: float
    throughput_positions_per_sec: float
    status: str = "success"
    error: Optional[str] = None


# Scale configurations targeting genomics workloads
SCALE_CONFIGS = [
    # Baseline: Should work on most GPUs
    ScaleConfig(batch=2, T=10000, K=100, C=24, name="baseline_10K", expected_memory_gb=2.0),
    # Medium scale: Requires ~8-16GB
    ScaleConfig(batch=2, T=50000, K=500, C=24, name="medium_50K", expected_memory_gb=8.0),
    # Large scale: Target for Phase 3
    ScaleConfig(batch=2, T=100000, K=1000, C=24, name="large_100K", expected_memory_gb=16.0),
    # Very large: Pushing limits
    ScaleConfig(batch=1, T=200000, K=1500, C=24, name="xlarge_200K", expected_memory_gb=24.0),
    # Target genomics scale (may require 48GB+ GPU)
    ScaleConfig(batch=1, T=400000, K=3000, C=24, name="genomics_400K", expected_memory_gb=40.0),
]


def create_scale_inputs(
    config: ScaleConfig, device: torch.device, dtype: torch.dtype = torch.float32
):
    """Create inputs for scale testing."""
    torch.manual_seed(42)

    batch, T, K, C = config.batch, config.T, config.K, config.C

    # Simulate projected encoder features (zero-centered for numerical stability)
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

    # Clean up intermediate tensor
    del projected
    torch.cuda.empty_cache()

    print("Completed creation of scale input.")

    return cum_scores, transition, duration_bias, lengths


def benchmark_scale(
    config: ScaleConfig,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
    use_triton: bool = True,
) -> ScaleResult:
    """Benchmark at scale with memory-conscious approach."""
    from torch_semimarkov.streaming import HAS_TRITON, semi_crf_streaming_forward

    if use_triton and not HAS_TRITON:
        return ScaleResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            throughput_positions_per_sec=0.0,
            status="error",
            error="Triton not available",
        )

    # Check available memory
    torch.cuda.empty_cache()
    gc.collect()
    free_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    if config.expected_memory_gb > free_memory_gb * 0.9:
        return ScaleResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            throughput_positions_per_sec=0.0,
            status="skipped",
            error=f"Expected {config.expected_memory_gb:.1f}GB > available {free_memory_gb:.1f}GB",
        )

    try:
        # Create inputs
        cum_scores, transition, duration_bias, lengths = create_scale_inputs(config, device)
        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # Warmup
        for _ in range(warmup):
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=use_triton
            )
            partition.sum().backward()
            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)

        # Timed runs
        forward_times = []
        backward_times = []

        for _ in range(repeats):
            torch.cuda.synchronize(device)

            start_fwd = time.perf_counter()
            partition = semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, config.K, use_triton=use_triton
            )
            torch.cuda.synchronize(device)
            end_fwd = time.perf_counter()

            start_bwd = time.perf_counter()
            partition.sum().backward()
            torch.cuda.synchronize(device)
            end_bwd = time.perf_counter()

            forward_times.append((end_fwd - start_fwd) * 1000)
            backward_times.append((end_bwd - start_bwd) * 1000)

            cum_scores.grad = None
            transition.grad = None
            duration_bias.grad = None

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)

        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        avg_total = avg_forward + avg_backward

        # Throughput: positions processed per second (batch * T per iteration)
        total_positions = config.batch * config.T
        throughput = total_positions / (avg_total / 1000)  # positions per second

        return ScaleResult(
            config=config,
            forward_ms=avg_forward,
            backward_ms=avg_backward,
            total_ms=avg_total,
            peak_memory_gb=peak_memory,
            throughput_positions_per_sec=throughput,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return ScaleResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            throughput_positions_per_sec=0.0,
            status="oom",
            error="CUDA out of memory",
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return ScaleResult(
            config=config,
            forward_ms=0.0,
            backward_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
            throughput_positions_per_sec=0.0,
            status="error",
            error=str(e),
        )


def print_scale_results(results: list[ScaleResult]):
    """Print scale benchmark results."""
    print("\n" + "=" * 110)
    print("STREAMING SEMI-CRF SCALE BENCHMARK")
    print("=" * 110)

    print(
        f"\n{'Config':<18} | {'B×T×K':<18} | {'Forward':>10} | {'Backward':>10} | {'Total':>10} | {'Memory':>8} | {'Throughput':>15} | {'Status':<8}"
    )
    print("-" * 110)

    for r in results:
        btk = f"{r.config.batch}×{r.config.T//1000}K×{r.config.K}"
        if r.status == "success":
            throughput_str = f"{r.throughput_positions_per_sec/1e6:.2f}M pos/s"
            print(
                f"{r.config.name:<18} | {btk:<18} | {r.forward_ms:>8.1f}ms | {r.backward_ms:>8.1f}ms | {r.total_ms:>8.1f}ms | {r.peak_memory_gb:>6.1f}GB | {throughput_str:>15} | {r.status:<8}"
            )
        else:
            print(
                f"{r.config.name:<18} | {btk:<18} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8} | {'---':>15} | {r.status:<8}"
            )
            if r.error:
                print(f"{'':18} | Error: {r.error}")

    print("-" * 110)

    # Summary
    successful = [r for r in results if r.status == "success"]
    if successful:
        max_scale = max(successful, key=lambda r: r.config.T * r.config.K)
        print(f"\nLargest successful scale: {max_scale.config.name}")
        print(f"  T={max_scale.config.T:,}, K={max_scale.config.K}, batch={max_scale.config.batch}")
        print(f"  Throughput: {max_scale.throughput_positions_per_sec/1e6:.2f}M positions/second")
        print(f"  Memory: {max_scale.peak_memory_gb:.1f}GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to benchmark on")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations")
    parser.add_argument(
        "--configs", type=str, default="all", help="Comma-separated config names or 'all'"
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Also benchmark PyTorch reference (very slow at scale)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    device = torch.device(args.device)
    print("Scale Benchmark")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f}GB"
    )

    # Select configs
    if args.configs == "all":
        configs = SCALE_CONFIGS
    else:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = [c for c in SCALE_CONFIGS if c.name in config_names]

    print(f"\nRunning {len(configs)} scale configurations...")

    results = []
    results_pytorch = []
    for config in configs:
        print(f"\n--- {config.name}: T={config.T:,}, K={config.K}, batch={config.batch} ---")

        torch.cuda.empty_cache()
        gc.collect()

        print("  Running Triton...", end=" ", flush=True)
        result = benchmark_scale(config, device, args.warmup, args.repeats, use_triton=True)
        if result.status == "success":
            print(f"OK ({result.total_ms:.1f}ms, {result.peak_memory_gb:.1f}GB)")
        else:
            print(f"{result.status}")
        results.append(result)

        # Run PyTorch reference for comparison if requested
        if args.pytorch:
            torch.cuda.empty_cache()
            gc.collect()
            print("  Running PyTorch reference...", end=" ", flush=True)
            result_pytorch = benchmark_scale(
                config, device, args.warmup, args.repeats, use_triton=False
            )
            if result_pytorch.status == "success":
                print(
                    f"OK ({result_pytorch.total_ms:.1f}ms, {result_pytorch.peak_memory_gb:.1f}GB)"
                )
            else:
                print(f"{result_pytorch.status}")
            results_pytorch.append(result_pytorch)

    print_scale_results(results)
    if args.pytorch and results_pytorch:
        print("\n\nPYTORCH REFERENCE RESULTS:")
        print_scale_results(results_pytorch)


if __name__ == "__main__":
    main()
