#!/usr/bin/env python3
"""
TIMIT Kernel Benchmark: Triton vs PyTorch Reference

This benchmark measures forward and backward pass performance for Semi-Markov CRF
kernels using TIMIT-specific parameters (C=39, K=30, T~100-500).

Usage:
    # Run all benchmarks
    python benchmark_timit_kernels.py

    # Run specific kernel
    python benchmark_timit_kernels.py --kernel forward
    python benchmark_timit_kernels.py --kernel backward

    # Run with profiler
    python benchmark_timit_kernels.py --profile

    # Run specific config
    python benchmark_timit_kernels.py --batch 32 --seq-len 200 --max-duration 30
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch

# Check for CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for benchmarking")


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    batch: int
    T: int  # Sequence length
    K: int  # Max duration
    C: int  # Number of classes
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"batch={self.batch}, T={self.T}, K={self.K}, C={self.C}"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    triton_forward_ms: float
    triton_backward_ms: float
    pytorch_forward_ms: float
    pytorch_backward_ms: float
    triton_peak_memory_mb: float
    pytorch_peak_memory_mb: float

    @property
    def forward_speedup(self) -> float:
        """Speedup of Triton over PyTorch (>1 means Triton faster)."""
        return self.pytorch_forward_ms / self.triton_forward_ms

    @property
    def backward_speedup(self) -> float:
        """Speedup of Triton over PyTorch (>1 means Triton faster)."""
        return self.pytorch_backward_ms / self.triton_backward_ms

    @property
    def total_speedup(self) -> float:
        """Speedup of Triton over PyTorch for full forward+backward."""
        triton_total = self.triton_forward_ms + self.triton_backward_ms
        pytorch_total = self.pytorch_forward_ms + self.pytorch_backward_ms
        return pytorch_total / triton_total


# Default TIMIT configurations
TIMIT_CONFIGS = [
    BenchmarkConfig(batch=32, T=200, K=30, C=39, name="typical_training"),
    BenchmarkConfig(batch=32, T=500, K=30, C=39, name="long_sequences"),
    BenchmarkConfig(batch=16, T=100, K=30, C=39, name="small_batch"),
    BenchmarkConfig(batch=64, T=300, K=30, C=39, name="large_batch"),
    BenchmarkConfig(batch=32, T=200, K=10, C=39, name="short_duration"),
]


def create_test_inputs(config: BenchmarkConfig, device: str = "cuda"):
    """Create test inputs matching TIMIT parameters."""
    batch, T, K, C = config.batch, config.T, config.K, config.C

    # Cumulative scores: (batch, T+1, C)
    # Simulate projected hidden states with cumsum
    hidden = torch.randn(batch, T, C, device=device, dtype=torch.float32)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 1:] = torch.cumsum(hidden, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1

    # Lengths: (batch,) - use full length for simplicity
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    return cum_scores, transition, duration_bias, lengths


def benchmark_triton_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    warmup: int = 10,
    iterations: int = 50,
) -> tuple[float, float]:
    """
    Benchmark Triton forward pass.

    Returns:
        Tuple of (mean_ms, peak_memory_mb)
    """
    from torch_semimarkov.streaming.triton_forward import (
        HAS_TRITON,
        launch_streaming_triton_kernel,
    )

    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    # Warmup
    for _ in range(warmup):
        partition, ring_ckpts, interval = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            validate_cache=False,
        )
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        partition, ring_ckpts, interval = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            validate_cache=False,
        )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    mean_time = sum(times) / len(times)

    return mean_time, peak_memory


def benchmark_triton_backward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    warmup: int = 10,
    iterations: int = 50,
) -> tuple[float, float]:
    """
    Benchmark Triton backward pass (requires forward first for checkpoints).

    Returns:
        Tuple of (mean_ms, peak_memory_mb)
    """
    from torch_semimarkov.streaming.triton_backward import (
        HAS_TRITON,
        launch_streaming_triton_backward,
    )
    from torch_semimarkov.streaming.triton_forward import (
        launch_streaming_triton_kernel,
    )

    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    batch = cum_scores.shape[0]
    device = cum_scores.device

    # Run forward to get checkpoints
    partition, ring_ckpts, interval = launch_streaming_triton_kernel(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        K,
        semiring="log",
        validate_cache=False,
    )
    torch.cuda.synchronize()

    grad_output = torch.ones(batch, device=device, dtype=cum_scores.dtype)

    # Warmup backward
    for _ in range(warmup):
        launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
            ring_ckpts,
            interval,
            grad_output,
            validate_cache=False,
        )
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark backward only
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
            ring_ckpts,
            interval,
            grad_output,
            validate_cache=False,
        )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    mean_time = sum(times) / len(times)

    return mean_time, peak_memory


def benchmark_pytorch_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    warmup: int = 10,
    iterations: int = 50,
) -> tuple[float, float]:
    """
    Benchmark PyTorch reference forward pass.

    Returns:
        Tuple of (mean_ms, peak_memory_mb)
    """
    from torch_semimarkov.streaming.pytorch_reference import (
        semi_crf_streaming_forward_pytorch,
    )

    # Warmup
    for _ in range(warmup):
        partition, ring_ckpts, interval = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
        )
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        partition, ring_ckpts, interval = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
        )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    mean_time = sum(times) / len(times)

    return mean_time, peak_memory


def benchmark_pytorch_backward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    warmup: int = 10,
    iterations: int = 50,
) -> tuple[float, float]:
    """
    Benchmark PyTorch reference backward pass.

    Returns:
        Tuple of (mean_ms, peak_memory_mb)
    """
    from torch_semimarkov.streaming.pytorch_reference import (
        semi_crf_streaming_backward_pytorch,
        semi_crf_streaming_forward_pytorch,
    )

    # Run forward to get checkpoints
    partition, ring_ckpts, interval = semi_crf_streaming_forward_pytorch(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        K,
    )
    torch.cuda.synchronize()

    # Warmup backward
    # Note: PyTorch version has different signature than Triton version
    # and returns unscaled gradients (no grad_output parameter)
    for _ in range(warmup):
        semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            partition,
            ring_ckpts,
            interval,
            semiring="log",
        )
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark backward only
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            partition,
            ring_ckpts,
            interval,
            semiring="log",
        )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    mean_time = sum(times) / len(times)

    return mean_time, peak_memory


def run_benchmark(config: BenchmarkConfig, kernel: str = "all") -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    print(f"\nBenchmarking: {config.name}")
    print(f"  Config: batch={config.batch}, T={config.T}, K={config.K}, C={config.C}")

    # Create inputs
    cum_scores, transition, duration_bias, lengths = create_test_inputs(config)

    # Initialize results
    triton_fwd_ms = 0.0
    triton_bwd_ms = 0.0
    pytorch_fwd_ms = 0.0
    pytorch_bwd_ms = 0.0
    triton_mem = 0.0
    pytorch_mem = 0.0

    # Run benchmarks
    if kernel in ("all", "forward"):
        print("  Running Triton forward...")
        triton_fwd_ms, triton_fwd_mem = benchmark_triton_forward(
            cum_scores, transition, duration_bias, lengths, config.K
        )
        print(f"    Triton forward: {triton_fwd_ms:.2f} ms")

        print("  Running PyTorch forward...")
        pytorch_fwd_ms, pytorch_fwd_mem = benchmark_pytorch_forward(
            cum_scores, transition, duration_bias, lengths, config.K
        )
        print(f"    PyTorch forward: {pytorch_fwd_ms:.2f} ms")

        fwd_speedup = pytorch_fwd_ms / triton_fwd_ms
        print(
            f"    Forward speedup: {fwd_speedup:.2f}x {'(Triton faster)' if fwd_speedup > 1 else '(PyTorch faster)'}"
        )

    # Clean up between forward and backward
    gc.collect()
    torch.cuda.empty_cache()

    if kernel in ("all", "backward"):
        print("  Running Triton backward...")
        triton_bwd_ms, triton_bwd_mem = benchmark_triton_backward(
            cum_scores, transition, duration_bias, lengths, config.K
        )
        triton_mem = triton_bwd_mem
        print(f"    Triton backward: {triton_bwd_ms:.2f} ms, peak mem: {triton_mem:.1f} MB")

        print("  Running PyTorch backward...")
        pytorch_bwd_ms, pytorch_bwd_mem = benchmark_pytorch_backward(
            cum_scores, transition, duration_bias, lengths, config.K
        )
        pytorch_mem = pytorch_bwd_mem
        print(f"    PyTorch backward: {pytorch_bwd_ms:.2f} ms, peak mem: {pytorch_mem:.1f} MB")

        bwd_speedup = pytorch_bwd_ms / triton_bwd_ms
        print(
            f"    Backward speedup: {bwd_speedup:.2f}x {'(Triton faster)' if bwd_speedup > 1 else '(PyTorch faster)'}"
        )

    return BenchmarkResult(
        config=config,
        triton_forward_ms=triton_fwd_ms,
        triton_backward_ms=triton_bwd_ms,
        pytorch_forward_ms=pytorch_fwd_ms,
        pytorch_backward_ms=pytorch_bwd_ms,
        triton_peak_memory_mb=triton_mem,
        pytorch_peak_memory_mb=pytorch_mem,
    )


def run_profiled_benchmark(config: BenchmarkConfig):
    """Run benchmark with torch.profiler for detailed analysis."""
    from torch.profiler import ProfilerActivity, profile, record_function

    print(f"\nProfiling: {config.name}")
    print(f"  Config: batch={config.batch}, T={config.T}, K={config.K}, C={config.C}")

    # Create inputs
    cum_scores, transition, duration_bias, lengths = create_test_inputs(config)

    # Import kernels
    from torch_semimarkov.streaming.triton_backward import (
        launch_streaming_triton_backward,
    )
    from torch_semimarkov.streaming.triton_forward import (
        launch_streaming_triton_kernel,
    )

    batch = cum_scores.shape[0]
    device = cum_scores.device
    grad_output = torch.ones(batch, device=device, dtype=cum_scores.dtype)

    # Warmup
    print("  Warming up...")
    for _ in range(5):
        partition, ring_ckpts, interval = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            config.K,
            semiring="log",
            validate_cache=False,
        )
        launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
            ring_ckpts,
            interval,
            grad_output,
            validate_cache=False,
        )
        torch.cuda.synchronize()

    # Profile
    print("  Running profiler...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            with record_function("triton_forward"):
                partition, ring_ckpts, interval = launch_streaming_triton_kernel(
                    cum_scores,
                    transition,
                    duration_bias,
                    lengths,
                    config.K,
                    semiring="log",
                    validate_cache=False,
                )
                torch.cuda.synchronize()

            with record_function("triton_backward"):
                launch_streaming_triton_backward(
                    cum_scores,
                    transition,
                    duration_bias,
                    lengths,
                    partition,
                    ring_ckpts,
                    interval,
                    grad_output,
                    validate_cache=False,
                )
                torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("PROFILER RESULTS (sorted by CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n" + "=" * 80)
    print("PROFILER RESULTS (sorted by CPU time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Export trace for viewing in chrome://tracing
    trace_file = f"trace_{config.name}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace exported to: {trace_file}")
    print("Open in chrome://tracing to view detailed timeline")


def print_summary(results: list[BenchmarkResult]):
    """Print summary table of all benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"\n{'Config':<20} {'Triton Fwd':>12} {'PyTorch Fwd':>12} {'Fwd Speedup':>12} "
        f"{'Triton Bwd':>12} {'PyTorch Bwd':>12} {'Bwd Speedup':>12}"
    )
    print("-" * 100)

    for r in results:
        fwd_speedup = r.forward_speedup if r.triton_forward_ms > 0 else 0
        bwd_speedup = r.backward_speedup if r.triton_backward_ms > 0 else 0

        print(
            f"{r.config.name:<20} "
            f"{r.triton_forward_ms:>10.2f}ms "
            f"{r.pytorch_forward_ms:>10.2f}ms "
            f"{fwd_speedup:>11.2f}x "
            f"{r.triton_backward_ms:>10.2f}ms "
            f"{r.pytorch_backward_ms:>10.2f}ms "
            f"{bwd_speedup:>11.2f}x"
        )

    print("-" * 100)

    # Overall summary
    total_triton_fwd = sum(r.triton_forward_ms for r in results)
    total_pytorch_fwd = sum(r.pytorch_forward_ms for r in results)
    total_triton_bwd = sum(r.triton_backward_ms for r in results)
    total_pytorch_bwd = sum(r.pytorch_backward_ms for r in results)

    if total_triton_fwd > 0:
        avg_fwd_speedup = total_pytorch_fwd / total_triton_fwd
        print(f"\nAverage forward speedup: {avg_fwd_speedup:.2f}x")

    if total_triton_bwd > 0:
        avg_bwd_speedup = total_pytorch_bwd / total_triton_bwd
        print(f"Average backward speedup: {avg_bwd_speedup:.2f}x")

    if total_triton_fwd > 0 and total_triton_bwd > 0:
        total_speedup = (total_pytorch_fwd + total_pytorch_bwd) / (
            total_triton_fwd + total_triton_bwd
        )
        print(f"Overall speedup: {total_speedup:.2f}x")

    if avg_fwd_speedup < 1 or avg_bwd_speedup < 1:
        print("\n*** NOTE: Triton is SLOWER than PyTorch reference ***")
        print("Bottleneck investigation recommended. Run with --profile for details.")


def main():
    parser = argparse.ArgumentParser(description="TIMIT Kernel Benchmark")
    parser.add_argument(
        "--kernel",
        choices=["forward", "backward", "all"],
        default="all",
        help="Which kernel to benchmark",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run with torch.profiler for detailed analysis"
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--max-duration", type=int, default=None, help="Override max duration K")
    parser.add_argument(
        "--num-classes", type=int, default=39, help="Number of classes (default: 39 for TIMIT)"
    )
    parser.add_argument("--config", type=str, default=None, help="Run specific config by name")
    args = parser.parse_args()

    print("=" * 80)
    print("TIMIT Kernel Benchmark: Triton vs PyTorch Reference")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Determine configs to run
    if args.batch is not None or args.seq_len is not None or args.max_duration is not None:
        # Custom single config
        configs = [
            BenchmarkConfig(
                batch=args.batch or 32,
                T=args.seq_len or 200,
                K=args.max_duration or 30,
                C=args.num_classes,
                name="custom",
            )
        ]
    elif args.config:
        # Named config
        configs = [c for c in TIMIT_CONFIGS if c.name == args.config]
        if not configs:
            print(f"Unknown config: {args.config}")
            print(f"Available configs: {[c.name for c in TIMIT_CONFIGS]}")
            return
    else:
        # All default configs
        configs = TIMIT_CONFIGS

    if args.profile:
        # Run profiled benchmark on first config only
        run_profiled_benchmark(configs[0])
    else:
        # Run all benchmarks
        results = []
        for config in configs:
            result = run_benchmark(config, kernel=args.kernel)
            results.append(result)
            gc.collect()
            torch.cuda.empty_cache()

        print_summary(results)


if __name__ == "__main__":
    main()
