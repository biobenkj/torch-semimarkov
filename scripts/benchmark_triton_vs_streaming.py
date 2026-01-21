#!/usr/bin/env python
"""Benchmark: triton_scan vs streaming API.

Compares performance when the full edge tensor fits in GPU memory.

- triton_scan: Uses pre-computed edge tensor (batch, T-1, K, C, C)
- streaming: Computes edges on-the-fly from cum_scores (batch, T+1, C)

Usage:
    # Single batch size comparison
    python scripts/benchmark_triton_vs_streaming.py
    python scripts/benchmark_triton_vs_streaming.py --T 500 --K 32 --C 24 --batch 8

    # Batch scaling test to find optimal batch size
    python scripts/benchmark_triton_vs_streaming.py --batch-scaling
    python scripts/benchmark_triton_vs_streaming.py --batch-scaling --T 500 --K 32 --C 24

Requirements:
    - CUDA GPU
    - torch-semimarkov installed
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_semimarkov.streaming import semi_crf_streaming_forward
from torch_semimarkov.triton_scan import semi_crf_triton_forward


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    partition_value: float
    memory_mb: float


def compute_edge_from_cumscores(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Materialize the full edge tensor from streaming inputs.

    This constructs the same edge tensor that the streaming kernel computes
    on-the-fly, allowing fair comparison between the two approaches.

    Args:
        cum_scores: (batch, T+1, C) cumulative projected scores
        transition: (C, C) label transition matrix
        duration_bias: (K, C) duration-specific bias

    Returns:
        edge: (batch, T-1, K, C, C) edge potentials
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1

    # Allocate edge tensor
    edge = torch.full(
        (batch, T - 1, K, C, C),
        -1e9,
        device=cum_scores.device,
        dtype=cum_scores.dtype,
    )

    for t_end in range(1, T):
        for k in range(1, min(K, t_end + 1)):
            t_start = t_end - k

            # Content score: sum of projected scores from t_start to t_end-1
            # content[b, c] = cum_scores[b, t_end, c] - cum_scores[b, t_start, c]
            content = cum_scores[:, t_end, :] - cum_scores[:, t_start, :]  # (batch, C)

            # Duration bias for this k
            dur_bias = duration_bias[k, :]  # (C,)

            # Combine: edge[b, t_start, k, c_dest, c_src] = content[b, c_dest] + dur_bias[c_dest] + trans[c_src, c_dest]
            # Note: transition convention varies - we use trans[c_src, c_dest]
            edge_val = content.unsqueeze(-1) + dur_bias.unsqueeze(-1) + transition.T.unsqueeze(0)
            edge[:, t_start, k, :, :] = edge_val

    return edge


def warmup_cuda(device: torch.device, iterations: int = 10):
    """Warm up CUDA to get stable timing."""
    x = torch.randn(1000, 1000, device=device)
    for _ in range(iterations):
        _ = x @ x
    torch.cuda.synchronize(device)


def benchmark_forward(
    fn: Callable,
    *args,
    device: torch.device | None = None,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs,
) -> tuple[float, torch.Tensor]:
    """Benchmark forward pass timing.

    Returns:
        (time_ms, result): Average time per call in milliseconds and result tensor
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize(device)

    # Timed runs
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000, result


def benchmark_backward(
    fn: Callable,
    grad_inputs: list[torch.Tensor],
    *args,
    device: torch.device | None = None,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs,
) -> float:
    """Benchmark backward pass timing.

    Args:
        fn: Forward function
        grad_inputs: List of tensors that require gradients
        device: CUDA device to synchronize

    Returns:
        Average time per backward call in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        for t in grad_inputs:
            if t.grad is not None:
                t.grad.zero_()
        result = fn(*args, **kwargs)
        loss = result.sum()
        loss.backward()
        torch.cuda.synchronize(device)

    # Timed runs
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        for t in grad_inputs:
            if t.grad is not None:
                t.grad.zero_()
        result = fn(*args, **kwargs)
        loss = result.sum()
        loss.backward()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def run_triton_scan_benchmark(
    edge: torch.Tensor,
    lengths: torch.Tensor,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark triton_scan API (pre-computed edge tensor)."""
    # Memory for edge tensor
    memory_mb = edge.numel() * edge.element_size() / (1024 * 1024)

    # Forward (inference)
    forward_time, partition = benchmark_forward(
        semi_crf_triton_forward,
        edge,
        lengths,
        warmup=warmup,
        iterations=iterations,
        use_triton=True,
    )

    # Forward + Backward (training)
    edge_train = edge.detach().clone().requires_grad_(True)
    backward_time = benchmark_backward(
        semi_crf_triton_forward,
        [edge_train],
        edge_train,
        lengths,
        warmup=warmup,
        iterations=iterations,
        use_triton=True,
        use_compile=True,
    )

    return BenchmarkResult(
        name="triton_scan (pre-computed edge)",
        forward_time_ms=forward_time,
        backward_time_ms=backward_time,
        total_time_ms=forward_time + backward_time,
        partition_value=partition.mean().item(),
        memory_mb=memory_mb,
    )


def run_streaming_benchmark(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark streaming API (on-the-fly edge computation)."""
    # Memory for streaming inputs (much smaller than edge tensor)
    memory_mb = (
        cum_scores.numel() * cum_scores.element_size()
        + transition.numel() * transition.element_size()
        + duration_bias.numel() * duration_bias.element_size()
    ) / (1024 * 1024)

    # Forward (inference)
    forward_time, partition = benchmark_forward(
        semi_crf_streaming_forward,
        cum_scores,
        transition,
        duration_bias,
        lengths,
        K,
        warmup=warmup,
        iterations=iterations,
        use_triton=True,
    )

    # Forward + Backward (training)
    cum_scores_train = cum_scores.detach().clone().requires_grad_(True)
    transition_train = transition.detach().clone().requires_grad_(True)
    duration_bias_train = duration_bias.detach().clone().requires_grad_(True)

    backward_time = benchmark_backward(
        semi_crf_streaming_forward,
        [cum_scores_train, transition_train, duration_bias_train],
        cum_scores_train,
        transition_train,
        duration_bias_train,
        lengths,
        K,
        warmup=warmup,
        iterations=iterations,
        use_triton=True,
    )

    return BenchmarkResult(
        name="streaming (on-the-fly edge)",
        forward_time_ms=forward_time,
        backward_time_ms=backward_time,
        total_time_ms=forward_time + backward_time,
        partition_value=partition.mean().item(),
        memory_mb=memory_mb,
    )


def verify_correctness(
    edge: torch.Tensor,
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> tuple[bool, float]:
    """Verify both methods produce the same result."""
    with torch.no_grad():
        result_triton = semi_crf_triton_forward(edge, lengths, use_triton=True)
        result_streaming = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )

    max_diff = (result_triton - result_streaming).abs().max().item()
    is_close = torch.allclose(result_triton, result_streaming, rtol=rtol, atol=atol)

    return is_close, max_diff


@dataclass
class BatchScalingResult:
    """Results from batch scaling test."""

    batch_size: int
    forward_time_ms: float
    backward_time_ms: float
    throughput_seq_per_sec: float
    memory_mb: float
    error: str | None = None  # None=success, "OOM", "KERNEL_ERROR"


def estimate_edge_memory_mb(batch: int, T: int, K: int, C: int) -> float:
    """Estimate memory for edge tensor in MB."""
    # edge shape: (batch, T-1, K, C, C), float32
    return batch * (T - 1) * K * C * C * 4 / (1024 * 1024)


def _reset_cuda_state(device: torch.device) -> None:
    """Attempt to reset CUDA state after an error."""
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_error(e: Exception) -> bool:
    """Check if an exception is a CUDA-related error."""
    err_str = str(e).lower()
    err_type = type(e).__name__.lower()
    return (
        "out of memory" in err_str
        or "illegal" in err_str
        or "cuda" in err_str
        or "accelerator" in err_type
    )


def _is_oom_error(e: Exception) -> bool:
    """Check if an exception is specifically an out-of-memory error."""
    err_str = str(e).lower()
    return "out of memory" in err_str


def run_batch_scaling_test(
    T: int,
    K: int,
    C: int,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int = 3,
    iterations: int = 10,
    mode: str = "forward",  # "forward" or "backward"
) -> tuple[list[BatchScalingResult], list[BatchScalingResult]]:
    """Run batch scaling test for both triton_scan and streaming.

    Tests all streaming batch sizes first, then all triton_scan batch sizes.
    This prevents triton_scan OOMs from corrupting GPU state and affecting
    streaming results.

    Args:
        T: Sequence length
        K: Max segment duration
        C: Number of classes
        device: CUDA device
        batch_sizes: List of batch sizes to test
        warmup: Warmup iterations
        iterations: Timing iterations
        mode: "forward" for inference, "backward" for training

    Returns:
        (triton_results, streaming_results): Lists of BatchScalingResult
    """
    streaming_results = []
    triton_results = []

    # Shared parameters (don't change with batch)
    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1

    # Phase 1: Test all streaming batch sizes
    print("  Phase 1: Testing streaming API...")
    streaming_error: str | None = None

    for batch in batch_sizes:
        if streaming_error:
            streaming_results.append(
                BatchScalingResult(
                    batch_size=batch,
                    forward_time_ms=0,
                    backward_time_ms=0,
                    throughput_seq_per_sec=0,
                    memory_mb=0,
                    error=streaming_error,
                )
            )
            continue

        print(f"    batch={batch}...")
        torch.cuda.empty_cache()

        projected = torch.randn(batch, T, C, device=device, dtype=torch.float32)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        lengths = torch.full((batch,), T, device=device, dtype=torch.long)

        try:
            if mode == "forward":
                time_ms, _ = benchmark_forward(
                    semi_crf_streaming_forward,
                    cum_scores,
                    transition,
                    duration_bias,
                    lengths,
                    K,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    use_triton=True,
                )
            else:  # backward
                cum_scores_train = cum_scores.detach().clone().requires_grad_(True)
                transition_train = transition.detach().clone().requires_grad_(True)
                duration_bias_train = duration_bias.detach().clone().requires_grad_(True)
                time_ms = benchmark_backward(
                    semi_crf_streaming_forward,
                    [cum_scores_train, transition_train, duration_bias_train],
                    cum_scores_train,
                    transition_train,
                    duration_bias_train,
                    lengths,
                    K,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    use_triton=True,
                )

            throughput = (batch / time_ms) * 1000  # sequences per second
            memory_mb = (
                cum_scores.numel() * 4 + transition.numel() * 4 + duration_bias.numel() * 4
            ) / (1024 * 1024)
            streaming_results.append(
                BatchScalingResult(
                    batch_size=batch,
                    forward_time_ms=time_ms if mode == "forward" else 0,
                    backward_time_ms=time_ms if mode == "backward" else 0,
                    throughput_seq_per_sec=throughput,
                    memory_mb=memory_mb,
                )
            )
        except Exception as e:
            if _is_cuda_error(e):
                is_oom = _is_oom_error(e)
                streaming_error = "OOM" if is_oom else "KERNEL_ERROR"
                err_msg = str(e).split("\n")[0][:80]
                print(f"      streaming {streaming_error}: {type(e).__name__}")
                print(f"        Error: {err_msg}")
                streaming_results.append(
                    BatchScalingResult(
                        batch_size=batch,
                        forward_time_ms=0,
                        backward_time_ms=0,
                        throughput_seq_per_sec=0,
                        memory_mb=0,
                        error=streaming_error,
                    )
                )
                _reset_cuda_state(device)
            else:
                raise

        del projected, cum_scores, lengths
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # GPU state may be corrupted after error

    # Phase 2: Test all triton_scan batch sizes
    print("  Phase 2: Testing triton_scan API...")
    triton_error: str | None = None

    for batch in batch_sizes:
        edge_memory_mb = estimate_edge_memory_mb(batch, T, K, C)

        if triton_error:
            triton_results.append(
                BatchScalingResult(
                    batch_size=batch,
                    forward_time_ms=0,
                    backward_time_ms=0,
                    throughput_seq_per_sec=0,
                    memory_mb=edge_memory_mb,
                    error=triton_error,
                )
            )
            continue

        print(f"    batch={batch}...")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # GPU state may be corrupted after error

        projected = torch.randn(batch, T, C, device=device, dtype=torch.float32)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        lengths = torch.full((batch,), T, device=device, dtype=torch.long)

        try:
            edge = compute_edge_from_cumscores(cum_scores, transition, duration_bias, K)

            if mode == "forward":
                time_ms, _ = benchmark_forward(
                    semi_crf_triton_forward,
                    edge,
                    lengths,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    use_triton=True,
                )
            else:  # backward
                edge_train = edge.detach().clone().requires_grad_(True)
                time_ms = benchmark_backward(
                    semi_crf_triton_forward,
                    [edge_train],
                    edge_train,
                    lengths,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    use_triton=True,
                    use_compile=True,
                )

            throughput = (batch / time_ms) * 1000
            triton_results.append(
                BatchScalingResult(
                    batch_size=batch,
                    forward_time_ms=time_ms if mode == "forward" else 0,
                    backward_time_ms=time_ms if mode == "backward" else 0,
                    throughput_seq_per_sec=throughput,
                    memory_mb=edge_memory_mb,
                )
            )
            del edge
        except Exception as e:
            if _is_cuda_error(e):
                is_oom = _is_oom_error(e)
                triton_error = "OOM" if is_oom else "KERNEL_ERROR"
                err_msg = str(e).split("\n")[0][:80]
                print(
                    f"      triton_scan {triton_error} "
                    f"(edge would need {edge_memory_mb:.0f} MB): {type(e).__name__}"
                )
                print(f"        Error: {err_msg}")
                triton_results.append(
                    BatchScalingResult(
                        batch_size=batch,
                        forward_time_ms=0,
                        backward_time_ms=0,
                        throughput_seq_per_sec=0,
                        memory_mb=edge_memory_mb,
                        error=triton_error,
                    )
                )
                _reset_cuda_state(device)
            else:
                raise

        del projected, cum_scores, lengths
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # GPU state may be corrupted after error

    return triton_results, streaming_results


def format_batch_scaling_results(
    triton_results: list[BatchScalingResult],
    streaming_results: list[BatchScalingResult],
    config: dict,
    mode: str,
) -> str:
    """Format batch scaling results as a table."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"BATCH SCALING TEST ({mode.upper()})")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Configuration:")
    for k, v in config.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Table header
    lines.append(f"{'Batch':<8} {'triton_scan':<40} {'streaming':<40}")
    lines.append(
        f"{'':8} {'Time(ms)':<12} {'Thru(seq/s)':<14} {'Mem(MB)':<12} "
        f"{'Time(ms)':<12} {'Thru(seq/s)':<14} {'Mem(MB)':<12}"
    )
    lines.append("-" * 100)

    # Find best throughput for each
    best_triton_throughput = 0
    best_triton_batch = 0
    best_streaming_throughput = 0
    best_streaming_batch = 0

    for tr, sr in zip(triton_results, streaming_results, strict=True):
        batch = tr.batch_size

        # triton_scan columns
        if tr.error:
            # Show error type: OOM or ERR (kernel error)
            err_label = "OOM" if tr.error == "OOM" else "ERR"
            triton_str = f"{err_label:<12} {'-':<14} {tr.memory_mb:<12.0f}"
        else:
            time_ms = tr.forward_time_ms if mode == "forward" else tr.backward_time_ms
            triton_str = (
                f"{time_ms:<12.2f} {tr.throughput_seq_per_sec:<14.1f} {tr.memory_mb:<12.0f}"
            )
            if tr.throughput_seq_per_sec > best_triton_throughput:
                best_triton_throughput = tr.throughput_seq_per_sec
                best_triton_batch = batch

        # streaming columns
        if sr.error:
            err_label = "OOM" if sr.error == "OOM" else "ERR"
            streaming_str = f"{err_label:<12} {'-':<14} {sr.memory_mb:<12.0f}"
        else:
            time_ms = sr.forward_time_ms if mode == "forward" else sr.backward_time_ms
            streaming_str = (
                f"{time_ms:<12.2f} {sr.throughput_seq_per_sec:<14.1f} {sr.memory_mb:<12.0f}"
            )
            if sr.throughput_seq_per_sec > best_streaming_throughput:
                best_streaming_throughput = sr.throughput_seq_per_sec
                best_streaming_batch = batch

        lines.append(f"{batch:<8} {triton_str} {streaming_str}")

    lines.append("-" * 100)
    lines.append("")
    lines.append("Summary:")
    lines.append(
        f"  triton_scan optimal: batch={best_triton_batch}, "
        f"throughput={best_triton_throughput:.1f} seq/s"
    )
    lines.append(
        f"  streaming optimal:   batch={best_streaming_batch}, "
        f"throughput={best_streaming_throughput:.1f} seq/s"
    )

    if best_triton_throughput > 0:
        ratio = best_streaming_throughput / best_triton_throughput
        lines.append(f"  Peak throughput ratio: streaming is {ratio:.2f}x of triton_scan")
    lines.append("=" * 100)

    return "\n".join(lines)


def format_results(results: list[BenchmarkResult], config: dict) -> str:
    """Format benchmark results as a table."""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK: triton_scan vs streaming API")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration:")
    for k, v in config.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Table header
    lines.append(f"{'Method':<35} {'Forward':<12} {'Backward':<12} {'Total':<12} {'Memory':<12}")
    lines.append(f"{'':35} {'(ms)':<12} {'(ms)':<12} {'(ms)':<12} {'(MB)':<12}")
    lines.append("-" * 80)

    for r in results:
        lines.append(
            f"{r.name:<35} {r.forward_time_ms:<12.3f} {r.backward_time_ms:<12.3f} "
            f"{r.total_time_ms:<12.3f} {r.memory_mb:<12.2f}"
        )

    lines.append("-" * 80)

    # Speedup comparison
    if len(results) == 2:
        fwd_speedup = results[0].forward_time_ms / results[1].forward_time_ms
        bwd_speedup = results[0].backward_time_ms / results[1].backward_time_ms
        total_speedup = results[0].total_time_ms / results[1].total_time_ms
        mem_ratio = results[0].memory_mb / results[1].memory_mb

        lines.append("")
        lines.append("Comparison (streaming relative to triton_scan):")
        lines.append(f"  Forward speedup:  {fwd_speedup:.2f}x")
        lines.append(f"  Backward speedup: {bwd_speedup:.2f}x")
        lines.append(f"  Total speedup:    {total_speedup:.2f}x")
        lines.append(
            f"  Memory ratio:     {mem_ratio:.1f}x (triton_scan uses {mem_ratio:.1f}x more)"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark triton_scan vs streaming API")
    parser.add_argument("--T", type=int, default=1000, help="Sequence length")
    parser.add_argument("--K", type=int, default=32, help="Max segment duration")
    parser.add_argument("--C", type=int, default=24, help="Number of classes")
    parser.add_argument(
        "--batch", type=int, default=4, help="Batch size (ignored if --batch-scaling)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Timing iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--batch-scaling",
        action="store_true",
        help="Run batch scaling test to find optimal batch size",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024",
        help="Comma-separated batch sizes for scaling test (default: 1,2,4,...,1024)",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        choices=["forward", "backward", "both"],
        default="forward",
        help="Mode for batch scaling: forward (inference), backward (training), or both",
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return

    # Parse device
    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"ERROR: Device must be a CUDA device, got {args.device}")
        return

    # Set the device as current for operations
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"Running on: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    T, K, C = args.T, args.K, args.C

    # Batch scaling mode
    if args.batch_scaling:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        print(f"Batch scaling test: T={T}, K={K}, C={C}")
        print(f"Testing batch sizes: {batch_sizes}")
        print()

        warmup_cuda(device)

        config = {
            "Sequence length (T)": T,
            "Max duration (K)": K,
            "Num classes (C)": C,
            "Batch sizes": args.batch_sizes,
            "Warmup iterations": args.warmup,
            "Timing iterations": args.iterations,
        }

        modes = ["forward", "backward"] if args.scaling_mode == "both" else [args.scaling_mode]

        for mode in modes:
            print(f"Running {mode} scaling test...")
            triton_results, streaming_results = run_batch_scaling_test(
                T,
                K,
                C,
                device,
                batch_sizes,
                warmup=args.warmup,
                iterations=args.iterations,
                mode=mode,
            )
            print()
            print(format_batch_scaling_results(triton_results, streaming_results, config, mode))
            print()

        return

    # Single batch comparison mode
    batch = args.batch

    # Projected scores (what an encoder would output)
    projected = torch.randn(batch, T, C, device=device, dtype=torch.float32)
    projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center

    # Cumulative scores for streaming API
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition and duration bias
    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1

    # Sequence lengths
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    # Compute edge tensor for triton_scan
    print("Computing edge tensor for triton_scan...")
    edge = compute_edge_from_cumscores(cum_scores, transition, duration_bias, K)
    print(f"Edge tensor shape: {edge.shape}")
    print(f"Edge tensor memory: {edge.numel() * 4 / 1024 / 1024:.2f} MB")
    print()

    # Verify correctness
    print("Verifying correctness...")
    is_correct, max_diff = verify_correctness(
        edge, cum_scores, transition, duration_bias, lengths, K
    )
    print(f"  Results match: {is_correct}")
    print(f"  Max difference: {max_diff:.2e}")
    if not is_correct:
        print("  WARNING: Results differ significantly!")
    print()

    # Warmup CUDA
    print("Warming up CUDA...")
    warmup_cuda(device)
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print()

    results = []

    # triton_scan benchmark
    print("  Benchmarking triton_scan...")
    result_triton = run_triton_scan_benchmark(
        edge, lengths, warmup=args.warmup, iterations=args.iterations
    )
    results.append(result_triton)

    # streaming benchmark
    print("  Benchmarking streaming...")
    result_streaming = run_streaming_benchmark(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        K,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    results.append(result_streaming)

    # Print results
    print()
    config = {
        "Sequence length (T)": T,
        "Max duration (K)": K,
        "Num classes (C)": C,
        "Batch size": batch,
        "Warmup iterations": args.warmup,
        "Timing iterations": args.iterations,
    }
    print(format_results(results, config))


if __name__ == "__main__":
    main()
