"""Benchmark execution logic."""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import EntropySemiring, LogSemiring, MaxSemiring

from .memory import bytes_to_gb, estimate_memory_breakdown

# Mapping of semiring names to classes
SEMIRING_MAP = {
    "Log": LogSemiring,
    "Max": MaxSemiring,
    "Entropy": EntropySemiring,
}

# Edge-tensor backends (use SemiMarkov class with pre-computed edges)
EDGE_TENSOR_BACKENDS = {
    "binary_tree",
    "binary_tree_sharded",
    "banded",
    "block_triangular",
    "linear_scan",
    "linear_scan_vectorized",
    "linear_scan_streaming",
}

# Streaming backends (compute edges on-the-fly)
STREAMING_BACKENDS = {"triton_streaming"}

# Backends supporting all semirings (Log, Max, Entropy)
ALL_SEMIRING_BACKENDS = {
    "binary_tree",
    "binary_tree_sharded",
    "linear_scan",
    "linear_scan_vectorized",
    "linear_scan_streaming",
    "block_triangular",
}

# Backends that only work with LogSemiring
LOG_SEMIRING_ONLY_BACKENDS = {"banded"}

# Backends that support Log and Max semirings only
LOG_MAX_BACKENDS = {"triton_streaming"}


@dataclass
class BenchmarkResult:
    """Single benchmark result with full metrics."""

    T: int
    K: int
    C: int
    B: int
    KC: int  # state-space size
    backend: str
    semiring: str  # "Log", "Max", "Entropy", etc.
    phase: str  # "forward", "backward", "both"

    # Timing (all runs)
    time_ms_median: float
    time_ms_iqr_low: float
    time_ms_iqr_high: float
    time_per_position_ms: float  # time_ms_median / T

    # Memory in GB (consistent units)
    peak_allocated_gb: float
    peak_reserved_gb: float

    # Status
    status: str  # "success", "oom", "not_tested", "error"
    error_msg: str = ""

    # Memory breakdown (if available)
    memory_potentials_gb: float = 0.0
    memory_dp_state_gb: float = 0.0
    memory_workspace_gb: float = 0.0
    memory_autograd_gb: float = 0.0


def run_single_benchmark(
    T: int,
    K: int,
    C: int,
    B: int,
    backend: str,
    device: torch.device,
    repeats: int = 5,
    semiring_name: str = "Log",
    phase: str = "both",
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        T: Sequence length
        K: Maximum duration
        C: Number of labels
        B: Batch size
        backend: Backend name (linear_scan, binary_tree, triton_streaming, etc.)
        device: Torch device
        repeats: Number of repetitions for timing
        semiring_name: Semiring to use ("Log", "Max", "Entropy")
        phase: "forward" (forward only), "backward" (backward only), or "both"
    """
    KC = K * C
    result_base = {
        "T": T,
        "K": K,
        "C": C,
        "B": B,
        "KC": KC,
        "backend": backend,
        "semiring": semiring_name,
        "phase": phase,
    }

    # Estimate memory breakdown
    breakdown = estimate_memory_breakdown(T, K, C, B, backend)

    # Route to appropriate benchmark function
    if backend in STREAMING_BACKENDS:
        return _run_streaming_benchmark(
            T, K, C, B, backend, device, repeats, semiring_name, phase, result_base, breakdown
        )
    elif backend in EDGE_TENSOR_BACKENDS:
        return _run_edge_tensor_benchmark(
            T, K, C, B, backend, device, repeats, semiring_name, phase, result_base, breakdown
        )
    else:
        return BenchmarkResult(
            **result_base,
            time_ms_median=float("nan"),
            time_ms_iqr_low=float("nan"),
            time_ms_iqr_high=float("nan"),
            time_per_position_ms=float("nan"),
            peak_allocated_gb=float("nan"),
            peak_reserved_gb=float("nan"),
            status="error",
            error_msg=f"Unknown backend: {backend}",
        )


def _run_edge_tensor_benchmark(
    T: int,
    K: int,
    C: int,
    B: int,
    backend: str,
    device: torch.device,
    repeats: int,
    semiring_name: str,
    phase: str,
    result_base: dict,
    breakdown: dict,
) -> BenchmarkResult:
    """Benchmark edge-tensor backends (SemiMarkov class methods)."""
    # Get semiring class
    semiring_cls = SEMIRING_MAP.get(semiring_name, LogSemiring)

    try:
        # Create struct with appropriate semiring
        struct = SemiMarkov(semiring_cls)

        # Create potentials: (B, T-1, K, C, C)
        edge = torch.randn(B, T - 1, K, C, C, device=device, requires_grad=False)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        times_ms = []
        peak_allocated = 0
        peak_reserved = 0

        def run_backend_forward(edge_input, struct_to_use):
            """Run forward pass for edge-tensor backends."""
            if backend == "binary_tree":
                v, _, _ = struct_to_use._dp_binary_tree(edge_input, lengths, force_grad=True)
                return v
            elif backend == "banded":
                v, _, _ = struct_to_use._dp_banded(edge_input, lengths, force_grad=True)
                return v
            elif backend == "block_triangular":
                v, _, _ = struct_to_use._dp_blocktriangular(edge_input, lengths, force_grad=True)
                return v
            elif backend == "linear_scan":
                v, _, _ = struct_to_use._dp_standard(edge_input, lengths, force_grad=True)
                return v
            elif backend == "linear_scan_vectorized":
                v, _, _ = struct_to_use._dp_standard_vectorized(
                    edge_input, lengths, force_grad=True
                )
                return v
            elif backend == "linear_scan_streaming":
                v, _, _ = struct_to_use._dp_scan_streaming(edge_input, lengths, force_grad=True)
                return v
            elif backend == "binary_tree_sharded":
                # Use CheckpointShardSemiring to reduce peak memory at cost of time
                from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring

                ShardedSemiring = CheckpointShardSemiring(
                    type(struct_to_use.semiring), max_size=10000
                )
                struct_sharded = SemiMarkov(ShardedSemiring)
                v, _, _ = struct_sharded._dp_binary_tree(edge_input, lengths, force_grad=True)
                return v
            else:
                raise ValueError(f"Unknown edge-tensor backend: {backend}")

        # Warmup run
        if device.type == "cuda":
            warmup_needs_grad = phase in ("backward", "both")
            edge_warmup = edge.clone().detach().requires_grad_(warmup_needs_grad)
            try:
                v_warm = run_backend_forward(edge_warmup, struct)
                if phase in ("backward", "both"):
                    v_warm.sum().backward()
                del edge_warmup, v_warm
            except Exception:
                pass
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()

        for _rep in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            gc.collect()

            needs_grad = phase in ("backward", "both")
            edge_run = edge.clone().detach().requires_grad_(needs_grad)

            if phase == "forward":
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                loss = None
            elif phase == "backward":
                v = run_backend_forward(edge_run, struct)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            else:  # phase == "both"
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct)
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times_ms.append(elapsed_ms)

            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))

            del edge_run, v
            if loss is not None:
                del loss
            gc.collect()

        # Compute statistics
        times_sorted = sorted(times_ms)
        n = len(times_sorted)
        median = statistics.median(times_sorted)
        q1 = times_sorted[n // 4] if n >= 4 else times_sorted[0]
        q3 = times_sorted[3 * n // 4] if n >= 4 else times_sorted[-1]

        return BenchmarkResult(
            **result_base,
            time_ms_median=round(median, 2),
            time_ms_iqr_low=round(q1, 2),
            time_ms_iqr_high=round(q3, 2),
            time_per_position_ms=round(median / T, 4),
            peak_allocated_gb=bytes_to_gb(peak_allocated),
            peak_reserved_gb=bytes_to_gb(peak_reserved),
            status="success",
            memory_potentials_gb=breakdown["potentials_gb"],
            memory_dp_state_gb=breakdown["dp_state_gb"],
            memory_workspace_gb=breakdown["workspace_gb"],
            memory_autograd_gb=breakdown["autograd_gb"],
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="oom",
                error_msg=str(e)[:100],
                memory_potentials_gb=breakdown["potentials_gb"],
                memory_dp_state_gb=breakdown["dp_state_gb"],
                memory_workspace_gb=breakdown["workspace_gb"],
                memory_autograd_gb=breakdown["autograd_gb"],
            )
        else:
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="error",
                error_msg=str(e)[:100],
            )
    except NotImplementedError as e:
        return BenchmarkResult(
            **result_base,
            time_ms_median=float("nan"),
            time_ms_iqr_low=float("nan"),
            time_ms_iqr_high=float("nan"),
            time_per_position_ms=float("nan"),
            peak_allocated_gb=float("nan"),
            peak_reserved_gb=float("nan"),
            status="not_implemented",
            error_msg=str(e)[:100],
        )
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_streaming_benchmark(
    T: int,
    K: int,
    C: int,
    B: int,
    backend: str,
    device: torch.device,
    repeats: int,
    semiring_name: str,
    phase: str,
    result_base: dict,
    breakdown: dict,
) -> BenchmarkResult:
    """Benchmark streaming backends (compute edges on-the-fly)."""
    try:
        from torch_semimarkov.streaming import HAS_TRITON, semi_crf_streaming_forward
    except ImportError:
        return BenchmarkResult(
            **result_base,
            time_ms_median=float("nan"),
            time_ms_iqr_low=float("nan"),
            time_ms_iqr_high=float("nan"),
            time_per_position_ms=float("nan"),
            peak_allocated_gb=float("nan"),
            peak_reserved_gb=float("nan"),
            status="error",
            error_msg="streaming module not available",
        )

    try:
        # Create streaming inputs (cumulative scores, not edge tensor)
        projected = torch.randn(B, T, C, device=device)
        cum_scores = torch.zeros(B, T + 1, C, device=device, requires_grad=False)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        transition = torch.randn(C, C, device=device) * 0.1
        duration_bias = torch.randn(K, C, device=device) * 0.1
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        times_ms = []
        peak_allocated = 0
        peak_reserved = 0

        # Map semiring name to streaming semiring string
        streaming_semiring = semiring_name.lower()

        def run_streaming_forward(cum_scores_input, needs_grad=False):
            """Run forward pass for streaming backend."""
            if needs_grad:
                cum_scores_input = cum_scores_input.clone().requires_grad_(True)
            return semi_crf_streaming_forward(
                cum_scores_input,
                transition,
                duration_bias,
                lengths,
                K,
                semiring=streaming_semiring,
                use_triton=HAS_TRITON,
            )

        # Warmup run
        if device.type == "cuda":
            warmup_needs_grad = phase in ("backward", "both")
            try:
                v_warm = run_streaming_forward(cum_scores, needs_grad=warmup_needs_grad)
                if phase in ("backward", "both"):
                    v_warm.sum().backward()
                del v_warm
            except Exception:
                pass
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()

        for _rep in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            gc.collect()

            if phase == "forward":
                t0 = time.perf_counter()
                v = run_streaming_forward(cum_scores, needs_grad=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                loss = None
            elif phase == "backward":
                v = run_streaming_forward(cum_scores, needs_grad=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            else:  # phase == "both"
                t0 = time.perf_counter()
                v = run_streaming_forward(cum_scores, needs_grad=True)
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times_ms.append(elapsed_ms)

            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))

            del v
            if loss is not None:
                del loss
            gc.collect()

        # Compute statistics
        times_sorted = sorted(times_ms)
        n = len(times_sorted)
        median = statistics.median(times_sorted)
        q1 = times_sorted[n // 4] if n >= 4 else times_sorted[0]
        q3 = times_sorted[3 * n // 4] if n >= 4 else times_sorted[-1]

        return BenchmarkResult(
            **result_base,
            time_ms_median=round(median, 2),
            time_ms_iqr_low=round(q1, 2),
            time_ms_iqr_high=round(q3, 2),
            time_per_position_ms=round(median / T, 4),
            peak_allocated_gb=bytes_to_gb(peak_allocated),
            peak_reserved_gb=bytes_to_gb(peak_reserved),
            status="success",
            memory_potentials_gb=breakdown["potentials_gb"],
            memory_dp_state_gb=breakdown["dp_state_gb"],
            memory_workspace_gb=breakdown["workspace_gb"],
            memory_autograd_gb=breakdown["autograd_gb"],
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="oom",
                error_msg=str(e)[:100],
                memory_potentials_gb=breakdown["potentials_gb"],
                memory_dp_state_gb=breakdown["dp_state_gb"],
                memory_workspace_gb=breakdown["workspace_gb"],
                memory_autograd_gb=breakdown["autograd_gb"],
            )
        else:
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="error",
                error_msg=str(e)[:100],
            )
    except NotImplementedError as e:
        return BenchmarkResult(
            **result_base,
            time_ms_median=float("nan"),
            time_ms_iqr_low=float("nan"),
            time_ms_iqr_high=float("nan"),
            time_per_position_ms=float("nan"),
            peak_allocated_gb=float("nan"),
            peak_reserved_gb=float("nan"),
            status="not_implemented",
            error_msg=str(e)[:100],
        )
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
