#!/usr/bin/env python3
"""
Memory analysis benchmark for Semi-Markov CRF backends.

Generates data for:
1. OOM feasibility heatmaps (with consistent GB units)
2. Time vs state-space size plots (with median/IQR)
3. Memory breakdown by allocation category

Supports:
- Multiple backends including Triton-accelerated scan
- Forward-only, backward-only, or combined timing
- Different semirings (Log, Max, Entropy, etc.)

Example:
    python benchmarks/benchmark_memory_analysis.py \
        --device cuda:0 \
        --T 128,256,512,1024 \
        --K 4,8,12,16,20,24 \
        --C 3,6,9,12 \
        --B 4 \
        --repeats 5 \
        --phases both \
        --semirings Log,Max \
        --output-dir results/
"""

import argparse
import csv
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import EntropySemiring, LogSemiring, MaxSemiring
from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring
from torch_semimarkov.triton_scan import HAS_TRITON, semi_crf_triton_forward


# Reset torch.compile caches to avoid state issues between configurations
def reset_compile_caches():
    """Reset torch.compile and triton caches to avoid inter-configuration issues."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    # Also reset the module-level compiled function caches
    try:
        import torch_semimarkov.triton_scan as ts

        ts._compiled_forward_log = None
        ts._compiled_forward_max = None
    except Exception:
        pass


# Mapping of semiring names to classes
SEMIRING_MAP = {
    "Log": LogSemiring,
    "Max": MaxSemiring,
    "Entropy": EntropySemiring,
}

# Backends that only work with LogSemiring (due to hardcoded logsumexp)
LOG_SEMIRING_ONLY_BACKENDS: set[str] = set()

# Backends that only support a subset of semirings
# triton/triton_pytorch support Log and Max, but not Entropy
TRITON_SUPPORTED_SEMIRINGS = {"Log", "Max"}


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


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def bytes_to_gb(b: int) -> float:
    """Convert bytes to GB with 3 decimal places."""
    return round(b / (1024**3), 3)


def estimate_memory_breakdown(T: int, K: int, C: int, B: int, backend: str) -> dict[str, float]:
    """
    Estimate memory breakdown by category (in GB).

    Categories:
    - potentials: Edge potential tensor (B, T-1, K, C, C)
    - dp_state: Forward/backward DP tables
    - workspace: Intermediate computation tensors
    - autograd: Saved tensors for backward pass

    NOTE: These are estimates based on actual implementation analysis.
    For precise measurements, use torch.cuda.memory_snapshot().

    Actual implementation details (as of code review):
    - linear_scan stores alpha: (ssize, B, N, K, C) -> O(T*K*C)
    - linear_scan stores beta: list of N tensors (ssize, B, C) -> O(T*C)
    - binary_tree log-semiring matmul materializes O((KC)^3) temporaries
    """
    float_bytes = 4  # float32
    N = T - 1  # number of positions

    # Potentials: (B, T-1, K, C, C) - always present
    potentials_bytes = B * N * K * C * C * float_bytes

    # DP state depends on backend
    if backend in ["linear_scan", "linear_scan_vectorized"]:
        # ACTUAL implementation (not theoretical O(TC)):
        # alpha: (ssize, B, N, K, C) -> O(T*K*C) resident
        # beta: list of N tensors of shape (ssize, B, C) -> O(T*C)
        # Total DP state is O(T*K*C + T*C) = O(T*K*C) when K > 1
        alpha_bytes = 1 * B * N * K * C * float_bytes  # ssize=1 for LogSemiring
        beta_bytes = 1 * B * N * C * float_bytes
        dp_state_bytes = alpha_bytes + beta_bytes

        # Workspace for vectorized: index tensors + gathered tensors
        if backend == "linear_scan_vectorized":
            # time_indices, dur_indices: (K,) each
            # gathered: (ssize, B, K, C) per step
            workspace_bytes = B * K * C * float_bytes * 2  # gathered + temp
        else:
            workspace_bytes = B * C * float_bytes  # minimal

        # Autograd: saves computation graph, roughly proportional to potentials
        autograd_bytes = potentials_bytes

    elif backend == "linear_scan_streaming":
        # TRUE streaming scan: O(K*C) DP state, independent of T
        # beta_hist: (ssize, B, K, C) ring buffer of last K betas
        # final_beta: (ssize, B, C)
        # NO alpha storage, NO full beta history
        dp_state_bytes = 1 * B * K * C * float_bytes + 1 * B * C * float_bytes
        # Workspace: edge_slice (B, k_eff, C, C), scores (B, k_eff, C) per step
        workspace_bytes = B * K * C * C * float_bytes + B * K * C * float_bytes
        # Autograd: saves edge (potentials) + ring buffer states
        autograd_bytes = potentials_bytes + dp_state_bytes * N  # graph grows with T

    elif backend == "binary_tree":
        # Tree stores chart matrices of shape (ssize, B, KC, KC) at each level
        # CRITICAL: log-semiring matmul materializes O((KC)^3) temporary
        # because it computes: result[i,k] = logsumexp_j(A[i,j] + B[j,k])
        # via broadcast: (KC, KC, 1) + (1, KC, KC) -> (KC, KC, KC)
        KC = K * C
        # Chart storage across ~log(T) levels, but dominant cost is at base
        dp_state_bytes = B * N * KC * float_bytes  # vector at each position
        # Workspace: the O((KC)^3) temporary per matmul is the killer
        # At each level, we do ~T/2^level matmuls; base level is worst
        workspace_bytes = B * KC * KC * KC * float_bytes  # (KC)^3 temporary!
        # Autograd saves inputs for backward through all matmuls
        autograd_bytes = B * N * KC * KC * float_bytes * 2

    elif backend == "binary_tree_sharded":
        # Same algorithm as binary_tree but using CheckpointShardSemiring
        # which splits the O((KC)^3) matmul into smaller shards
        # This reduces peak memory at the cost of more serial computation
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        # Workspace is reduced because we shard the matmul
        # Instead of (KC)^3 all at once, we do it in chunks
        shard_size = 10000  # default shard size from checkpoint.py
        workspace_bytes = B * min(KC * KC * KC, shard_size * KC) * float_bytes
        # Autograd still needs to save inputs but recomputes forward in backward
        autograd_bytes = B * N * KC * KC * float_bytes  # ~half of non-sharded

    elif backend == "banded":
        # O(T * KC * BW) where BW is bandwidth
        KC = K * C
        bw = min(KC, K * 2)  # rough bandwidth estimate
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * bw * float_bytes  # per-multiply workspace
        autograd_bytes = B * N * KC * bw * float_bytes

    elif backend == "block_triangular":
        # Similar to binary tree but with block sparsity
        # Current impl converts to/from dense at each level (suboptimal)
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * KC * float_bytes  # dense conversion overhead
        autograd_bytes = B * N * KC * KC * float_bytes

    elif backend in ("triton", "triton_pytorch", "triton_checkpointing"):
        # Triton scan uses O(K*C) ring buffer, same as streaming scan
        # triton: Fused GPU kernel with ring buffer in L1/L2
        # triton_pytorch: Reference PyTorch implementation (same algorithm)
        # Ring buffer: (B, K, C_PAD) where C_PAD = next_power_of_2(C)
        C_PAD = 1
        while C_PAD < C:
            C_PAD *= 2
        ring_buffer_bytes = B * K * C_PAD * float_bytes
        # Output partition: (B,)
        output_bytes = B * float_bytes
        dp_state_bytes = ring_buffer_bytes + output_bytes
        # Workspace: minimal for triton (fused kernel)
        workspace_bytes = B * C * float_bytes  # final_beta temporary
        # Autograd: Triton uses gradient checkpointing (recomputes forward in backward)
        # So autograd memory is primarily the saved edge tensor
        autograd_bytes = potentials_bytes

    else:
        dp_state_bytes = 0
        workspace_bytes = 0
        autograd_bytes = 0

    return {
        "potentials_gb": bytes_to_gb(potentials_bytes),
        "dp_state_gb": bytes_to_gb(dp_state_bytes),
        "workspace_gb": bytes_to_gb(workspace_bytes),
        "autograd_gb": bytes_to_gb(autograd_bytes),
        "total_estimated_gb": bytes_to_gb(
            potentials_bytes + dp_state_bytes + workspace_bytes + autograd_bytes
        ),
    }


def should_skip_config(
    T: int,
    K: int,
    C: int,
    backend: str,
    oom_history: dict[str, list[tuple[int, int, int]]],
    max_memory_gb: float = 40.0,
) -> tuple[bool, str]:
    """
    Determine if we should skip this config based on:
    1. Predicted memory > max_memory_gb
    2. Adjacent config already OOM'd

    Returns (should_skip, reason).
    """
    # Check if adjacent (smaller) config OOM'd for this backend
    key = backend
    if key in oom_history:
        for oom_t, oom_k, oom_c in oom_history[key]:
            # If a smaller config OOM'd, skip larger ones
            if T >= oom_t and K >= oom_k and C >= oom_c:
                if T > oom_t or K > oom_k or C > oom_c:
                    return True, f"adjacent OOM at T={oom_t},K={oom_k},C={oom_c}"

    # Estimate memory and skip if too large
    breakdown = estimate_memory_breakdown(T, K, C, 4, backend)  # assume B=4
    if breakdown["total_estimated_gb"] > max_memory_gb:
        return True, f"predicted {breakdown['total_estimated_gb']:.1f}GB > {max_memory_gb}GB limit"

    return False, ""


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
    use_compile: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        T: Sequence length
        K: Maximum duration
        C: Number of labels
        B: Batch size
        backend: Backend name (linear_scan, triton, etc.)
        device: Torch device
        repeats: Number of repetitions for timing
        semiring_name: Semiring to use ("Log", "Max", "Entropy")
        phase: "forward" (forward only), "backward" (backward only), or "both"
        use_compile: If True, use torch.compile for triton training backward pass.
            If False, use gradient checkpointing. Default: True
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

    # Get semiring class
    semiring_cls = SEMIRING_MAP.get(semiring_name, LogSemiring)

    try:
        # Create struct with appropriate semiring
        struct = SemiMarkov(semiring_cls)

        # Create potentials
        edge = torch.randn(B, T - 1, K, C, C, device=device, requires_grad=False)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        times_ms = []
        peak_allocated = 0
        peak_reserved = 0

        # Map semiring name to triton semiring string
        triton_semiring = semiring_name.lower()  # "Log" -> "log", "Max" -> "max"

        def run_backend_forward(edge_input, struct_to_use, use_compile=True):
            """Run forward pass for the specified backend, returning partition value.

            For triton backends, the execution path depends on edge_input.requires_grad:
            - requires_grad=False (inference): Custom Triton kernel (~45x speedup)
            - requires_grad=True + use_compile=True: torch.compile for efficient backward
            - requires_grad=True + use_compile=False: Gradient checkpointing (recomputes forward)
            """
            if backend == "triton":
                # Triton-accelerated scan (supports Log and Max semirings)
                # Hybrid routing: custom kernel for inference, torch.compile for training
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=True,
                    semiring=triton_semiring,
                    use_compile=use_compile,
                )
            elif backend == "triton_pytorch":
                # PyTorch reference for Triton (supports Log and Max semirings)
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=False,
                    semiring=triton_semiring,
                    use_compile=use_compile,
                )
            elif backend == "triton_checkpointing":
                # Triton with gradient checkpointing (old approach, for comparison)
                # Always uses use_compile=False to force checkpointing path
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=True,
                    semiring=triton_semiring,
                    use_compile=False,
                )
            elif backend == "binary_tree":
                v, _ = struct_to_use.logpartition(
                    edge_input, lengths=lengths, use_linear_scan=False
                )
                return v
            elif backend == "linear_scan":
                v, _, _ = struct_to_use._dp_standard(edge_input, lengths, force_grad=True)
                return v
            elif backend == "linear_scan_vectorized":
                v, _, _ = struct_to_use._dp_standard_vectorized(
                    edge_input, lengths, force_grad=True
                )
                return v
            elif backend == "banded":
                v, _, _ = struct_to_use.logpartition(
                    edge_input,
                    lengths=lengths,
                    use_linear_scan=True,
                    use_vectorized=True,
                    use_banded=True,
                    banded_perm="auto",
                    banded_bw_ratio=0.6,
                )
                return v
            elif backend == "block_triangular":
                if hasattr(struct_to_use, "_dp_blocktriangular"):
                    v, _, _ = struct_to_use._dp_blocktriangular(
                        edge_input, lengths, force_grad=True
                    )
                    return v
                else:
                    raise NotImplementedError("block_triangular not available")
            elif backend == "binary_tree_sharded":
                # Use CheckpointShardSemiring to reduce peak memory
                ShardedSemiring = CheckpointShardSemiring(semiring_cls, max_size=10000)
                struct_sharded = SemiMarkov(ShardedSemiring)
                v, _ = struct_sharded.logpartition(
                    edge_input, lengths=lengths, use_linear_scan=False
                )
                return v
            elif backend == "linear_scan_streaming":
                v, _, _ = struct_to_use._dp_scan_streaming(edge_input, lengths, force_grad=True)
                return v
            else:
                raise ValueError(f"Unknown backend: {backend}")

        # Warmup run (not recorded) to stabilize CUDA state
        # IMPORTANT: Run the exact same computation as the timed run for accurate warmup
        # For triton with torch.compile, this triggers the JIT compilation
        if device.type == "cuda":
            edge_warmup = edge.clone().detach().requires_grad_(True)
            try:
                v_warm = run_backend_forward(edge_warmup, struct, use_compile=use_compile)
                # Warmup backward pass too if we're timing backward
                if phase in ("backward", "both"):
                    v_warm.sum().backward()
                del edge_warmup, v_warm
            except Exception:
                pass  # If warmup fails, actual run will catch it
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()

        for _rep in range(repeats):
            # IMPORTANT: Reset memory stats BEFORE allocating edge_run
            # so that edge_run allocation is included in peak measurement
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            gc.collect()

            # Only need gradients if running backward pass
            needs_grad = phase in ("backward", "both")
            edge_run = edge.clone().detach().requires_grad_(needs_grad)

            # Phase-aware timing
            if phase == "forward":
                # Forward only - no backward pass
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                loss = None
            elif phase == "backward":
                # Run forward (untimed), then time backward only
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            else:  # phase == "both"
                # Time forward + backward together (original behavior)
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times_ms.append(elapsed_ms)

            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))

            # Clean up (no empty_cache here - only between configs, not repeats)
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--T", type=str, default="128,256,512,1024")
    parser.add_argument("--K", type=str, default="4,8,12,16,20,24")
    parser.add_argument("--C", type=str, default="3,6,9,12")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backends",
        type=str,
        default="linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,binary_tree_sharded,block_triangular",
        help=(
            "Comma-separated list of backends. Options: "
            "linear_scan, linear_scan_vectorized, linear_scan_streaming, "
            "binary_tree, binary_tree_sharded, block_triangular, "
            "triton (GPU Triton kernel with torch.compile for training), "
            "triton_pytorch (PyTorch reference), "
            "triton_checkpointing (Triton with gradient checkpointing, for comparison). "
            "Note: triton backends support Log and Max semirings."
        ),
    )
    parser.add_argument(
        "--semirings",
        type=str,
        default="Log",
        help=(
            "Comma-separated list of semirings. Options: Log, Max, Entropy. "
            "Note: triton/triton_pytorch backends support Log and Max semirings."
        ),
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="both",
        help=(
            "Comma-separated list of phases to time. Options: "
            "forward (forward pass only), backward (backward pass only), "
            "both (forward + backward together). Default: both"
        ),
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=True,
        help=(
            "Use torch.compile for triton training backward pass (default). "
            "This generates optimized kernels for both forward and backward. "
            "Disable with --no-use-compile to use gradient checkpointing instead."
        ),
    )
    parser.add_argument(
        "--no-use-compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile for triton (use gradient checkpointing instead).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Skip configs predicted to exceed this memory",
    )
    parser.add_argument(
        "--skip-adjacent-oom",
        action="store_true",
        default=True,
        help="Skip configs if smaller adjacent config OOM'd",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    T_list = parse_int_list(args.T)
    K_list = parse_int_list(args.K)
    C_list = parse_int_list(args.C)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    semirings = [s.strip() for s in args.semirings.split(",") if s.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Validate semirings
    for s in semirings:
        if s not in SEMIRING_MAP:
            print(f"WARNING: Unknown semiring '{s}', available: {list(SEMIRING_MAP.keys())}")

    # Validate phases
    valid_phases = {"forward", "backward", "both"}
    for p in phases:
        if p not in valid_phases:
            print(f"WARNING: Unknown phase '{p}', available: {valid_phases}")

    # Check Triton availability if needed
    if any(b in ("triton", "triton_pytorch", "triton_checkpointing") for b in backends):
        if not HAS_TRITON:
            print("WARNING: Triton not available, triton backends will use PyTorch fallback")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    results: list[BenchmarkResult] = []
    # OOM history is tracked per (backend, semiring, phase) combination
    oom_history: dict[str, list[tuple[int, int, int]]] = {}
    for b in backends:
        for s in semirings:
            for p in phases:
                oom_history[f"{b}_{s}_{p}"] = []

    total_configs = (
        len(T_list) * len(K_list) * len(C_list) * len(backends) * len(semirings) * len(phases)
    )
    completed = 0

    print(f"Running {total_configs} configurations...")
    print(f"Device: {device}")
    print(f"T: {T_list}, K: {K_list}, C: {C_list}, B: {args.B}")
    print(f"Backends: {backends}")
    print(f"Semirings: {semirings}")
    print(f"Phases: {phases}")
    print(f"Repeats: {args.repeats}")
    print(f"Triton use_compile: {args.use_compile}")
    print("-" * 80)

    for backend in backends:
        for semiring_name in semirings:
            # Reset caches once per backend/semiring pair to avoid state issues
            if args.use_compile and backend in ("triton", "triton_pytorch", "triton_checkpointing"):
                reset_compile_caches()
            for phase in phases:
                for T in T_list:
                    for K in K_list:
                        for C in C_list:
                            KC = K * C
                            completed += 1
                            oom_key = f"{backend}_{semiring_name}_{phase}"

                            # Check backend/semiring compatibility
                            if backend in LOG_SEMIRING_ONLY_BACKENDS and semiring_name != "Log":
                                print(
                                    f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: "
                                    f"{backend} only supports Log semiring"
                                )
                                results.append(
                                    BenchmarkResult(
                                        T=T,
                                        K=K,
                                        C=C,
                                        B=args.B,
                                        KC=KC,
                                        backend=backend,
                                        semiring=semiring_name,
                                        phase=phase,
                                        time_ms_median=float("nan"),
                                        time_ms_iqr_low=float("nan"),
                                        time_ms_iqr_high=float("nan"),
                                        time_per_position_ms=float("nan"),
                                        peak_allocated_gb=float("nan"),
                                        peak_reserved_gb=float("nan"),
                                        status="not_supported",
                                        error_msg=f"{backend} only supports Log semiring",
                                    )
                                )
                                continue

                            # Check triton backend semiring compatibility (Log, Max only)
                            if (
                                backend in ("triton", "triton_pytorch", "triton_checkpointing")
                                and semiring_name not in TRITON_SUPPORTED_SEMIRINGS
                            ):
                                print(
                                    f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: "
                                    f"{backend} only supports Log/Max semirings"
                                )
                                results.append(
                                    BenchmarkResult(
                                        T=T,
                                        K=K,
                                        C=C,
                                        B=args.B,
                                        KC=KC,
                                        backend=backend,
                                        semiring=semiring_name,
                                        phase=phase,
                                        time_ms_median=float("nan"),
                                        time_ms_iqr_low=float("nan"),
                                        time_ms_iqr_high=float("nan"),
                                        time_per_position_ms=float("nan"),
                                        peak_allocated_gb=float("nan"),
                                        peak_reserved_gb=float("nan"),
                                        status="not_supported",
                                        error_msg=f"{backend} only supports Log/Max semirings",
                                    )
                                )
                                continue

                            # Check if we should skip based on OOM history
                            if args.skip_adjacent_oom:
                                # Use backend-only key for memory estimation (semiring doesn't affect memory much)
                                skip, reason = should_skip_config(
                                    T,
                                    K,
                                    C,
                                    backend,
                                    {backend: oom_history.get(oom_key, [])},
                                    args.max_memory_gb,
                                )
                                if skip:
                                    print(
                                        f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: {reason}"
                                    )
                                    results.append(
                                        BenchmarkResult(
                                            T=T,
                                            K=K,
                                            C=C,
                                            B=args.B,
                                            KC=KC,
                                            backend=backend,
                                            semiring=semiring_name,
                                            phase=phase,
                                            time_ms_median=float("nan"),
                                            time_ms_iqr_low=float("nan"),
                                            time_ms_iqr_high=float("nan"),
                                            time_per_position_ms=float("nan"),
                                            peak_allocated_gb=float("nan"),
                                            peak_reserved_gb=float("nan"),
                                            status="not_tested",
                                            error_msg=reason,
                                        )
                                    )
                                    continue

                            print(
                                f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, {backend}/{semiring_name}/{phase}...",
                                end=" ",
                                flush=True,
                            )

                            result = run_single_benchmark(
                                T,
                                K,
                                C,
                                args.B,
                                backend,
                                device,
                                args.repeats,
                                semiring_name=semiring_name,
                                phase=phase,
                                use_compile=args.use_compile,
                            )
                            results.append(result)

                            if result.status == "success":
                                print(
                                    f"OK: {result.time_ms_median:.1f}ms, {result.peak_allocated_gb:.3f}GB allocated, {result.peak_reserved_gb:.3f}GB reserved"
                                )
                            elif result.status == "oom":
                                print("OOM")
                                oom_history[oom_key].append((T, K, C))
                            else:
                                print(f"{result.status}: {result.error_msg}")

    # Save results
    # 1. Full CSV with all metrics
    csv_path = args.output_dir / "benchmark_full.csv"
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nSaved full results to {csv_path}")

    # 2. Heatmap data (for OOM feasibility figures)
    heatmap_path = args.output_dir / "heatmap_data.json"
    heatmap_data = {}
    for r in results:
        key = f"{r.backend}_{r.semiring}_{r.phase}_T{r.T}"
        if key not in heatmap_data:
            heatmap_data[key] = {
                "backend": r.backend,
                "semiring": r.semiring,
                "phase": r.phase,
                "T": r.T,
                "cells": [],
            }
        heatmap_data[key]["cells"].append(
            {
                "K": r.K,
                "C": r.C,
                "KC": r.KC,
                "status": r.status,
                "peak_gb": r.peak_allocated_gb if r.status == "success" else None,
                "time_ms": r.time_ms_median if r.status == "success" else None,
            }
        )
    with open(heatmap_path, "w") as f:
        json.dump(heatmap_data, f, indent=2)
    print(f"Saved heatmap data to {heatmap_path}")

    # 3. Memory breakdown summary
    breakdown_path = args.output_dir / "memory_breakdown.csv"
    breakdown_fields = [
        "backend",
        "semiring",
        "phase",
        "T",
        "K",
        "C",
        "KC",
        "status",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "est_potentials_gb",
        "est_dp_state_gb",
        "est_workspace_gb",
        "est_autograd_gb",
    ]
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=breakdown_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "backend": r.backend,
                    "semiring": r.semiring,
                    "phase": r.phase,
                    "T": r.T,
                    "K": r.K,
                    "C": r.C,
                    "KC": r.KC,
                    "status": r.status,
                    "peak_allocated_gb": r.peak_allocated_gb,
                    "peak_reserved_gb": r.peak_reserved_gb,
                    "est_potentials_gb": r.memory_potentials_gb,
                    "est_dp_state_gb": r.memory_dp_state_gb,
                    "est_workspace_gb": r.memory_workspace_gb,
                    "est_autograd_gb": r.memory_autograd_gb,
                }
            )
    print(f"Saved memory breakdown to {breakdown_path}")

    # 4. Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for backend in backends:
        for semiring_name in semirings:
            for phase in phases:
                key_results = [
                    r
                    for r in results
                    if r.backend == backend and r.semiring == semiring_name and r.phase == phase
                ]
                if not key_results:
                    continue

                success = sum(1 for r in key_results if r.status == "success")
                oom = sum(1 for r in key_results if r.status == "oom")
                skipped = sum(1 for r in key_results if r.status in ("not_tested", "not_supported"))

                successful = [r for r in key_results if r.status == "success"]
                if successful:
                    max_kc = max(r.KC for r in successful)
                    max_mem = max(r.peak_allocated_gb for r in successful)
                else:
                    max_kc = 0
                    max_mem = 0

                label = f"{backend}/{semiring_name}/{phase}"
                print(
                    f"{label:40s}: {success:3d} success, {oom:3d} OOM, {skipped:3d} skipped | max KC={max_kc:4d}, max mem={max_mem:.2f}GB"
                )


if __name__ == "__main__":
    main()
