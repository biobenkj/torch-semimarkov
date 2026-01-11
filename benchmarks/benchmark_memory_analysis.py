#!/usr/bin/env python3
"""
Memory analysis benchmark for Semi-Markov CRF backends.

Generates data for:
1. OOM feasibility heatmaps (with consistent GB units)
2. Time vs state-space size plots (with median/IQR)
3. Memory breakdown by allocation category

Addresses reviewer feedback:
- Consistent units (GB with 2 sig figs)
- Colorblind-safe output (blue/orange scheme recommended)
- Memory breakdown: DP state, potentials, workspace, autograd
- Reports both max_memory_allocated and max_memory_reserved
- Time per position normalization

Example:
    python benchmarks/benchmark_memory_analysis.py \
        --device cuda:0 \
        --T 128,256,512,1024 \
        --K 4,8,12,16,20,24 \
        --C 3,6,9,12 \
        --B 4 \
        --repeats 5 \
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
from torch_semimarkov.semirings import LogSemiring
from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring


@dataclass
class BenchmarkResult:
    """Single benchmark result with full metrics."""

    T: int
    K: int
    C: int
    B: int
    KC: int  # state-space size
    backend: str

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
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    KC = K * C
    result_base = {
        "T": T,
        "K": K,
        "C": C,
        "B": B,
        "KC": KC,
        "backend": backend,
    }

    # Estimate memory breakdown
    breakdown = estimate_memory_breakdown(T, K, C, B, backend)

    try:
        struct = SemiMarkov(LogSemiring)

        # Create potentials
        edge = torch.randn(B, T - 1, K, C, C, device=device, requires_grad=False)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        times_ms = []
        peak_allocated = 0
        peak_reserved = 0

        # Warmup run (not recorded) to stabilize CUDA state
        if device.type == "cuda":
            edge_warmup = edge.clone().detach().requires_grad_(True)
            try:
                if backend == "binary_tree":
                    v_warm, _ = struct.logpartition(
                        edge_warmup, lengths=lengths, use_linear_scan=False
                    )
                elif backend == "linear_scan":
                    v_warm, _, _ = struct._dp_standard(edge_warmup, lengths, force_grad=True)
                elif backend == "linear_scan_vectorized":
                    v_warm, _, _ = struct._dp_standard_vectorized(
                        edge_warmup, lengths, force_grad=True
                    )
                elif backend == "banded":
                    v_warm, _, _ = struct.logpartition(
                        edge_warmup,
                        lengths=lengths,
                        use_linear_scan=True,
                        use_vectorized=True,
                        use_banded=True,
                        banded_perm="auto",
                        banded_bw_ratio=0.6,
                    )
                elif backend == "block_triangular":
                    if hasattr(struct, "_dp_blocktriangular"):
                        v_warm, _, _ = struct._dp_blocktriangular(
                            edge_warmup, lengths, force_grad=True
                        )
                elif backend == "binary_tree_sharded":
                    # Use CheckpointShardSemiring to reduce peak memory
                    ShardedLogSemiring = CheckpointShardSemiring(LogSemiring, max_size=10000)
                    struct_sharded = SemiMarkov(ShardedLogSemiring)
                    v_warm, _ = struct_sharded.logpartition(
                        edge_warmup, lengths=lengths, use_linear_scan=False
                    )
                elif backend == "linear_scan_streaming":
                    # True streaming scan with O(K*C) DP state
                    v_warm, _, _ = struct._dp_scan_streaming(edge_warmup, lengths, force_grad=True)
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

            edge_run = edge.clone().detach().requires_grad_(True)

            t0 = time.perf_counter()

            # Run the appropriate backend
            if backend == "binary_tree":
                v, _ = struct.logpartition(edge_run, lengths=lengths, use_linear_scan=False)
            elif backend == "linear_scan":
                v, _, _ = struct._dp_standard(edge_run, lengths, force_grad=True)
            elif backend == "linear_scan_vectorized":
                v, _, _ = struct._dp_standard_vectorized(edge_run, lengths, force_grad=True)
            elif backend == "banded":
                v, _, _ = struct.logpartition(
                    edge_run,
                    lengths=lengths,
                    use_linear_scan=True,
                    use_vectorized=True,
                    use_banded=True,
                    banded_perm="auto",
                    banded_bw_ratio=0.6,
                )
            elif backend == "block_triangular":
                if hasattr(struct, "_dp_blocktriangular"):
                    v, _, _ = struct._dp_blocktriangular(edge_run, lengths, force_grad=True)
                else:
                    raise NotImplementedError("block_triangular not available")
            elif backend == "binary_tree_sharded":
                # Use CheckpointShardSemiring to reduce peak memory at cost of time
                # This is a "fair" tree baseline that doesn't suffer from O((KC)^3) peak
                ShardedLogSemiring = CheckpointShardSemiring(LogSemiring, max_size=10000)
                struct_sharded = SemiMarkov(ShardedLogSemiring)
                v, _ = struct_sharded.logpartition(edge_run, lengths=lengths, use_linear_scan=False)
            elif backend == "linear_scan_streaming":
                # True streaming scan: O(K*C) DP state, matches paper's memory narrative
                v, _, _ = struct._dp_scan_streaming(edge_run, lengths, force_grad=True)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            # Backward pass
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
            del edge_run, v, loss
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
        help="Comma-separated list of backends. linear_scan_streaming has O(K*C) DP state. binary_tree_sharded uses CheckpointShardSemiring.",
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    results: list[BenchmarkResult] = []
    oom_history: dict[str, list[tuple[int, int, int]]] = {b: [] for b in backends}

    total_configs = len(T_list) * len(K_list) * len(C_list) * len(backends)
    completed = 0

    print(f"Running {total_configs} configurations...")
    print(f"Device: {device}")
    print(f"T: {T_list}, K: {K_list}, C: {C_list}, B: {args.B}")
    print(f"Backends: {backends}")
    print(f"Repeats: {args.repeats}")
    print("-" * 80)

    for T in T_list:
        for K in K_list:
            for C in C_list:
                KC = K * C
                for backend in backends:
                    completed += 1

                    # Check if we should skip
                    if args.skip_adjacent_oom:
                        skip, reason = should_skip_config(
                            T, K, C, backend, oom_history, args.max_memory_gb
                        )
                        if skip:
                            print(
                                f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, KC={KC}, {backend}: {reason}"
                            )
                            results.append(
                                BenchmarkResult(
                                    T=T,
                                    K=K,
                                    C=C,
                                    B=args.B,
                                    KC=KC,
                                    backend=backend,
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
                        f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, {backend}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_single_benchmark(T, K, C, args.B, backend, device, args.repeats)
                    results.append(result)

                    if result.status == "success":
                        print(
                            f"OK: {result.time_ms_median:.1f}ms, {result.peak_allocated_gb:.3f}GB allocated, {result.peak_reserved_gb:.3f}GB reserved"
                        )
                    elif result.status == "oom":
                        print("OOM")
                        oom_history[backend].append((T, K, C))
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
        key = f"{r.backend}_T{r.T}"
        if key not in heatmap_data:
            heatmap_data[key] = {"backend": r.backend, "T": r.T, "cells": []}
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
        backend_results = [r for r in results if r.backend == backend]
        success = sum(1 for r in backend_results if r.status == "success")
        oom = sum(1 for r in backend_results if r.status == "oom")
        skipped = sum(1 for r in backend_results if r.status == "not_tested")

        successful = [r for r in backend_results if r.status == "success"]
        if successful:
            max_kc = max(r.KC for r in successful)
            max_mem = max(r.peak_allocated_gb for r in successful)
        else:
            max_kc = 0
            max_mem = 0

        print(
            f"{backend:25s}: {success:3d} success, {oom:3d} OOM, {skipped:3d} skipped | max KC={max_kc:4d}, max mem={max_mem:.2f}GB"
        )


if __name__ == "__main__":
    main()
