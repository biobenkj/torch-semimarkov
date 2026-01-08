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
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


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


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def bytes_to_gb(b: int) -> float:
    """Convert bytes to GB with 3 decimal places."""
    return round(b / (1024**3), 3)


def estimate_memory_breakdown(T: int, K: int, C: int, B: int, backend: str) -> Dict[str, float]:
    """
    Estimate memory breakdown by category (in GB).

    Categories:
    - potentials: Edge potential tensor (B, T-1, K, C, C)
    - dp_state: Forward/backward DP tables
    - workspace: Intermediate computation tensors
    - autograd: Saved tensors for backward pass
    """
    float_bytes = 4  # float32

    # Potentials: (B, T-1, K, C, C) - always present
    potentials_bytes = B * (T - 1) * K * C * C * float_bytes

    # DP state depends on backend
    if backend in ["linear_scan", "linear_scan_vectorized"]:
        # O(TC) state - just alpha table
        dp_state_bytes = B * T * C * float_bytes
        # Workspace for vectorized: O(KC²) per step
        if backend == "linear_scan_vectorized":
            workspace_bytes = B * K * C * C * float_bytes * 2  # temp buffers
        else:
            workspace_bytes = B * C * float_bytes  # minimal
        # Autograd saves potentials slice per step
        autograd_bytes = potentials_bytes  # roughly

    elif backend == "binary_tree":
        # O(T * KC * KC) for tree matrices
        KC = K * C
        dp_state_bytes = B * T * KC * float_bytes
        workspace_bytes = B * T * KC * KC * float_bytes  # the killer!
        autograd_bytes = workspace_bytes * 2  # saved for backward

    elif backend == "banded":
        # O(T * KC * BW) where BW is bandwidth
        KC = K * C
        bw = min(KC, K * 2)  # rough bandwidth estimate
        dp_state_bytes = B * T * KC * float_bytes
        workspace_bytes = B * T * KC * bw * float_bytes
        autograd_bytes = workspace_bytes

    elif backend == "block_triangular":
        # Similar to binary tree but with block structure
        KC = K * C
        dp_state_bytes = B * T * KC * float_bytes
        workspace_bytes = B * T * KC * KC * float_bytes
        autograd_bytes = workspace_bytes * 2

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
    T: int, K: int, C: int, backend: str,
    oom_history: Dict[str, List[Tuple[int, int, int]]],
    max_memory_gb: float = 40.0
) -> Tuple[bool, str]:
    """
    Determine if we should skip this config based on:
    1. Predicted memory > max_memory_gb
    2. Adjacent config already OOM'd

    Returns (should_skip, reason).
    """
    # Check if adjacent (smaller) config OOM'd for this backend
    key = backend
    if key in oom_history:
        for (oom_t, oom_k, oom_c) in oom_history[key]:
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
    T: int, K: int, C: int, B: int,
    backend: str,
    device: torch.device,
    repeats: int = 5,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    KC = K * C
    result_base = {
        "T": T, "K": K, "C": C, "B": B, "KC": KC, "backend": backend,
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

        for rep in range(repeats):
            edge_run = edge.clone().detach().requires_grad_(True)

            # Clear cache and reset stats
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            gc.collect()
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
                    edge_run, lengths=lengths,
                    use_linear_scan=True, use_vectorized=True,
                    use_banded=True, banded_perm="auto", banded_bw_ratio=0.6,
                )
            elif backend == "block_triangular":
                if hasattr(struct, "_dp_blocktriangular"):
                    v, _, _ = struct._dp_blocktriangular(edge_run, lengths, force_grad=True)
                else:
                    raise NotImplementedError("block_triangular not available")
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

            # Clean up
            del edge_run, v, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--T", type=str, default="128,256,512,1024")
    parser.add_argument("--K", type=str, default="4,8,12,16,20,24")
    parser.add_argument("--C", type=str, default="3,6,9,12")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backends", type=str,
        default="linear_scan,linear_scan_vectorized,binary_tree,banded,block_triangular"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-memory-gb", type=float, default=40.0,
                        help="Skip configs predicted to exceed this memory")
    parser.add_argument("--skip-adjacent-oom", action="store_true", default=True,
                        help="Skip configs if smaller adjacent config OOM'd")
    args = parser.parse_args()

    device = torch.device(args.device)
    T_list = parse_int_list(args.T)
    K_list = parse_int_list(args.K)
    C_list = parse_int_list(args.C)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    results: List[BenchmarkResult] = []
    oom_history: Dict[str, List[Tuple[int, int, int]]] = {b: [] for b in backends}

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
                            print(f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, KC={KC}, {backend}: {reason}")
                            results.append(BenchmarkResult(
                                T=T, K=K, C=C, B=args.B, KC=KC, backend=backend,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="not_tested",
                                error_msg=reason,
                            ))
                            continue

                    print(f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, {backend}...", end=" ", flush=True)

                    result = run_single_benchmark(
                        T, K, C, args.B, backend, device, args.repeats
                    )
                    results.append(result)

                    if result.status == "success":
                        print(f"OK: {result.time_ms_median:.1f}ms, {result.peak_allocated_gb:.3f}GB allocated, {result.peak_reserved_gb:.3f}GB reserved")
                    elif result.status == "oom":
                        print(f"OOM")
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
        heatmap_data[key]["cells"].append({
            "K": r.K, "C": r.C, "KC": r.KC,
            "status": r.status,
            "peak_gb": r.peak_allocated_gb if r.status == "success" else None,
            "time_ms": r.time_ms_median if r.status == "success" else None,
        })
    with open(heatmap_path, "w") as f:
        json.dump(heatmap_data, f, indent=2)
    print(f"Saved heatmap data to {heatmap_path}")

    # 3. Memory breakdown summary
    breakdown_path = args.output_dir / "memory_breakdown.csv"
    breakdown_fields = ["backend", "T", "K", "C", "KC", "status",
                        "peak_allocated_gb", "peak_reserved_gb",
                        "est_potentials_gb", "est_dp_state_gb",
                        "est_workspace_gb", "est_autograd_gb"]
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=breakdown_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "backend": r.backend, "T": r.T, "K": r.K, "C": r.C, "KC": r.KC,
                "status": r.status,
                "peak_allocated_gb": r.peak_allocated_gb,
                "peak_reserved_gb": r.peak_reserved_gb,
                "est_potentials_gb": r.memory_potentials_gb,
                "est_dp_state_gb": r.memory_dp_state_gb,
                "est_workspace_gb": r.memory_workspace_gb,
                "est_autograd_gb": r.memory_autograd_gb,
            })
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

        print(f"{backend:25s}: {success:3d} success, {oom:3d} OOM, {skipped:3d} skipped | max KC={max_kc:4d}, max mem={max_mem:.2f}GB")


if __name__ == "__main__":
    main()
