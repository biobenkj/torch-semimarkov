#!/usr/bin/env python3
"""
Benchmark SemiMarkov CRF backends on GPU:

- binary_tree   : original O(log T) binary tree DP
- linear_scan   : O(T) _dp_standard
- linear_scan_vectorized : Vectorized O(T) DP (2-3x speedup)
- banded        : banded binary-tree / banded-matmul path
- block_triangular : block-triangular matmul backend

For each (T, K, C, B) configuration, runs forward+backward on random
log-potentials and records:

- time_ms  : average wall-clock per run (ms)
- peak_mb  : peak CUDA memory allocated (MB)

Example usage:

    python benchmarks/benchmark_backends.py \
      --device cuda:0 \
      --T 512,1024,2048,3072,4096 \
      --K 12,16,20 \
      --C 3,6 \
      --B 4 \
      --repeats 3 \
      --banded-bw-ratio 0.6 \
      --banded-perm auto \
      --backends binary_tree,linear_scan,linear_scan_vectorized,banded,block_triangular \
      --csv semimarkov_backends_gpu.csv
"""

import argparse
import csv
import time

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def parse_int_list(s: str):
    return [int(x) for x in s.split(",")] if s else []


def run_backend_once(
    struct,
    edge: torch.Tensor,
    lengths: torch.Tensor,
    backend: str,
    device: torch.device,
    banded_bw_ratio: float,
    banded_perm: str,
):
    """
    Run a single forward+backward pass for a given backend and measure
    wall time + peak CUDA memory.

    backend in {"binary_tree", "linear_scan", "linear_scan_vectorized", "banded", "block_triangular"}.
    """
    # Fresh tensor with grad each time to avoid stale gradients
    edge_run = edge.clone().detach().requires_grad_(True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()

    if backend == "binary_tree":
        # Original binary-tree DP (no linear scan)
        v, _ = struct.logpartition(
            edge_run,
            lengths=lengths,
            use_linear_scan=False,
        )

    elif backend == "linear_scan":
        # O(T) DP
        v, _, _ = struct._dp_standard(
            edge_run,
            lengths,
            force_grad=True,
        )

    elif backend == "linear_scan_vectorized":
        # Vectorized linear scan DP
        v, _, _ = struct._dp_standard_vectorized(
            edge_run,
            lengths,
            force_grad=True,
        )

    elif backend == "banded":
        # Banded path (gated by bandwidth + permutation)
        v, _, _ = struct.logpartition(
            edge_run,
            lengths=lengths,
            use_linear_scan=True,
            use_vectorized=True,
            use_banded=True,
            banded_perm=banded_perm,
            banded_bw_ratio=banded_bw_ratio,
        )

    elif backend == "block_triangular":
        # Block-triangular DP
        if hasattr(struct, "_dp_blocktriangular"):
            v, _, _ = struct._dp_blocktriangular(edge_run, lengths, force_grad=True)
        elif "use_block_triangular" in struct._dp_standard.__code__.co_varnames:
            v, _, _ = struct._dp_standard(
                edge_run,
                lengths,
                force_grad=True,
                use_block_triangular=True,
            )
        else:
            raise NotImplementedError(
                "block_triangular backend: no _dp_blocktriangular or "
                "use_block_triangular flag found on SemiMarkov."
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    loss = v.sum()
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = 0.0  # no easy CPU peak tracking here

    return wall_ms, peak_mb


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark for SemiMarkov backends")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0 or cpu)")
    parser.add_argument("--T", type=str, default="512", help="Comma-separated sequence lengths")
    parser.add_argument("--K", type=str, default="16", help="Comma-separated max durations")
    parser.add_argument("--C", type=str, default="4", help="Comma-separated num labels")
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per backend")
    parser.add_argument(
        "--banded-bw-ratio",
        type=float,
        default=0.6,
        help="Bandwidth ratio threshold for banded gating",
    )
    parser.add_argument(
        "--banded-perm",
        type=str,
        default="auto",
        choices=["auto", "none", "snake", "rcm"],
        help="Permutation strategy for banded path",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="binary_tree,linear_scan,linear_scan_vectorized,banded,block_triangular",
        help="Comma-separated list of backends to run",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file (if omitted, just prints to stdout)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    T_list = parse_int_list(args.T)
    K_list = parse_int_list(args.K)
    C_list = parse_int_list(args.C)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    torch.manual_seed(0)

    results = []

    for T in T_list:
        for K in K_list:
            for C in C_list:
                # Build random edge potentials on the target device
                B = args.B
                edge = torch.randn(B, T - 1, K, C, C, device=device, requires_grad=False)
                lengths = torch.full((B,), T, dtype=torch.long, device=device)

                struct = SemiMarkov(LogSemiring)

                for backend in backends:
                    # Handle backends that aren't wired yet gracefully
                    try:
                        times = []
                        peaks = []
                        for _ in range(args.repeats):
                            wall_ms, peak_mb = run_backend_once(
                                struct,
                                edge,
                                lengths,
                                backend,
                                device,
                                banded_bw_ratio=args.banded_bw_ratio,
                                banded_perm=args.banded_perm,
                            )
                            times.append(wall_ms)
                            peaks.append(peak_mb)

                        avg_time = sum(times) / len(times)
                        max_peak = max(peaks)

                        row = {
                            "T": T,
                            "K": K,
                            "C": C,
                            "B": B,
                            "backend": backend,
                            "time_ms": avg_time,
                            "peak_mb": max_peak,
                        }
                        results.append(row)

                        print(
                            f"T={T:5d}, K={K:3d}, C={C:3d}, B={B:2d}, backend={backend:24s} | "
                            f"time={avg_time:8.3f} ms, peak={max_peak:8.2f} MB"
                        )

                    except NotImplementedError as e:
                        print(f"Skipping backend={backend} for T={T},K={K},C={C}: {e}")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(
                                f"OOM for backend={backend}, T={T},K={K},C={C},B={B} "
                                f"on device={device}"
                            )
                            row = {
                                "T": T,
                                "K": K,
                                "C": C,
                                "B": B,
                                "backend": backend,
                                "time_ms": float("nan"),
                                "peak_mb": float("nan"),
                                "error": "OOM",
                            }
                            results.append(row)
                        else:
                            raise

    if args.csv:
        fieldnames = ["T", "K", "C", "B", "backend", "time_ms", "peak_mb", "error"]
        # Ensure 'error' exists for rows without errors
        for row in results:
            if "error" not in row:
                row["error"] = ""

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    main()
