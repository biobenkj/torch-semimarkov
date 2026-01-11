#!/usr/bin/env python3
"""
Grid benchmark for SemiMarkov: dense (linear scan) vs banded (with permutation/gating).

CPU-focused, intended for algorithmic exploration. Logs per-config averages:
- forward+backward wall time (ms)
- peak Python memory (MB; tracemalloc)
- banded gating summary (per tree level spans)

Example:
    python benchmarks/benchmark_grid.py \
        --T 512,1024 \
        --K 12,16 \
        --C 3,6 \
        --B 4 \
        --repeats 3 \
        --bw-ratio 0.6 \
        --banded-perm auto \
        --csv banded_grid.csv
"""

import argparse
import csv
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def parse_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def measure_time_and_peak(
    struct: SemiMarkov,
    edge: torch.Tensor,
    lengths: torch.Tensor,
    use_banded: bool,
    banded_perm: str,
    bw_ratio: float,
    repeats: int,
    device: torch.device,
) -> Dict[str, float]:
    times = []
    peaks = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        else:
            tracemalloc.start()
        t0 = time.time()
        v, _, _ = struct.logpartition(
            edge,
            lengths=lengths,
            use_linear_scan=True,
            use_vectorized=True,
            use_banded=use_banded,
            banded_perm=banded_perm,
            banded_bw_ratio=bw_ratio,
        )
        loss = v.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall = (time.time() - t0) * 1000.0  # ms
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1e6  # MB
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = peak_bytes / 1e6  # MB
        times.append(wall)
        peaks.append(peak)
        # Reset grad for next repeat
        edge.grad = None
    return {
        "time_ms": sum(times) / len(times),
        "peak_mb": max(peaks),
    }


def span_lengths(T: int) -> Iterable[int]:
    """
    Binary tree spans for sequence length T:
    level 1 -> span 4, level 2 -> span 8, etc. (matches _dp_banded loop).
    """
    struct = SemiMarkov(LogSemiring)
    log_N, _ = struct._bin_length(T - 1)
    for level in range(1, log_N + 1):
        yield 2 ** (level + 1)


def gating_summary(
    struct: SemiMarkov, T: int, K: int, C: int, perm: str, bw_ratio: float, device
) -> Dict[str, object]:
    spans = list(span_lengths(T))
    size = (K - 1) * C
    flags = []
    rel_bw = []
    for s in spans:
        use_banded, _, bw_best, threshold = struct._choose_banded_permutation(
            span_length=s,
            K=K,
            C=C,
            perm_mode=perm,
            bw_ratio=bw_ratio,
            device=device,
        )
        flags.append(use_banded)
        rel_bw.append(bw_best / size if size > 0 else 1.0)
    return {
        "num_levels": len(spans),
        "num_banded": sum(flags),
        "spans": spans,
        "flags": flags,
        "rel_bw": rel_bw,
        "min_rel_bw": min(rel_bw) if rel_bw else 1.0,
        "max_rel_bw": max(rel_bw) if rel_bw else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--T", type=str, default="512,1024,2048,3072,4096", help="Comma-separated sequence lengths."
    )
    parser.add_argument(
        "--K", type=str, default="8,12,16,20", help="Comma-separated max durations."
    )
    parser.add_argument("--C", type=str, default="3,6", help="Comma-separated label counts.")
    parser.add_argument("--B", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Repeats per config/mode for timing."
    )
    parser.add_argument(
        "--bw-ratio", type=float, default=0.6, help="Bandwidth ratio threshold for banded gating."
    )
    parser.add_argument(
        "--banded-perm",
        type=str,
        default="auto",
        choices=["auto", "none", "snake", "rcm"],
        help="Permutation strategy for banded path.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device to use (e.g., cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    struct = SemiMarkov(LogSemiring)

    rows = []
    for T in parse_list(args.T):
        for K in parse_list(args.K):
            for C in parse_list(args.C):
                # Build potentials once per config
                edge = torch.randn(args.B, T - 1, K, C, C, device=device, requires_grad=True)
                lengths = torch.full((args.B,), T, dtype=torch.long, device=device)

                # Dense baseline (linear scan)
                dense_metrics = measure_time_and_peak(
                    struct,
                    edge.clone().detach().requires_grad_(True),
                    lengths,
                    use_banded=False,
                    banded_perm="none",
                    bw_ratio=args.bw_ratio,
                    repeats=args.repeats,
                    device=device,
                )

                # Banded with gating/permutation
                banded_metrics = measure_time_and_peak(
                    struct,
                    edge.clone().detach().requires_grad_(True),
                    lengths,
                    use_banded=True,
                    banded_perm=args.banded_perm,
                    bw_ratio=args.bw_ratio,
                    repeats=args.repeats,
                    device=device,
                )

                gs = gating_summary(struct, T, K, C, args.banded_perm, args.bw_ratio, device)

                rows.extend(
                    [
                        {
                            "T": T,
                            "K": K,
                            "C": C,
                            "B": args.B,
                            "mode": "dense",
                            "time_ms": dense_metrics["time_ms"],
                            "peak_mb": dense_metrics["peak_mb"],
                            "num_levels": gs["num_levels"],
                            "num_banded": 0,
                            "spans": gs["spans"],
                            "flags": [False] * gs["num_levels"],
                            "rel_bw": gs["rel_bw"],
                            "min_rel_bw": gs["min_rel_bw"],
                            "max_rel_bw": gs["max_rel_bw"],
                        },
                        {
                            "T": T,
                            "K": K,
                            "C": C,
                            "B": args.B,
                            "mode": "banded",
                            "time_ms": banded_metrics["time_ms"],
                            "peak_mb": banded_metrics["peak_mb"],
                            "num_levels": gs["num_levels"],
                            "num_banded": gs["num_banded"],
                            "spans": gs["spans"],
                            "flags": gs["flags"],
                            "rel_bw": gs["rel_bw"],
                            "min_rel_bw": gs["min_rel_bw"],
                            "max_rel_bw": gs["max_rel_bw"],
                        },
                    ]
                )

                print(
                    f"T={T}, K={K}, C={C}, B={args.B} | "
                    f"dense time={dense_metrics['time_ms']:.1f}ms, peak={dense_metrics['peak_mb']:.2f}MB; "
                    f"banded time={banded_metrics['time_ms']:.1f}ms, peak={banded_metrics['peak_mb']:.2f}MB; "
                    f"levels banded={gs['num_banded']}/{gs['num_levels']}"
                )

    # CSV output
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            fieldnames = [
                "T",
                "K",
                "C",
                "B",
                "mode",
                "time_ms",
                "peak_mb",
                "num_levels",
                "num_banded",
                "spans",
                "flags",
                "rel_bw",
                "min_rel_bw",
                "max_rel_bw",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nWrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
