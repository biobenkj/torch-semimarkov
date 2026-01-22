#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PCTS = ["p50", "p75", "p90", "p95", "p99"]


def read_summary(summary_csv: Path) -> dict[str, dict[str, float]]:
    """
    Returns:
      metrics[metric_name][col_name] = value
    Example:
      metrics["gene_length_bp"]["p95"] = 134465.0
    """
    metrics: dict[str, dict[str, float]] = {}
    with open(summary_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            metric = row["metric"]
            metrics[metric] = {}
            for k in ["n", "min", "max", "mean"] + PCTS:
                if k in row and row[k] != "":
                    metrics[metric][k] = float(row[k])
    return metrics


def nice_round(x: float) -> int:
    """
    Round to a 'nice' number for benchmarks.
    Heuristic:
      - < 1,000: round to nearest 10
      - < 10,000: round to nearest 50
      - < 100,000: round to nearest 100
      - < 1,000,000: round to nearest 1,000
      - else: round to nearest 10,000
    """
    x = float(x)
    if x < 1_000:
        base = 10
    elif x < 10_000:
        base = 50
    elif x < 100_000:
        base = 100
    elif x < 1_000_000:
        base = 1_000
    else:
        base = 10_000
    return int(base * round(x / base))


def choose_values(metric: dict[str, float], pcts: list[str], do_round: bool) -> list[int]:
    vals = [metric[p] for p in pcts if p in metric and not math.isnan(metric[p])]
    out = [nice_round(v) if do_round else int(round(v)) for v in vals]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def markdown_table(T_vals: list[int], K_vals: list[int], C: int, mode: str) -> str:
    lines = []
    lines.append(f"### Benchmark suite ({mode})")
    lines.append("")
    lines.append("| Config | T (bp) | K (bp) | C |")
    lines.append("|---|---:|---:|---:|")
    for i, (T, K) in enumerate(zip(T_vals, K_vals, strict=True), start=1):
        lines.append(f"| {mode}{i} | {T:,} | {K:,} | {C} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv", type=Path, required=True, help="Path to gtf_stats_out/.../summary.csv"
    )
    ap.add_argument("--C", type=int, default=6, help="State count C to use in configs")
    ap.add_argument("--round", action="store_true", help="Round values to nice benchmark numbers")
    ap.add_argument("--T-pcts", default="p50,p90,p95,p99", help="Comma-separated percentiles for T")
    ap.add_argument("--K-pcts", default="p75,p90,p95,p99", help="Comma-separated percentiles for K")
    ap.add_argument(
        "--K-source",
        choices=["exon", "intron"],
        default="exon",
        help="Use exon_length_bp or intron_length_bp percentiles for K",
    )
    ap.add_argument(
        "--backend-cmd",
        type=str,
        default="benchmark_memory_analysis.py",
        help="Benchmark script name",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="If set, write outputs into this directory"
    )
    args = ap.parse_args()

    metrics = read_summary(args.summary_csv)

    if "gene_length_bp" not in metrics:
        raise SystemExit("summary.csv missing gene_length_bp row")

    K_metric_name = "exon_length_bp" if args.K_source == "exon" else "intron_length_bp"
    if K_metric_name not in metrics:
        raise SystemExit(f"summary.csv missing {K_metric_name} row (did you compute introns?)")

    T_pcts = [p.strip() for p in args.T_pcts.split(",") if p.strip()]
    K_pcts = [p.strip() for p in args.K_pcts.split(",") if p.strip()]

    T_vals = choose_values(metrics["gene_length_bp"], T_pcts, args.round)
    K_vals = choose_values(metrics[K_metric_name], K_pcts, args.round)

    # Make paired configs by min length (or you can do cartesian in your benchmark driver)
    n = min(len(T_vals), len(K_vals))
    T_vals = T_vals[:n]
    K_vals = K_vals[:n]

    mode = f"{args.K_source.upper()}K"

    md = markdown_table(T_vals, K_vals, args.C, mode)

    cmd = (
        f"python {args.backend_cmd} "
        f"--T {','.join(map(str, T_vals))} "
        f"--K {','.join(map(str, K_vals))} "
        f"--C {args.C}"
    )

    print(md)
    print("Command:")
    print(cmd)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "benchmark_configs.md").write_text(md + "\n", encoding="utf-8")
        (args.out_dir / "benchmark_command.txt").write_text(cmd + "\n", encoding="utf-8")
        print(f"\nWrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
