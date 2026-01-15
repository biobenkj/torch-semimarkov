"""Output and result saving utilities for benchmarks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .runner import BenchmarkResult


def save_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    backends: list[str],
    semirings: list[str],
    phases: list[str],
) -> None:
    """
    Save benchmark results in multiple formats.

    Saves:
    - benchmark_full.csv: Full results with all metrics
    - heatmap_data.json: Data for OOM feasibility heatmaps
    - memory_breakdown.csv: Memory breakdown by category
    """
    if not results:
        print("No results to save")
        return

    # 1. Full CSV with all metrics
    csv_path = output_dir / "benchmark_full.csv"
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nSaved full results to {csv_path}")

    # 2. Heatmap data (for OOM feasibility figures)
    heatmap_path = output_dir / "heatmap_data.json"
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
    breakdown_path = output_dir / "memory_breakdown.csv"
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


def print_summary(
    results: list[BenchmarkResult],
    backends: list[str],
    semirings: list[str],
    phases: list[str],
) -> None:
    """Print summary statistics for benchmark results."""
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
                skipped = sum(
                    1 for r in key_results if r.status in ("not_tested", "not_supported")
                )

                successful = [r for r in key_results if r.status == "success"]
                if successful:
                    max_kc = max(r.KC for r in successful)
                    max_mem = max(r.peak_allocated_gb for r in successful)
                else:
                    max_kc = 0
                    max_mem = 0

                label = f"{backend}/{semiring_name}/{phase}"
                print(
                    f"{label:40s}: {success:3d} success, {oom:3d} OOM, {skipped:3d} skipped | "
                    f"max KC={max_kc:4d}, max mem={max_mem:.2f}GB"
                )
