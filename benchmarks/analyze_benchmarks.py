#!/usr/bin/env python3
"""
Analyze benchmark results and generate publication-quality plots.

Reads benchmark_full.csv and produces:
1. Scalability plots (time vs T, time vs KC)
2. Throughput comparisons (positions/sec, state-transitions/sec)
3. Backend crossover analysis (when to use which backend)
4. Semiring overhead analysis (cost of Max/Entropy vs Log)
5. Forward/backward ratio analysis
6. Memory efficiency plots
7. Ratio-to-baseline plots (time and memory vs a reference backend)

Example:
    python benchmarks/analyze_benchmarks.py \
        --input results/benchmark_full.csv \
        --output-dir results/plots/ \
        --format pdf
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Try to import plotting libraries (optional for headless analysis)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Colorblind-safe palette (IBM Design)
COLORS = {
    "linear_scan_streaming": "#DC267F",
    "triton": "#000000",
    "triton_pytorch": "#666666",
    "triton_checkpointing": "#648FFF",
}

MARKERS = {
    "linear_scan_streaming": "^",
    "triton": "*",
    "triton_pytorch": "x",
    "triton_checkpointing": "o",
}


@dataclass
class DerivedMetrics:
    """Derived metrics computed from raw benchmark results."""

    # Throughput
    positions_per_sec: float
    state_transitions_per_sec: float

    # Efficiency (normalized by problem size)
    memory_per_state_kb: float  # peak_gb * 1e6 / (T * KC)
    time_per_state_us: float  # time_ms * 1e3 / (T * KC)

    # Ratios
    backward_forward_ratio: float | None
    semiring_overhead: float | None


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(csv_path)

    # Filter to successful runs only for analysis
    df_success = df[df["status"] == "success"].copy()

    # Compute derived metrics
    df_success["positions_per_sec"] = (df_success["B"] * df_success["T"]) / (
        df_success["time_ms_median"] / 1000.0
    )
    df_success["state_transitions_per_sec"] = (
        df_success["B"] * df_success["T"] * df_success["KC"]
    ) / (df_success["time_ms_median"] / 1000.0)
    df_success["memory_per_state_kb"] = (
        df_success["peak_allocated_gb"] * 1e6 / (df_success["T"] * df_success["KC"])
    )
    df_success["time_per_state_us"] = (
        df_success["time_ms_median"] * 1e3 / (df_success["T"] * df_success["KC"])
    )

    return df_success


def compute_backward_forward_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backward/forward time ratios for each configuration."""
    # Pivot to get forward and backward times side by side
    ratios = []

    for (backend, semiring, T, K, C), group in df.groupby(["backend", "semiring", "T", "K", "C"]):
        fwd = group[group["phase"] == "forward"]["time_ms_median"].values
        bwd = group[group["phase"] == "backward"]["time_ms_median"].values

        if len(fwd) > 0 and len(bwd) > 0:
            ratios.append(
                {
                    "backend": backend,
                    "semiring": semiring,
                    "T": T,
                    "K": K,
                    "C": C,
                    "KC": K * C,
                    "forward_ms": fwd[0],
                    "backward_ms": bwd[0],
                    "ratio": bwd[0] / fwd[0] if fwd[0] > 0 else float("nan"),
                }
            )

    return pd.DataFrame(ratios)


def compute_semiring_overhead(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overhead of each semiring relative to LogSemiring."""
    overheads = []

    for (backend, phase, T, K, C), group in df.groupby(["backend", "phase", "T", "K", "C"]):
        log_time = group[group["semiring"] == "Log"]["time_ms_median"].values

        if len(log_time) == 0:
            continue

        log_time = log_time[0]

        for _, row in group.iterrows():
            overheads.append(
                {
                    "backend": backend,
                    "semiring": row["semiring"],
                    "phase": phase,
                    "T": T,
                    "K": K,
                    "C": C,
                    "KC": K * C,
                    "time_ms": row["time_ms_median"],
                    "log_time_ms": log_time,
                    "overhead": row["time_ms_median"] / log_time if log_time > 0 else float("nan"),
                }
            )

    return pd.DataFrame(overheads)


def find_crossover_points(df: pd.DataFrame, backend_a: str, backend_b: str) -> pd.DataFrame:
    """Find KC values where backend_a becomes faster than backend_b."""
    crossovers = []

    for (semiring, phase, T), group in df.groupby(["semiring", "phase", "T"]):
        a_data = group[group["backend"] == backend_a].sort_values("KC")
        b_data = group[group["backend"] == backend_b].sort_values("KC")

        if len(a_data) == 0 or len(b_data) == 0:
            continue

        # Merge on KC
        merged = pd.merge(a_data, b_data, on="KC", suffixes=("_a", "_b"))

        # Find where a becomes faster
        for _, row in merged.iterrows():
            if row["time_ms_median_a"] < row["time_ms_median_b"]:
                crossovers.append(
                    {
                        "semiring": semiring,
                        "phase": phase,
                        "T": T,
                        "crossover_KC": row["KC"],
                        "backend_a": backend_a,
                        "backend_b": backend_b,
                        "time_a_ms": row["time_ms_median_a"],
                        "time_b_ms": row["time_ms_median_b"],
                        "speedup": row["time_ms_median_b"] / row["time_ms_median_a"],
                    }
                )
                break  # First crossover point

    return pd.DataFrame(crossovers)


def compute_baseline_ratios(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compute time and memory ratios relative to a baseline backend.

    Args:
        df: DataFrame with benchmark results
        baseline: Name of baseline backend to compare against

    Returns:
        DataFrame with time_ratio and memory_ratio columns added
    """
    if baseline not in df["backend"].unique():
        print(f"Warning: baseline '{baseline}' not found in results")
        return pd.DataFrame()

    ratios = []

    # Group by configuration (semiring, phase, T, K, C) and compute ratios
    for (semiring, phase, T, K, C), group in df.groupby(["semiring", "phase", "T", "K", "C"]):
        baseline_data = group[group["backend"] == baseline]

        if len(baseline_data) == 0:
            continue

        baseline_time = baseline_data["time_ms_median"].values[0]
        baseline_memory = baseline_data["peak_allocated_gb"].values[0]

        for _, row in group.iterrows():
            time_ratio = (
                row["time_ms_median"] / baseline_time if baseline_time > 0 else float("nan")
            )
            memory_ratio = (
                row["peak_allocated_gb"] / baseline_memory if baseline_memory > 0 else float("nan")
            )

            ratios.append(
                {
                    "backend": row["backend"],
                    "semiring": semiring,
                    "phase": phase,
                    "T": T,
                    "K": K,
                    "C": C,
                    "KC": K * C,
                    "time_ms": row["time_ms_median"],
                    "baseline_time_ms": baseline_time,
                    "time_ratio": time_ratio,
                    "peak_gb": row["peak_allocated_gb"],
                    "baseline_memory_gb": baseline_memory,
                    "memory_ratio": memory_ratio,
                }
            )

    return pd.DataFrame(ratios)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_scalability_T(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
    fixed_KC: int | None = None,
):
    """Plot time vs sequence length T (log-log scale)."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    subset = df[(df["phase"] == phase) & (df["semiring"] == semiring)]

    if fixed_KC is not None:
        subset = subset[subset["KC"] == fixed_KC]
    else:
        # Use median KC value
        fixed_KC = int(subset["KC"].median())
        subset = subset[subset["KC"] == fixed_KC]

    fig, ax = plt.subplots(figsize=(8, 6))

    for backend in subset["backend"].unique():
        data = subset[subset["backend"] == backend].sort_values("T")
        if len(data) > 1:
            ax.loglog(
                data["T"],
                data["time_ms_median"],
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Sequence Length T", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Scalability: Time vs T (KC={fixed_KC}, {semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add reference slopes
    T_range = subset["T"].unique()
    if len(T_range) >= 2:
        T_min, T_max = T_range.min(), T_range.max()
        y_mid = subset["time_ms_median"].median()
        # O(N) reference
        ax.loglog(
            [T_min, T_max],
            [y_mid * (T_min / T_max), y_mid],
            "--",
            color="gray",
            alpha=0.5,
            label="O(N)",
        )

    plt.tight_layout()
    plt.savefig(output_dir / f"scalability_T_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_scalability_KC(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
    fixed_T: int | None = None,
):
    """Plot time vs state-space size KC."""
    if not HAS_MATPLOTLIB:
        return

    subset = df[(df["phase"] == phase) & (df["semiring"] == semiring)]

    if fixed_T is not None:
        subset = subset[subset["T"] == fixed_T]
    else:
        fixed_T = int(subset["T"].median())
        subset = subset[subset["T"] == fixed_T]

    fig, ax = plt.subplots(figsize=(8, 6))

    for backend in subset["backend"].unique():
        data = subset[subset["backend"] == backend].sort_values("KC")
        if len(data) > 1:
            ax.plot(
                data["KC"],
                data["time_ms_median"],
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Scalability: Time vs KC (T={fixed_T}, {semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"scalability_KC_{semiring}_{phase}_T{fixed_T}.{fmt}", dpi=150)
    plt.close()


def plot_throughput(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
):
    """Plot throughput (positions/sec) vs KC."""
    if not HAS_MATPLOTLIB:
        return

    subset = df[(df["phase"] == phase) & (df["semiring"] == semiring)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for backend in subset["backend"].unique():
        data = (
            subset[subset["backend"] == backend]
            .groupby("KC")
            .agg({"positions_per_sec": "median"})
            .reset_index()
        )
        if len(data) > 1:
            ax.plot(
                data["KC"],
                data["positions_per_sec"] / 1e6,
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel("Throughput (M positions/sec)", fontsize=12)
    ax.set_title(f"Throughput vs State Space ({semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_backward_forward_ratio(
    ratios_df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    semiring: str = "Log",
):
    """Plot backward/forward time ratio by backend."""
    if not HAS_MATPLOTLIB or len(ratios_df) == 0:
        return

    subset = ratios_df[ratios_df["semiring"] == semiring]

    fig, ax = plt.subplots(figsize=(10, 6))

    backends = subset["backend"].unique()
    x = range(len(backends))

    # Compute mean and std for each backend
    means = []
    stds = []
    for backend in backends:
        data = subset[subset["backend"] == backend]["ratio"]
        means.append(data.mean())
        stds.append(data.std())

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS.get(b, "#333333") for b in backends],
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=45, ha="right")
    ax.set_ylabel("Backward / Forward Time Ratio", fontsize=12)
    ax.set_title(f"Backward Pass Overhead ({semiring})", fontsize=14)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"backward_forward_ratio_{semiring}.{fmt}", dpi=150)
    plt.close()


def plot_semiring_overhead(
    overhead_df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
):
    """Plot semiring overhead relative to LogSemiring."""
    if not HAS_MATPLOTLIB or len(overhead_df) == 0:
        return

    subset = overhead_df[overhead_df["phase"] == phase]

    fig, ax = plt.subplots(figsize=(10, 6))

    backends = subset["backend"].unique()
    semirings = [s for s in subset["semiring"].unique() if s != "Log"]

    x = range(len(backends))
    width = 0.8 / len(semirings)

    for i, semiring in enumerate(semirings):
        means = []
        stds = []
        for backend in backends:
            data = subset[(subset["backend"] == backend) & (subset["semiring"] == semiring)][
                "overhead"
            ]
            if len(data) > 0:
                means.append(data.mean())
                stds.append(data.std())
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(semirings) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            means,
            width,
            yerr=stds,
            capsize=3,
            label=semiring,
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=45, ha="right")
    ax.set_ylabel("Overhead vs LogSemiring", fontsize=12)
    ax.set_title(f"Semiring Overhead ({phase})", fontsize=14)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="LogSemiring baseline")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"semiring_overhead_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_memory_efficiency(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
):
    """Plot memory efficiency (GB per state-position)."""
    if not HAS_MATPLOTLIB:
        return

    subset = df[(df["phase"] == phase) & (df["semiring"] == semiring)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for backend in subset["backend"].unique():
        data = (
            subset[subset["backend"] == backend]
            .groupby("KC")
            .agg({"memory_per_state_kb": "median"})
            .reset_index()
        )
        if len(data) > 1:
            ax.plot(
                data["KC"],
                data["memory_per_state_kb"],
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel("Memory per State (KB)", fontsize=12)
    ax.set_title(f"Memory Efficiency ({semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"memory_efficiency_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_time_ratio_to_baseline(
    baseline_df: pd.DataFrame,
    baseline: str,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
):
    """Plot time ratio vs baseline backend across KC values."""
    if not HAS_MATPLOTLIB or len(baseline_df) == 0:
        return

    subset = baseline_df[(baseline_df["phase"] == phase) & (baseline_df["semiring"] == semiring)]
    # Exclude baseline from plot (it would just be a flat line at 1.0)
    subset = subset[subset["backend"] != baseline]

    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for backend in subset["backend"].unique():
        data = (
            subset[subset["backend"] == backend]
            .groupby("KC")
            .agg({"time_ratio": "median"})
            .reset_index()
        )
        if len(data) > 1:
            ax.plot(
                data["KC"],
                data["time_ratio"],
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=2, label=f"{baseline} (baseline)"
    )
    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel(f"Time Ratio (vs {baseline})", fontsize=12)
    ax.set_title(f"Time Ratio to Baseline ({semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add helper text
    ax.text(
        0.98,
        0.02,
        "< 1.0 = faster than baseline",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"time_ratio_baseline_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_memory_ratio_to_baseline(
    baseline_df: pd.DataFrame,
    baseline: str,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
):
    """Plot memory ratio vs baseline backend across KC values."""
    if not HAS_MATPLOTLIB or len(baseline_df) == 0:
        return

    subset = baseline_df[(baseline_df["phase"] == phase) & (baseline_df["semiring"] == semiring)]
    # Exclude baseline from plot
    subset = subset[subset["backend"] != baseline]

    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for backend in subset["backend"].unique():
        data = (
            subset[subset["backend"] == backend]
            .groupby("KC")
            .agg({"memory_ratio": "median"})
            .reset_index()
        )
        if len(data) > 1:
            ax.plot(
                data["KC"],
                data["memory_ratio"],
                marker=MARKERS.get(backend, "o"),
                color=COLORS.get(backend, "#333333"),
                label=backend,
                linewidth=2,
                markersize=8,
            )

    ax.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=2, label=f"{baseline} (baseline)"
    )
    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel(f"Memory Ratio (vs {baseline})", fontsize=12)
    ax.set_title(f"Memory Ratio to Baseline ({semiring}, {phase})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add helper text
    ax.text(
        0.98,
        0.02,
        "< 1.0 = less memory than baseline",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"memory_ratio_baseline_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


def plot_ratio_heatmap(
    baseline_df: pd.DataFrame,
    baseline: str,
    output_dir: Path,
    fmt: str = "pdf",
    phase: str = "both",
    semiring: str = "Log",
    metric: str = "time",
):
    """Plot heatmap of time or memory ratios (Backend x KC)."""
    if not HAS_MATPLOTLIB or len(baseline_df) == 0:
        return

    subset = baseline_df[(baseline_df["phase"] == phase) & (baseline_df["semiring"] == semiring)]
    subset = subset[subset["backend"] != baseline]

    if len(subset) == 0:
        return

    ratio_col = "time_ratio" if metric == "time" else "memory_ratio"

    # Pivot to create heatmap data
    pivot = subset.pivot_table(values=ratio_col, index="backend", columns="KC", aggfunc="median")

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.8)))

    # Create heatmap
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")

    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric.capitalize()} Ratio vs {baseline}", fontsize=11)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not math.isnan(val):
                text_color = "white" if val > 1.5 or val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xlabel("State Space Size (KC)", fontsize=12)
    ax.set_ylabel("Backend", fontsize=12)
    ax.set_title(f"{metric.capitalize()} Ratio Heatmap ({semiring}, {phase})", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_ratio_heatmap_{semiring}_{phase}.{fmt}", dpi=150)
    plt.close()


# =============================================================================
# Summary Tables
# =============================================================================


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by backend."""
    summary = (
        df.groupby(["backend", "semiring", "phase"])
        .agg(
            {
                "time_ms_median": ["mean", "std", "min", "max"],
                "peak_allocated_gb": ["mean", "max"],
                "positions_per_sec": ["mean", "max"],
                "KC": ["min", "max"],
                "T": ["min", "max"],
            }
        )
        .round(3)
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def generate_crossover_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Find crossover points between streaming scan and other backends."""
    crossovers = []

    # Compare streaming scan vs Triton backends
    streaming_backends = ["linear_scan_streaming"]
    triton_backends = ["triton", "triton_pytorch", "triton_checkpointing"]

    for stream_backend in streaming_backends:
        for other_backend in triton_backends:
            cross_df = find_crossover_points(df, stream_backend, other_backend)
            if len(cross_df) > 0:
                crossovers.append(cross_df)

    if crossovers:
        return pd.concat(crossovers, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/benchmark_full.csv"),
        help="Input CSV file from benchmark_memory_analysis.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (tables only)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="linear_scan_vectorized",
        help="Baseline backend for ratio comparisons (default: linear_scan_vectorized)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading results from {args.input}...")
    df = load_results(args.input)
    print(f"Loaded {len(df)} successful benchmark results")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compute derived data
    print("Computing derived metrics...")
    ratios_df = compute_backward_forward_ratios(df)
    overhead_df = compute_semiring_overhead(df)
    crossover_df = generate_crossover_summary(df)
    baseline_df = compute_baseline_ratios(df, args.baseline)

    # Generate summary tables
    print("Generating summary tables...")
    summary = generate_summary_table(df)
    summary.to_csv(args.output_dir / "summary_stats.csv", index=False)
    print("  Saved summary_stats.csv")

    if len(ratios_df) > 0:
        ratios_df.to_csv(args.output_dir / "backward_forward_ratios.csv", index=False)
        print("  Saved backward_forward_ratios.csv")

    if len(overhead_df) > 0:
        overhead_df.to_csv(args.output_dir / "semiring_overhead.csv", index=False)
        print("  Saved semiring_overhead.csv")

    if len(crossover_df) > 0:
        crossover_df.to_csv(args.output_dir / "crossover_points.csv", index=False)
        print("  Saved crossover_points.csv")

    if len(baseline_df) > 0:
        baseline_df.to_csv(args.output_dir / "baseline_ratios.csv", index=False)
        print(f"  Saved baseline_ratios.csv (baseline: {args.baseline})")

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print(f"Generating plots (format: {args.format})...")

        semirings = df["semiring"].unique()
        phases = df["phase"].unique()

        for semiring in semirings:
            for phase in phases:
                # Scalability plots
                plot_scalability_T(df, args.output_dir, args.format, phase, semiring)
                plot_scalability_KC(df, args.output_dir, args.format, phase, semiring)

                # Throughput
                plot_throughput(df, args.output_dir, args.format, phase, semiring)

                # Memory efficiency
                plot_memory_efficiency(df, args.output_dir, args.format, phase, semiring)

                # Ratio-to-baseline plots
                if len(baseline_df) > 0:
                    plot_time_ratio_to_baseline(
                        baseline_df, args.baseline, args.output_dir, args.format, phase, semiring
                    )
                    plot_memory_ratio_to_baseline(
                        baseline_df, args.baseline, args.output_dir, args.format, phase, semiring
                    )
                    plot_ratio_heatmap(
                        baseline_df,
                        args.baseline,
                        args.output_dir,
                        args.format,
                        phase,
                        semiring,
                        "time",
                    )
                    plot_ratio_heatmap(
                        baseline_df,
                        args.baseline,
                        args.output_dir,
                        args.format,
                        phase,
                        semiring,
                        "memory",
                    )

            # Backward/forward ratio (per semiring)
            plot_backward_forward_ratio(ratios_df, args.output_dir, args.format, semiring)

        # Semiring overhead (per phase)
        for phase in phases:
            plot_semiring_overhead(overhead_df, args.output_dir, args.format, phase)

        print(f"  Plots saved to {args.output_dir}/")
    elif not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")

    # Print summary to console
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nBackends by median throughput (positions/sec):")
    throughput_summary = (
        df.groupby("backend")["positions_per_sec"].median().sort_values(ascending=False)
    )
    for backend, throughput in throughput_summary.items():
        print(f"  {backend:30s}: {throughput/1e6:.2f}M pos/sec")

    if len(ratios_df) > 0:
        print("\nBackward/Forward time ratios:")
        ratio_summary = ratios_df.groupby("backend")["ratio"].mean().sort_values()
        for backend, ratio in ratio_summary.items():
            print(f"  {backend:30s}: {ratio:.2f}x")

    if len(crossover_df) > 0:
        print("\nCrossover points (streaming becomes faster):")
        for _, row in crossover_df.iterrows():
            print(
                f"  {row['backend_a']} beats {row['backend_b']} at KC >= {row['crossover_KC']}"
                f" ({row['speedup']:.2f}x speedup)"
            )

    if len(baseline_df) > 0:
        print(f"\nTime ratio vs {args.baseline} (median across configs):")
        time_ratio_summary = (
            baseline_df[baseline_df["backend"] != args.baseline]
            .groupby("backend")["time_ratio"]
            .median()
            .sort_values()
        )
        for backend, ratio in time_ratio_summary.items():
            speedup_str = f"{1/ratio:.2f}x faster" if ratio < 1 else f"{ratio:.2f}x slower"
            print(f"  {backend:30s}: {ratio:.3f} ({speedup_str})")

    print(f"\nFull results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
