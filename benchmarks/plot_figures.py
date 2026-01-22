#!/usr/bin/env python3
"""
Generate publication-quality figures for Semi-Markov CRF backend analysis.

Figures:
1. Time ratio vs state size (per T, with IQR bands)
2. Memory ratio vs state size (per T, with IQR bands)
3. OOM frontier summary
4. Backend comparison bars (representative config)

All metrics are normalized to the vectorized linear scan baseline.

Example:
    python benchmarks/plot_figures.py \
        --input-dir results/ \
        --output-dir figures/
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        return True
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return False


# -----------------------------------------------------------------------------
# Backend styling
# -----------------------------------------------------------------------------

BACKEND_ORDER = [
    "linear_scan",
    "linear_scan_vectorized",
    "linear_scan_streaming",
    "triton_streaming",
    "binary_tree",
    "binary_tree_sharded",
    "block_triangular",
    "banded",
]

BACKEND_LABELS = {
    "linear_scan": "Linear Scan",
    "linear_scan_vectorized": "Vectorized Linear",
    "linear_scan_streaming": "Streaming Linear",
    "triton_streaming": "Triton Streaming",
    "binary_tree": "Binary Tree",
    "binary_tree_sharded": "Sharded Tree",
    "block_triangular": "Block Triangular",
    "banded": "Banded",
}

# Colorblind-safe palette
BACKEND_COLORS = {
    "linear_scan": "#1f77b4",  # Blue
    "linear_scan_vectorized": "#ff7f0e",  # Orange (baseline)
    "linear_scan_streaming": "#2ca02c",  # Green
    "triton_streaming": "#000000",  # Black
    "binary_tree": "#d62728",  # Red
    "binary_tree_sharded": "#9467bd",  # Purple
    "block_triangular": "#8c564b",  # Brown
    "banded": "#e377c2",  # Pink
}

BACKEND_MARKERS = {
    "linear_scan": "o",
    "linear_scan_vectorized": "s",
    "linear_scan_streaming": "D",
    "triton_streaming": "h",  # Hexagon
    "binary_tree": "^",
    "binary_tree_sharded": "v",
    "block_triangular": "p",
    "banded": "H",  # Filled hexagon
}


def get_backends_from_data(df: pd.DataFrame) -> list[str]:
    """Extract and sort backends found in data."""
    backends_in_data = set(df["backend"].unique())

    def sort_key(b):
        if b in BACKEND_ORDER:
            return (0, BACKEND_ORDER.index(b))
        return (1, b)

    return sorted(backends_in_data, key=sort_key)


def get_backend_label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend.replace("_", " ").title())


def get_backend_color(backend: str, idx: int = 0) -> str:
    if backend in BACKEND_COLORS:
        return BACKEND_COLORS[backend]
    fallback_colors = ["#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78"]
    return fallback_colors[idx % len(fallback_colors)]


def get_backend_marker(backend: str, idx: int = 0) -> str:
    if backend in BACKEND_MARKERS:
        return BACKEND_MARKERS[backend]
    fallback_markers = ["*", "X", "P", "<", ">"]
    return fallback_markers[idx % len(fallback_markers)]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def prepare_ratio_data(df: pd.DataFrame, baseline: str = "linear_scan_vectorized") -> pd.DataFrame:
    """
    Compute ratios vs baseline for time and memory.

    Args:
        df: Raw benchmark DataFrame with columns: T, K, C, backend, status,
            time_per_position_ms, peak_reserved_gb
        baseline: Backend to normalize against

    Returns:
        DataFrame with columns: T, K, C, state_n, backend, time_ratio, mem_ratio
    """
    # Define state size n = (K-1)*C (semi-Markov state space dimension)
    df = df.copy()
    df["state_n"] = (df["K"] - 1) * df["C"]

    # Keep only successful runs
    succ = df[df["status"] == "success"].copy()

    if len(succ) == 0:
        print("Warning: No successful runs found")
        return pd.DataFrame()

    # Check for required columns
    time_col = "time_per_position_ms"
    mem_col = "peak_reserved_gb"

    if time_col not in succ.columns:
        # Fall back to computing from time_ms_median
        if "time_ms_median" in succ.columns:
            succ[time_col] = succ["time_ms_median"] / succ["T"]
        else:
            print(f"Warning: Neither {time_col} nor time_ms_median found")
            return pd.DataFrame()

    if mem_col not in succ.columns:
        if "peak_allocated_gb" in succ.columns:
            mem_col = "peak_allocated_gb"
        else:
            print(f"Warning: {mem_col} not found")
            return pd.DataFrame()

    # Pivot to get baseline values
    pivot = succ.pivot_table(
        index=["T", "K", "C", "state_n"],
        columns="backend",
        values=[time_col, mem_col],
    )
    pivot.columns = ["__".join(c) for c in pivot.columns]
    pivot = pivot.reset_index()

    # Get baseline columns
    base_time_col = f"{time_col}__{baseline}"
    base_mem_col = f"{mem_col}__{baseline}"

    if base_time_col not in pivot.columns:
        print(f"Warning: Baseline {baseline} not found in data")
        return pd.DataFrame()

    base_time = pivot[base_time_col]
    base_mem = pivot[base_mem_col]

    # Build ratio dataframe for each backend
    ratio_dfs = []
    backends = [b for b in get_backends_from_data(succ) if b != baseline]

    for backend in backends:
        tcol = f"{time_col}__{backend}"
        mcol = f"{mem_col}__{backend}"

        if tcol not in pivot.columns:
            continue

        ratio_df = pivot[["T", "K", "C", "state_n"]].copy()
        ratio_df["backend"] = backend
        ratio_df["time_ratio"] = pivot[tcol] / base_time
        ratio_df["mem_ratio"] = pivot[mcol] / base_mem
        ratio_df = ratio_df.dropna(subset=["time_ratio", "mem_ratio"])
        ratio_dfs.append(ratio_df)

    if not ratio_dfs:
        return pd.DataFrame()

    return pd.concat(ratio_dfs, ignore_index=True)


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------


def plot_ratio_vs_state(
    ratios: pd.DataFrame,
    output_dir: Path,
    metric: str = "time",
    T_values: Optional[list[int]] = None,
    baseline_label: str = "Vectorized Linear",
):
    """
    Plot ratio vs state size with IQR bands, split by T.

    Args:
        ratios: DataFrame from prepare_ratio_data()
        output_dir: Where to save figures
        metric: 'time' or 'mem'
        T_values: List of T values to plot (separate subplot each)
        baseline_label: Human-readable name of baseline
    """
    import matplotlib.pyplot as plt

    if len(ratios) == 0:
        print(f"Skipping {metric} ratio plot - no data")
        return

    if T_values is None:
        T_values = sorted(ratios["T"].unique())

    ratio_col = f"{metric}_ratio"
    if ratio_col not in ratios.columns:
        print(f"Skipping {metric} ratio plot - column not found")
        return

    backends = sorted(
        ratios["backend"].unique(),
        key=lambda b: BACKEND_ORDER.index(b) if b in BACKEND_ORDER else 999,
    )

    # Create figure with subplots for each T
    n_plots = len(T_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, T in enumerate(T_values):
        ax = axes[ax_idx]
        t_data = ratios[ratios["T"] == T]

        for b_idx, backend in enumerate(backends):
            b_data = t_data[t_data["backend"] == backend]
            if len(b_data) == 0:
                continue

            # Aggregate by state_n: median and IQR
            agg = b_data.groupby("state_n")[ratio_col].agg(
                ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            )
            agg.columns = ["median", "q25", "q75"]
            agg = agg.reset_index().sort_values("state_n")

            color = get_backend_color(backend, b_idx)
            marker = get_backend_marker(backend, b_idx)
            label = get_backend_label(backend)

            # Plot median line with markers
            ax.plot(
                agg["state_n"],
                agg["median"],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=label,
            )

            # Add IQR band
            ax.fill_between(agg["state_n"], agg["q25"], agg["q75"], color=color, alpha=0.15)

        # Reference line at ratio = 1
        ax.axhline(1.0, color="black", linewidth=1, linestyle="-", alpha=0.5)

        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("State size n = (K-1) * C")
        ax.set_title(f"T = {T}")
        ax.grid(True, alpha=0.3, which="both")

        # Set reasonable y-axis limits
        ax.set_ylim(0.1, 100)

    # Y-label only on first subplot
    if metric == "time":
        axes[0].set_ylabel(f"Time ratio vs {baseline_label}\n(log scale, lower is better)")
    else:
        axes[0].set_ylabel(f"Memory ratio vs {baseline_label}\n(log scale, lower is better)")

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(backends), 6),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            fontsize=9,
        )

    if metric == "time":
        fig.suptitle("Runtime vs State Size\n(median with 25-75% IQR band)", fontsize=12)
    else:
        fig.suptitle("Peak Memory vs State Size\n(median with 25-75% IQR band)", fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    output_path = output_dir / f"{metric}_ratio_vs_state.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_oom_frontier(df: pd.DataFrame, output_dir: Path):
    """
    Plot OOM frontier: max feasible state_n vs T for each backend.
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    df["state_n"] = (df["K"] - 1) * df["C"]

    backends = get_backends_from_data(df)
    T_values = sorted(df["T"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for b_idx, backend in enumerate(backends):
        max_state_per_T = []
        for T in T_values:
            successful = df[
                (df["T"] == T) & (df["backend"] == backend) & (df["status"] == "success")
            ]
            if len(successful) > 0:
                max_state = successful["state_n"].max()
            else:
                max_state = 0
            max_state_per_T.append(max_state)

        if any(s > 0 for s in max_state_per_T):
            color = get_backend_color(backend, b_idx)
            marker = get_backend_marker(backend, b_idx)
            ax.plot(
                T_values,
                max_state_per_T,
                marker=marker,
                color=color,
                label=get_backend_label(backend),
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Sequence Length (T)")
    ax.set_ylabel("Max Feasible State Size n = (K-1) * C")
    ax.set_title("OOM Frontier by Backend\n(higher is better)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "oom_frontier.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_backend_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Bar chart comparing all backends at a representative config.
    """
    import matplotlib.pyplot as plt

    backends = get_backends_from_data(df)

    # Find config with most backend coverage
    success_df = df[df["status"] == "success"]
    if len(success_df) == 0:
        print("No successful runs for backend comparison")
        return

    config_counts = success_df.groupby(["T", "K", "C"]).size().reset_index(name="count")
    config_counts = config_counts.sort_values("count", ascending=False)
    best_config = config_counts.iloc[0]
    T, K, C = int(best_config["T"]), int(best_config["K"]), int(best_config["C"])

    subset = df[(df["T"] == T) & (df["K"] == K) & (df["C"] == C)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_positions = []
    x_labels = []
    times = []
    memories = []
    colors = []

    for b_idx, backend in enumerate(backends):
        row = subset[subset["backend"] == backend]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        x_positions.append(len(x_labels))
        x_labels.append(get_backend_label(backend))
        colors.append(get_backend_color(backend, b_idx))

        if row["status"] == "success":
            times.append(row.get("time_ms_median", 0))
            memories.append(row.get("peak_reserved_gb", row.get("peak_allocated_gb", 0)))
        else:
            times.append(0)
            memories.append(0)

    # Time bars
    ax1.bar(x_positions, times, color=colors, edgecolor="white")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Forward+Backward Time\n(T={T}, K={K}, C={C})")

    for i, t in enumerate(times):
        if t == 0:
            ax1.text(
                i,
                ax1.get_ylim()[1] * 0.5,
                "OOM",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

    # Memory bars
    ax2.bar(x_positions, memories, color=colors, edgecolor="white")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Peak Memory (GB)")
    ax2.set_title(f"Peak Reserved Memory\n(T={T}, K={K}, C={C})")

    for i, m in enumerate(memories):
        if m == 0:
            ax2.text(
                i,
                ax2.get_ylim()[1] * 0.5,
                "OOM",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

    plt.tight_layout()

    output_path = output_dir / "backend_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_absolute_metrics(df: pd.DataFrame, output_dir: Path, T_values: Optional[list[int]] = None):
    """
    Plot absolute time and memory vs state size (not ratios).
    Useful when baseline is missing or for overall picture.
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    df["state_n"] = (df["K"] - 1) * df["C"]
    succ = df[df["status"] == "success"]

    if len(succ) == 0:
        print("No successful runs for absolute metrics")
        return

    if T_values is None:
        T_values = sorted(succ["T"].unique())

    backends = get_backends_from_data(succ)

    # Time plot
    n_plots = len(T_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, T in enumerate(T_values):
        ax = axes[ax_idx]
        t_data = succ[succ["T"] == T]

        for b_idx, backend in enumerate(backends):
            b_data = t_data[t_data["backend"] == backend].sort_values("state_n")
            if len(b_data) == 0:
                continue

            color = get_backend_color(backend, b_idx)
            marker = get_backend_marker(backend, b_idx)

            # Use time_per_position if available, else compute
            if "time_per_position_ms" in b_data.columns:
                y = b_data["time_per_position_ms"]
            elif "time_ms_median" in b_data.columns:
                y = b_data["time_ms_median"] / b_data["T"]
            else:
                continue

            ax.plot(
                b_data["state_n"],
                y,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=get_backend_label(backend),
            )

        ax.set_yscale("log")
        ax.set_xlabel("State size n = (K-1) * C")
        ax.set_title(f"T = {T}")
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel("Time per position (ms)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(backends), 6),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle("Time per Position vs State Size", fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    output_path = output_dir / "time_absolute_vs_state.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

    # Memory plot
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, T in enumerate(T_values):
        ax = axes[ax_idx]
        t_data = succ[succ["T"] == T]

        for b_idx, backend in enumerate(backends):
            b_data = t_data[t_data["backend"] == backend].sort_values("state_n")
            if len(b_data) == 0:
                continue

            color = get_backend_color(backend, b_idx)
            marker = get_backend_marker(backend, b_idx)

            mem_col = (
                "peak_reserved_gb" if "peak_reserved_gb" in b_data.columns else "peak_allocated_gb"
            )
            if mem_col not in b_data.columns:
                continue

            ax.plot(
                b_data["state_n"],
                b_data[mem_col],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=get_backend_label(backend),
            )

        ax.set_yscale("log")
        ax.set_xlabel("State size n = (K-1) * C")
        ax.set_title(f"T = {T}")
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel("Peak Memory (GB)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(backends), 6),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle("Peak Memory vs State Size", fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    output_path = output_dir / "memory_absolute_vs_state.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--T", type=str, default=None, help="Comma-separated T values to plot (default: all)"
    )
    parser.add_argument(
        "--csv", type=str, default="benchmark_full.csv", help="CSV filename within input-dir"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="linear_scan_vectorized",
        help="Backend to normalize ratios against",
    )
    args = parser.parse_args()

    if not check_matplotlib():
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = args.input_dir / args.csv
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    backends = get_backends_from_data(df)
    print(f"Backends: {', '.join(backends)}")

    # Parse T values
    if args.T:
        T_values = [int(x) for x in args.T.split(",")]
    else:
        T_values = sorted(df["T"].unique())
    print(f"T values: {T_values}")

    print("\nGenerating figures...")

    # Prepare ratio data
    ratios = prepare_ratio_data(df, baseline=args.baseline)

    if len(ratios) > 0:
        # Ratio plots (main figures)
        plot_ratio_vs_state(
            ratios,
            args.output_dir,
            metric="time",
            T_values=T_values,
            baseline_label=get_backend_label(args.baseline),
        )
        plot_ratio_vs_state(
            ratios,
            args.output_dir,
            metric="mem",
            T_values=T_values,
            baseline_label=get_backend_label(args.baseline),
        )
    else:
        print("Warning: Could not compute ratios - baseline may be missing")

    # Absolute metric plots (useful regardless)
    plot_absolute_metrics(df, args.output_dir, T_values)

    # OOM frontier
    plot_oom_frontier(df, args.output_dir)

    # Backend comparison bars
    plot_backend_comparison(df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
