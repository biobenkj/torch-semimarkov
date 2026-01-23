#!/usr/bin/env python3
"""
Unified Benchmark Runner with Plotting

This script runs the practical demonstration benchmarks (GENCODE and/or TIMIT),
saves structured results to JSON, and generates publication-quality figures.

Usage:
    # Run GENCODE benchmark
    python run_benchmarks.py --task gencode --data-dir data/gencode/ --output-dir results/gencode/

    # Run TIMIT benchmark
    python run_benchmarks.py --task timit --data-dir data/timit/ --output-dir results/timit/

    # Run both benchmarks
    python run_benchmarks.py --task all --gencode-dir data/gencode/ --timit-dir data/timit/ --output-dir results/

    # Quick test run
    python run_benchmarks.py --task gencode --data-dir data/gencode/ --output-dir results/test/ \
        --epochs 2 --max-duration 50

Outputs:
    results/
    ├── gencode/
    │   ├── metrics.json
    │   ├── comparison_table.txt
    │   └── figures/
    │       ├── comparison_bar.pdf
    │       ├── boundary_tolerance.pdf
    │       └── duration_kl.pdf
    └── timit/
        ├── metrics.json
        ├── comparison_table.txt
        └── figures/
            ├── comparison_bar.pdf
            └── boundary_tolerance.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Colorblind-safe palette (IBM Design)
COLORS = {
    "semi_crf": "#DC267F",   # Pink/Magenta
    "linear_crf": "#648FFF",  # Blue
    "highlight": "#FFB000",   # Gold
}


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_comparison_bar(
    results: dict[str, Any],
    task: str,
    output_path: Path,
) -> None:
    """
    Generate side-by-side bar chart comparing Linear CRF vs Semi-CRF.

    Args:
        results: Dict with 'linear_crf' and 'semi_crf' keys containing metric dicts
        task: 'gencode' or 'timit'
        output_path: Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping comparison bar plot")
        return

    linear = results["linear_crf"]
    semi = results["semi_crf"]

    if task == "gencode":
        metrics = ["position_f1_macro", "boundary_f1", "segment_f1"]
        labels = ["Position F1\n(macro)", "Boundary F1", "Segment F1"]
        ylabel = "F1 Score"
        title = "GENCODE Exon/Intron Segmentation"
    else:
        metrics = ["boundary_f1", "segment_f1"]
        labels = ["Boundary F1", "Segment F1"]
        ylabel = "F1 Score"
        title = "TIMIT Phoneme Segmentation"

    linear_vals = [linear.get(m, 0) for m in metrics]
    semi_vals = [semi.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, linear_vals, width, label="Linear CRF (K=1)",
                   color=COLORS["linear_crf"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, semi_vals, width, label="Semi-CRF",
                   color=COLORS["semi_crf"], edgecolor="black", linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=9)

    # Add PER for TIMIT (inverted - lower is better)
    if task == "timit":
        ax2 = ax.twinx()
        per_x = len(metrics)
        linear_per = linear.get("phone_error_rate", 0)
        semi_per = semi.get("phone_error_rate", 0)

        # Add PER bars on secondary axis
        ax.bar(per_x - width/2, 0, width)  # Placeholder
        ax.bar(per_x + width/2, 0, width)  # Placeholder

        ax2.bar(per_x - width/2, linear_per, width, color=COLORS["linear_crf"],
               edgecolor="black", linewidth=0.5, alpha=0.7)
        ax2.bar(per_x + width/2, semi_per, width, color=COLORS["semi_crf"],
               edgecolor="black", linewidth=0.5, alpha=0.7)

        ax2.set_ylabel("Phone Error Rate (↓)", fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax.set_xticks(list(x) + [per_x])
        ax.set_xticklabels(labels + ["PER\n(lower=better)"], fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison bar chart to {output_path}")


def plot_boundary_tolerance(
    results: dict[str, Any],
    task: str,
    output_path: Path,
) -> None:
    """
    Generate line plot showing Boundary F1 vs tolerance level.

    Args:
        results: Dict with 'linear_crf' and 'semi_crf' keys
        task: 'gencode' or 'timit'
        output_path: Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping boundary tolerance plot")
        return

    linear = results["linear_crf"]
    semi = results["semi_crf"]

    # Get tolerance values (stored as string keys in JSON)
    if task == "gencode":
        tolerances = [0, 1, 2, 5, 10]
        xlabel = "Tolerance (bp)"
        title = "GENCODE: Boundary F1 vs Tolerance"
    else:
        tolerances = [0, 1, 2]
        xlabel = "Tolerance (frames)"
        title = "TIMIT: Boundary F1 vs Tolerance"

    # Extract values
    linear_tol = linear.get("boundary_f1_tolerance", linear.get("boundary_f1_tolerances", {}))
    semi_tol = semi.get("boundary_f1_tolerance", semi.get("boundary_f1_tolerances", {}))

    linear_vals = [linear_tol.get(str(t), linear_tol.get(t, 0)) for t in tolerances]
    semi_vals = [semi_tol.get(str(t), semi_tol.get(t, 0)) for t in tolerances]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(tolerances, linear_vals, "o--", color=COLORS["linear_crf"],
            label="Linear CRF (K=1)", linewidth=2, markersize=8)
    ax.plot(tolerances, semi_vals, "o-", color=COLORS["semi_crf"],
            label="Semi-CRF", linewidth=2, markersize=8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Boundary F1", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(tolerances)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Highlight the advantage at exact match (tol=0)
    if len(linear_vals) > 0 and len(semi_vals) > 0:
        delta = semi_vals[0] - linear_vals[0]
        if delta > 0:
            ax.annotate(f"+{delta:.3f}",
                       xy=(0, semi_vals[0]),
                       xytext=(0.5, semi_vals[0] + 0.05),
                       fontsize=10, color=COLORS["semi_crf"],
                       arrowprops=dict(arrowstyle="->", color=COLORS["semi_crf"]))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved boundary tolerance plot to {output_path}")


def plot_duration_kl(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate grouped bar chart of Duration KL divergence per class (GENCODE only).

    Args:
        results: Dict with 'linear_crf' and 'semi_crf' keys
        output_path: Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping duration KL plot")
        return

    linear = results["linear_crf"]
    semi = results["semi_crf"]

    linear_kl = linear.get("duration_kl", {})
    semi_kl = semi.get("duration_kl", {})

    if not linear_kl or not semi_kl:
        logger.warning("No duration KL data available, skipping plot")
        return

    classes = list(linear_kl.keys())
    linear_vals = [linear_kl.get(c, 0) for c in classes]
    semi_vals = [semi_kl.get(c, 0) for c in classes]

    # Replace NaN with 0 for plotting
    linear_vals = [0 if np.isnan(v) else v for v in linear_vals]
    semi_vals = [0 if np.isnan(v) else v for v in semi_vals]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, linear_vals, width, label="Linear CRF (K=1)",
                   color=COLORS["linear_crf"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, semi_vals, width, label="Semi-CRF",
                   color=COLORS["semi_crf"], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("KL Divergence (lower = better)", fontsize=12)
    ax.set_title("GENCODE: Duration Distribution Calibration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved duration KL plot to {output_path}")


def generate_comparison_table(results: dict[str, Any], task: str) -> str:
    """Generate formatted comparison table as string."""
    linear = results["linear_crf"]
    semi = results["semi_crf"]

    lines = []
    lines.append("=" * 70)
    lines.append(f"COMPARISON: Linear CRF vs Semi-CRF ({task.upper()})")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Metric':<30} {'Linear CRF':>15} {'Semi-CRF':>15} {'Δ':>10}")
    lines.append("-" * 70)

    if task == "gencode":
        metrics = [
            ("position_f1_macro", "Position F1 (macro)"),
            ("boundary_f1", "Boundary F1"),
            ("segment_f1", "Segment F1"),
        ]
    else:
        metrics = [
            ("phone_error_rate", "Phone Error Rate ↓"),
            ("boundary_f1", "Boundary F1"),
            ("segment_f1", "Segment F1"),
        ]

    for key, name in metrics:
        l_val = linear.get(key, 0)
        s_val = semi.get(key, 0)
        delta = s_val - l_val
        lines.append(f"{name:<30} {l_val:>15.4f} {s_val:>15.4f} {delta:>+10.4f}")

    # Boundary F1 at tolerances
    lines.append("")
    lines.append("Boundary F1 at different tolerances:")

    linear_tol = linear.get("boundary_f1_tolerance", linear.get("boundary_f1_tolerances", {}))
    semi_tol = semi.get("boundary_f1_tolerance", semi.get("boundary_f1_tolerances", {}))

    tolerances = [0, 1, 2, 5, 10] if task == "gencode" else [0, 1, 2]
    for tol in tolerances:
        l_val = linear_tol.get(str(tol), linear_tol.get(tol, 0))
        s_val = semi_tol.get(str(tol), semi_tol.get(tol, 0))
        delta = s_val - l_val
        lines.append(f"  tol={tol:<3} {l_val:>15.4f} {s_val:>15.4f} {delta:>+10.4f}")

    # Duration KL for GENCODE
    if task == "gencode":
        linear_kl = linear.get("duration_kl", {})
        semi_kl = semi.get("duration_kl", {})
        if linear_kl:
            lines.append("")
            lines.append("Duration KL divergence (lower = better):")
            for label in linear_kl.keys():
                l_val = linear_kl.get(label, float("nan"))
                s_val = semi_kl.get(label, float("nan"))
                lines.append(f"  {label:<15} {l_val:>15.4f} {s_val:>15.4f}")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_gencode_benchmark(
    data_dir: Path,
    output_dir: Path,
    max_duration: int = 500,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run GENCODE benchmark and generate outputs."""
    # Import benchmark module
    sys.path.insert(0, str(Path(__file__).parent))
    from gencode.gencode_exon_intron import compare_models

    logger.info("Running GENCODE benchmark...")

    # Run comparison
    results = compare_models(
        data_dir,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Convert to dict
    output = {
        "task": "gencode",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_duration": max_duration,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "linear_crf": results["linear_crf"].to_dict(),
        "semi_crf": results["semi_crf"].to_dict(),
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Save comparison table
    table = generate_comparison_table(output, "gencode")
    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(table)

    # Generate plots
    plot_comparison_bar(output, "gencode", figures_dir / "comparison_bar.pdf")
    plot_boundary_tolerance(output, "gencode", figures_dir / "boundary_tolerance.pdf")
    plot_duration_kl(output, figures_dir / "duration_kl.pdf")

    return output


def run_timit_benchmark(
    data_dir: Path,
    output_dir: Path,
    max_duration: int = 30,
    hidden_dim: int = 256,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run TIMIT benchmark and generate outputs."""
    # Import benchmark module
    sys.path.insert(0, str(Path(__file__).parent))
    from timit.timit_phoneme import compare_models

    logger.info("Running TIMIT benchmark...")

    # Run comparison
    results = compare_models(
        data_dir,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Convert to dict
    output = {
        "task": "timit",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_duration": max_duration,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "linear_crf": results["linear_crf"].to_dict(),
        "semi_crf": results["semi_crf"].to_dict(),
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Save comparison table
    table = generate_comparison_table(output, "timit")
    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(table)

    # Generate plots
    plot_comparison_bar(output, "timit", figures_dir / "comparison_bar.pdf")
    plot_boundary_tolerance(output, "timit", figures_dir / "boundary_tolerance.pdf")

    return output


def plot_from_json(json_path: Path, output_dir: Path = None) -> None:
    """
    Regenerate plots from saved JSON results.

    Args:
        json_path: Path to metrics.json file
        output_dir: Output directory for figures (defaults to same dir as JSON)
    """
    with open(json_path) as f:
        results = json.load(f)

    task = results.get("task", "unknown")

    if output_dir is None:
        output_dir = json_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison_bar(results, task, output_dir / "comparison_bar.pdf")
    plot_boundary_tolerance(results, task, output_dir / "boundary_tolerance.pdf")

    if task == "gencode":
        plot_duration_kl(results, output_dir / "duration_kl.pdf")

    logger.info(f"Regenerated plots in {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--task", choices=["gencode", "timit", "all", "plot"],
                       required=True, help="Which benchmark to run")
    parser.add_argument("--data-dir", type=Path, help="Data directory (for single task)")
    parser.add_argument("--gencode-dir", type=Path, help="GENCODE data directory (for --task all)")
    parser.add_argument("--timit-dir", type=Path, help="TIMIT data directory (for --task all)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    # Training parameters
    parser.add_argument("--max-duration", type=int, default=None,
                       help="Max segment duration (default: 500 for GENCODE, 30 for TIMIT)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3, help="TIMIT only")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    # Plot-only mode
    parser.add_argument("--json-path", type=Path, help="Path to metrics.json for plot regeneration")

    args = parser.parse_args()

    if args.task == "plot":
        if not args.json_path:
            parser.error("--json-path required for --task plot")
        plot_from_json(args.json_path, args.output_dir)
        return

    if args.task == "gencode":
        if not args.data_dir:
            parser.error("--data-dir required for --task gencode")
        max_duration = args.max_duration or 500
        run_gencode_benchmark(
            args.data_dir,
            args.output_dir,
            max_duration=max_duration,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.task == "timit":
        if not args.data_dir:
            parser.error("--data-dir required for --task timit")
        max_duration = args.max_duration or 30
        run_timit_benchmark(
            args.data_dir,
            args.output_dir,
            max_duration=max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.task == "all":
        if args.gencode_dir:
            max_duration = args.max_duration or 500
            run_gencode_benchmark(
                args.gencode_dir,
                args.output_dir / "gencode",
                max_duration=max_duration,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        else:
            logger.warning("Skipping GENCODE: --gencode-dir not provided")

        if args.timit_dir:
            max_duration = args.max_duration or 30
            run_timit_benchmark(
                args.timit_dir,
                args.output_dir / "timit",
                max_duration=max_duration,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        else:
            logger.warning("Skipping TIMIT: --timit-dir not provided")


if __name__ == "__main__":
    main()
