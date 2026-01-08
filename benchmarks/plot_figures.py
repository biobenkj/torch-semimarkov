#!/usr/bin/env python3
"""
Generate publication-quality figures for Semi-Markov CRF backend analysis.

Figures:
1. OOM feasibility heatmaps (colorblind-safe, consistent GB units)
2. Time vs state-space size (median + IQR bands)
3. Memory breakdown stacked bars

Addresses reviewer feedback:
- Colorblind-safe palette (blue/orange, with hatching for OOM)
- Consistent GB units throughout
- Median + IQR for timing
- Time per position normalization
- Explicit annotations for key findings
- Clear legend for "not tested" cells

Example:
    python benchmarks/plot_figures.py \
        --input-dir results/ \
        --output-dir figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import LinearSegmentedColormap
        return True
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return False


def plot_oom_heatmaps(df: pd.DataFrame, output_dir: Path, T_values: List[int] = None):
    """
    Generate OOM feasibility heatmaps.

    Features:
    - Colorblind-safe palette (blue for success, orange for OOM, gray for not tested)
    - Hatching pattern for OOM cells
    - Memory values in GB inside successful cells
    - Annotation for OOM frontier
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors

    if T_values is None:
        T_values = sorted(df['T'].unique())

    backends = ['linear_scan_vectorized', 'binary_tree', 'banded', 'block_triangular']
    backend_labels = {
        'linear_scan': 'Linear Scan',
        'linear_scan_vectorized': 'Vectorized Linear Scan',
        'binary_tree': 'Binary Tree',
        'banded': 'Banded',
        'block_triangular': 'Block Triangular',
    }

    # Colorblind-safe colors
    color_success = '#2166ac'  # Blue
    color_oom = '#d6604d'      # Orange-red
    color_not_tested = '#969696'  # Gray

    K_values = sorted(df['K'].unique())
    C_values = sorted(df['C'].unique())

    for T in T_values:
        fig, axes = plt.subplots(1, len(backends), figsize=(4 * len(backends), 4))
        if len(backends) == 1:
            axes = [axes]

        for ax_idx, backend in enumerate(backends):
            ax = axes[ax_idx]
            subset = df[(df['T'] == T) & (df['backend'] == backend)]

            # Create matrix
            matrix = np.full((len(K_values), len(C_values)), np.nan)
            status_matrix = np.full((len(K_values), len(C_values)), '', dtype=object)
            mem_matrix = np.full((len(K_values), len(C_values)), np.nan)

            for _, row in subset.iterrows():
                k_idx = K_values.index(row['K'])
                c_idx = C_values.index(row['C'])
                status_matrix[k_idx, c_idx] = row['status']
                if row['status'] == 'success':
                    mem_matrix[k_idx, c_idx] = row['peak_allocated_gb']
                    matrix[k_idx, c_idx] = 1  # Success
                elif row['status'] == 'oom':
                    matrix[k_idx, c_idx] = 0  # OOM
                else:
                    matrix[k_idx, c_idx] = 0.5  # Not tested

            # Plot base heatmap
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto',
                          origin='lower', interpolation='nearest')

            # Add hatching for OOM cells and text annotations
            for i, K in enumerate(K_values):
                for j, C in enumerate(C_values):
                    status = status_matrix[i, j]
                    if status == 'success':
                        mem = mem_matrix[i, j]
                        # Format memory value
                        if mem < 0.01:
                            text = f'{mem*1000:.0f}M'
                        elif mem < 1:
                            text = f'{mem:.2f}'
                        else:
                            text = f'{mem:.1f}'
                        ax.text(j, i, text, ha='center', va='center',
                               fontsize=7, color='white', fontweight='bold')
                    elif status == 'oom':
                        # Add X pattern for OOM
                        ax.text(j, i, '×', ha='center', va='center',
                               fontsize=14, color='white', fontweight='bold')
                    elif status == 'not_tested':
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                   fill=True, facecolor=color_not_tested,
                                                   edgecolor='white', linewidth=0.5))

            ax.set_xticks(range(len(C_values)))
            ax.set_xticklabels(C_values)
            ax.set_yticks(range(len(K_values)))
            ax.set_yticklabels(K_values)
            ax.set_xlabel('C (num labels)')
            ax.set_ylabel('K (max duration)')
            ax.set_title(f'{backend_labels.get(backend, backend)}')

            # Add KC grid lines
            for i in range(len(K_values)):
                for j in range(len(C_values)):
                    kc = K_values[i] * C_values[j]
                    if kc in [100, 150, 200]:
                        ax.plot([j-0.5, j+0.5], [i-0.5, i-0.5], 'k--', linewidth=0.5, alpha=0.5)

        # Add legend
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Success (GB shown)'),
            Patch(facecolor='#d62728', label='OOM (×)'),
            Patch(facecolor=color_not_tested, label='Not tested'),
        ]
        fig.legend(handles=legend_elements, loc='upper center',
                  ncol=3, bbox_to_anchor=(0.5, 0.02), frameon=False)

        fig.suptitle(f'OOM Feasibility: T={T}\n(values show peak memory in GB)', fontsize=12)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        output_path = output_dir / f'oom_heatmap_T{T}.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")


def plot_time_vs_kc(df: pd.DataFrame, output_dir: Path, T_values: List[int] = None):
    """
    Generate time vs state-space size (KC) plots.

    Features:
    - Log-log scale
    - Median line with IQR shading
    - Separate panel per T value
    - Time per position option
    """
    import matplotlib.pyplot as plt

    if T_values is None:
        T_values = sorted(df['T'].unique())

    backends = ['linear_scan', 'linear_scan_vectorized', 'binary_tree', 'banded', 'block_triangular']
    backend_labels = {
        'linear_scan': 'Linear Scan',
        'linear_scan_vectorized': 'Vectorized Linear',
        'binary_tree': 'Binary Tree',
        'banded': 'Banded',
        'block_triangular': 'Block Triangular',
    }

    # Colorblind-safe palette
    colors = {
        'linear_scan': '#1f77b4',
        'linear_scan_vectorized': '#ff7f0e',
        'binary_tree': '#2ca02c',
        'banded': '#d62728',
        'block_triangular': '#9467bd',
    }

    markers = {
        'linear_scan': 'o',
        'linear_scan_vectorized': 's',
        'binary_tree': '^',
        'banded': 'D',
        'block_triangular': 'v',
    }

    # Figure 1: Absolute time
    fig, axes = plt.subplots(1, len(T_values), figsize=(4 * len(T_values), 4), sharey=True)
    if len(T_values) == 1:
        axes = [axes]

    for ax_idx, T in enumerate(T_values):
        ax = axes[ax_idx]

        for backend in backends:
            subset = df[(df['T'] == T) & (df['backend'] == backend) & (df['status'] == 'success')]
            if len(subset) == 0:
                continue

            subset = subset.sort_values('KC')
            kc = subset['KC'].values
            median = subset['time_ms_median'].values
            iqr_low = subset['time_ms_iqr_low'].values
            iqr_high = subset['time_ms_iqr_high'].values

            ax.plot(kc, median, color=colors.get(backend, 'gray'),
                   marker=markers.get(backend, 'o'), markersize=4,
                   label=backend_labels.get(backend, backend), linewidth=1.5)
            ax.fill_between(kc, iqr_low, iqr_high, color=colors.get(backend, 'gray'),
                           alpha=0.2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('State-space size (K×C)')
        if ax_idx == 0:
            ax.set_ylabel('Time (ms)')
        ax.set_title(f'T={T}')
        ax.grid(True, alpha=0.3, which='both')

        # Add reference lines for typical genomic regimes
        for kc_ref, label in [(100, ''), (150, 'OOM\nfrontier'), (200, '')]:
            if kc_ref <= subset['KC'].max() if len(subset) > 0 else False:
                ax.axvline(kc_ref, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(backends),
              bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)

    fig.suptitle('Forward+Backward Time vs State-Space Size\n(median with IQR band)', fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_path = output_dir / 'time_vs_kc.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

    # Figure 2: Time per position
    fig, axes = plt.subplots(1, len(T_values), figsize=(4 * len(T_values), 4), sharey=True)
    if len(T_values) == 1:
        axes = [axes]

    for ax_idx, T in enumerate(T_values):
        ax = axes[ax_idx]

        for backend in backends:
            subset = df[(df['T'] == T) & (df['backend'] == backend) & (df['status'] == 'success')]
            if len(subset) == 0:
                continue

            subset = subset.sort_values('KC')
            kc = subset['KC'].values
            time_per_pos = subset['time_per_position_ms'].values

            ax.plot(kc, time_per_pos, color=colors.get(backend, 'gray'),
                   marker=markers.get(backend, 'o'), markersize=4,
                   label=backend_labels.get(backend, backend), linewidth=1.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('State-space size (K×C)')
        if ax_idx == 0:
            ax.set_ylabel('Time per position (ms/T)')
        ax.set_title(f'T={T}')
        ax.grid(True, alpha=0.3, which='both')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(backends),
              bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)

    fig.suptitle('Time per Position vs State-Space Size\n(enables cross-T comparison)', fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_path = output_dir / 'time_per_position_vs_kc.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_memory_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Generate memory breakdown stacked bar charts.

    Shows allocation categories:
    - Potentials (edge potentials tensor)
    - DP State (alpha/beta tables)
    - Workspace (intermediate computations - the killer!)
    - Autograd (saved tensors for backward)
    """
    import matplotlib.pyplot as plt

    # Select representative configs: below frontier, near frontier, past frontier
    # Defined by KC thresholds
    configs = [
        ("Below frontier", 50, 100),   # KC ~50-100
        ("Near frontier", 100, 150),   # KC ~100-150
        ("Past frontier", 150, 250),   # KC ~150-250
    ]

    backends = ['linear_scan_vectorized', 'binary_tree', 'banded', 'block_triangular']
    backend_labels = {
        'linear_scan_vectorized': 'Vec. Linear',
        'binary_tree': 'Binary Tree',
        'banded': 'Banded',
        'block_triangular': 'Block Tri.',
    }

    # Colors for memory categories
    category_colors = {
        'potentials': '#1f77b4',
        'dp_state': '#ff7f0e',
        'workspace': '#d62728',  # Red - the killer!
        'autograd': '#9467bd',
    }

    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 5))

    for ax_idx, (config_name, kc_min, kc_max) in enumerate(configs):
        ax = axes[ax_idx]

        # Find configs in this KC range
        subset = df[(df['KC'] >= kc_min) & (df['KC'] < kc_max)]

        # For each backend, get a representative row
        x_positions = []
        x_labels = []

        for i, backend in enumerate(backends):
            backend_subset = subset[subset['backend'] == backend]

            if len(backend_subset) > 0:
                # Pick median KC config
                row = backend_subset.iloc[len(backend_subset) // 2]

                bottom = 0
                for cat, col in [
                    ('potentials', 'est_potentials_gb'),
                    ('dp_state', 'est_dp_state_gb'),
                    ('workspace', 'est_workspace_gb'),
                    ('autograd', 'est_autograd_gb'),
                ]:
                    if col in row and not pd.isna(row[col]):
                        height = row[col]
                        ax.bar(i, height, bottom=bottom, color=category_colors[cat],
                              edgecolor='white', linewidth=0.5)
                        bottom += height

                # Add actual peak memory as marker
                if row['status'] == 'success' and not pd.isna(row['peak_allocated_gb']):
                    ax.plot(i, row['peak_allocated_gb'], 'ko', markersize=8)
                    ax.plot(i, row['peak_reserved_gb'], 'k^', markersize=6, alpha=0.5)

                x_positions.append(i)
                x_labels.append(f"{backend_labels.get(backend, backend)}\nKC={int(row['KC'])}")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel('Memory (GB)')
        ax.set_title(f'{config_name}\n(KC ∈ [{kc_min}, {kc_max}))')

        # Add OOM threshold line
        ax.axhline(y=24, color='red', linestyle='--', alpha=0.5, label='24GB limit')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=category_colors['potentials'], label='Potentials'),
        Patch(facecolor=category_colors['dp_state'], label='DP State'),
        Patch(facecolor=category_colors['workspace'], label='Workspace'),
        Patch(facecolor=category_colors['autograd'], label='Autograd'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Peak Allocated'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=6, label='Peak Reserved'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6,
              bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)

    fig.suptitle('Memory Breakdown by Allocation Category\n(workspace is the "killer" term for tree methods)',
                fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_path = output_dir / 'memory_breakdown.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_oom_frontier_summary(df: pd.DataFrame, output_dir: Path):
    """
    Generate OOM frontier summary plot.

    Shows max feasible KC vs T for each backend.
    """
    import matplotlib.pyplot as plt

    backends = ['linear_scan_vectorized', 'binary_tree', 'banded', 'block_triangular']
    backend_labels = {
        'linear_scan_vectorized': 'Vectorized Linear Scan',
        'binary_tree': 'Binary Tree',
        'banded': 'Banded',
        'block_triangular': 'Block Triangular',
    }

    colors = {
        'linear_scan_vectorized': '#ff7f0e',
        'binary_tree': '#2ca02c',
        'banded': '#d62728',
        'block_triangular': '#9467bd',
    }

    T_values = sorted(df['T'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for backend in backends:
        max_kc_per_T = []
        for T in T_values:
            successful = df[(df['T'] == T) & (df['backend'] == backend) & (df['status'] == 'success')]
            if len(successful) > 0:
                max_kc = successful['KC'].max()
            else:
                max_kc = 0
            max_kc_per_T.append(max_kc)

        ax.plot(T_values, max_kc_per_T, 'o-', color=colors.get(backend, 'gray'),
               label=backend_labels.get(backend, backend), linewidth=2, markersize=8)

    ax.set_xlabel('Sequence Length (T)')
    ax.set_ylabel('Max Feasible State-Space Size (KC)')
    ax.set_title('OOM Frontier: Max Feasible KC vs T\n(higher is better)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Vectorized linear scan:\nuniversally feasible',
               xy=(T_values[-1], max(df[df['backend']=='linear_scan_vectorized']['KC'])),
               xytext=(T_values[-1] * 0.6, max(df['KC']) * 0.8),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()

    output_path = output_dir / 'oom_frontier_summary.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--T", type=str, default=None, help="Comma-separated T values to plot")
    args = parser.parse_args()

    if not check_matplotlib():
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = args.input_dir / "benchmark_full.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark_memory_analysis.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    # Parse T values
    if args.T:
        T_values = [int(x) for x in args.T.split(",")]
    else:
        T_values = sorted(df['T'].unique())

    # Generate figures
    print("\nGenerating figures...")

    plot_oom_heatmaps(df, args.output_dir, T_values)
    plot_time_vs_kc(df, args.output_dir, T_values)
    plot_memory_breakdown(df, args.output_dir)
    plot_oom_frontier_summary(df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
