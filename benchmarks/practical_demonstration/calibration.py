#!/usr/bin/env python3
"""
Calibration Evaluation Module for Semi-CRF vs Linear CRF Uncertainty Comparison

This module provides metrics to evaluate whether Semi-CRF boundary uncertainties
are better calibrated than those derived from linear CRFs.

Key insight: Both semi-CRFs and linear CRFs can produce uncertainty estimates,
but semi-CRFs reason about segments natively while linear CRFs derive segment
information post-hoc from position-level predictions. This should lead to
better-calibrated uncertainties from semi-CRFs.

Metrics:
    1. Expected Calibration Error (ECE) - Are confidence scores reliable?
    2. Selective Prediction Curves - Can we trust high-confidence predictions?
    3. Uncertainty-Error Correlation - Does uncertainty predict mistakes?
    4. Confidence Intervals - Can we give meaningful bounds on boundary positions?

Usage:
    from calibration import (
        CalibrationEvaluator,
        compute_boundary_calibration,
        compute_selective_prediction_curve,
        plot_calibration_comparison,
    )
    
    # Evaluate semi-CRF
    evaluator = CalibrationEvaluator()
    semicrf_results = evaluator.evaluate(
        boundary_probs=semicrf_boundary_probs,  # List of (T,) arrays
        true_boundaries=true_boundary_masks,     # List of (T,) boolean arrays
        boundary_errors=semicrf_boundary_errors, # List of (T,) arrays (distance to nearest true)
    )
    
    # Compare to linear CRF
    linear_results = evaluator.evaluate(
        boundary_probs=linear_boundary_probs,
        true_boundaries=true_boundary_masks,
        boundary_errors=linear_boundary_errors,
    )
    
    print(f"Semi-CRF ECE: {semicrf_results.ece:.4f}")
    print(f"Linear CRF ECE: {linear_results.ece:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CalibrationResults:
    """Results from calibration evaluation."""
    
    # Expected Calibration Error
    ece: float
    ece_bins: np.ndarray          # Confidence bin edges
    ece_accuracies: np.ndarray    # Actual accuracy per bin
    ece_confidences: np.ndarray   # Mean confidence per bin
    ece_counts: np.ndarray        # Samples per bin
    
    # Maximum Calibration Error
    mce: float
    
    # Selective prediction (accuracy at coverage thresholds)
    coverage_thresholds: np.ndarray
    accuracy_at_coverage: np.ndarray
    
    # Uncertainty-error correlation
    uncertainty_error_correlation: float  # Spearman correlation
    uncertainty_error_pvalue: float
    
    # Confidence interval coverage (for boundary position estimation)
    ci_coverage_50: float  # Fraction of true boundaries within 50% CI
    ci_coverage_90: float  # Fraction of true boundaries within 90% CI
    ci_coverage_95: float  # Fraction of true boundaries within 95% CI
    
    # Average confidence interval widths
    ci_width_50: float
    ci_width_90: float
    ci_width_95: float
    
    # Brier score (proper scoring rule)
    brier_score: float
    
    # Area under selective prediction curve
    auc_selective: float


@dataclass 
class BoundaryUncertaintyData:
    """Container for boundary uncertainty data from a single sequence."""
    boundary_probs: np.ndarray      # (T,) probability of boundary at each position
    true_boundaries: np.ndarray     # (T,) boolean mask of true boundaries
    boundary_errors: np.ndarray     # (T,) distance to nearest true boundary (optional)
    position_probs: np.ndarray | None = None  # (T, C) position-level class probs (for linear CRF comparison)


# =============================================================================
# Core Calibration Metrics
# =============================================================================

def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error.
    
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    
    where B_b is the set of predictions in bin b.
    
    Args:
        confidences: (N,) predicted probabilities
        accuracies: (N,) binary outcomes (1 if correct, 0 if wrong)
        n_bins: Number of bins for calibration
        
    Returns:
        ece: Expected Calibration Error
        bin_edges: (n_bins+1,) bin edges
        bin_accuracies: (n_bins,) accuracy per bin
        bin_confidences: (n_bins,) mean confidence per bin
        bin_counts: (n_bins,) samples per bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()
    
    # Compute ECE
    total = confidences.shape[0]
    ece = np.sum(bin_counts / total * np.abs(bin_accuracies - bin_confidences))
    
    return ece, bin_edges, bin_accuracies, bin_confidences, bin_counts


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Maximum Calibration Error.
    
    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    _, _, bin_accuracies, bin_confidences, bin_counts = compute_ece(
        confidences, accuracies, n_bins
    )
    
    # Only consider non-empty bins
    valid = bin_counts > 0
    if not valid.any():
        return 0.0
    
    return np.max(np.abs(bin_accuracies[valid] - bin_confidences[valid]))


def compute_brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """
    Compute Brier score (proper scoring rule).
    
    Brier = mean((p - y)^2)
    
    Lower is better. Range [0, 1].
    """
    return np.mean((probabilities - outcomes) ** 2)


def compute_selective_prediction_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute selective prediction curve: accuracy vs coverage.
    
    At each coverage level (fraction of predictions we keep),
    we keep only the most confident predictions and measure accuracy.
    
    A well-calibrated model should show increasing accuracy as coverage decreases.
    
    Args:
        confidences: (N,) predicted probabilities/confidences
        correct: (N,) binary outcomes
        n_points: Number of points on the curve
        
    Returns:
        coverage: (n_points,) coverage levels from 1.0 to ~0.0
        accuracy: (n_points,) accuracy at each coverage level
    """
    # Sort by confidence (descending)
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    
    # Compute cumulative accuracy at each coverage level
    n = len(confidences)
    coverage = np.linspace(1.0, 0.01, n_points)
    accuracy = np.zeros(n_points)
    
    for i, cov in enumerate(coverage):
        k = max(1, int(cov * n))
        accuracy[i] = sorted_correct[:k].mean()
    
    return coverage, accuracy


def compute_auc_selective(coverage: np.ndarray, accuracy: np.ndarray) -> float:
    """Compute area under selective prediction curve using trapezoidal rule."""
    return np.trapz(accuracy, coverage)


def compute_uncertainty_error_correlation(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Spearman correlation between uncertainty and error.
    
    We expect: higher uncertainty → larger error (positive correlation).
    A good uncertainty estimate should correlate with actual mistakes.
    
    Args:
        uncertainties: (N,) uncertainty values (e.g., 1 - confidence or entropy)
        errors: (N,) error magnitudes (e.g., distance to nearest true boundary)
        
    Returns:
        correlation: Spearman correlation coefficient
        pvalue: p-value for the correlation
    """
    from scipy import stats
    
    # Remove NaN/inf
    valid = np.isfinite(uncertainties) & np.isfinite(errors)
    if valid.sum() < 10:
        return 0.0, 1.0
    
    correlation, pvalue = stats.spearmanr(uncertainties[valid], errors[valid])
    return float(correlation), float(pvalue)


# =============================================================================
# Confidence Interval Estimation for Boundary Position
# =============================================================================

def estimate_boundary_confidence_interval(
    boundary_probs: np.ndarray,
    center_idx: int,
    confidence_level: float = 0.95,
    max_width: int = 50,
) -> tuple[int, int, float]:
    """
    Estimate confidence interval for boundary position.
    
    Given boundary probabilities and a predicted boundary location,
    find the smallest interval [low, high] such that the total probability
    mass within the interval exceeds the confidence level.
    
    This is a key advantage of semi-CRFs: they can naturally express
    "the boundary is at position 100 ± 5bp with 95% confidence".
    
    Args:
        boundary_probs: (T,) boundary probabilities
        center_idx: Index of predicted boundary (argmax)
        confidence_level: Desired confidence level (e.g., 0.95)
        max_width: Maximum interval half-width to consider
        
    Returns:
        low: Lower bound of interval
        high: Upper bound of interval
        actual_coverage: Actual probability mass in interval
    """
    T = len(boundary_probs)
    
    # Start from center and expand until we reach desired coverage
    for half_width in range(max_width + 1):
        low = max(0, center_idx - half_width)
        high = min(T - 1, center_idx + half_width)
        
        # Sum probability mass in interval
        prob_mass = boundary_probs[low:high + 1].sum()
        
        if prob_mass >= confidence_level:
            return low, high, prob_mass
    
    # If we couldn't reach the confidence level, return max interval
    low = max(0, center_idx - max_width)
    high = min(T - 1, center_idx + max_width)
    return low, high, boundary_probs[low:high + 1].sum()


def evaluate_confidence_intervals(
    boundary_probs_list: Sequence[np.ndarray],
    true_boundaries_list: Sequence[np.ndarray],
    confidence_levels: Sequence[float] = (0.50, 0.90, 0.95),
) -> dict[float, tuple[float, float]]:
    """
    Evaluate confidence interval coverage and width across all sequences.
    
    For each true boundary, find the nearest predicted boundary peak and
    compute whether the true boundary falls within the CI.
    
    Args:
        boundary_probs_list: List of (T,) boundary probability arrays
        true_boundaries_list: List of (T,) boolean arrays
        confidence_levels: Confidence levels to evaluate
        
    Returns:
        Dict mapping confidence level to (coverage, avg_width)
        coverage = fraction of true boundaries within CI
        avg_width = average CI width in positions
    """
    results = {level: {"covered": 0, "total": 0, "widths": []} for level in confidence_levels}
    
    for boundary_probs, true_boundaries in zip(boundary_probs_list, true_boundaries_list):
        true_indices = np.where(true_boundaries)[0]
        
        if len(true_indices) == 0:
            continue
        
        # Find predicted boundary peaks (local maxima above threshold)
        threshold = 0.1
        peaks = find_boundary_peaks(boundary_probs, threshold=threshold)
        
        if len(peaks) == 0:
            continue
        
        for true_idx in true_indices:
            # Find nearest predicted peak
            distances = np.abs(peaks - true_idx)
            nearest_peak = peaks[np.argmin(distances)]
            
            # Compute CI at each confidence level
            for level in confidence_levels:
                low, high, _ = estimate_boundary_confidence_interval(
                    boundary_probs, nearest_peak, confidence_level=level
                )
                
                # Check if true boundary is within CI
                covered = low <= true_idx <= high
                width = high - low + 1
                
                results[level]["covered"] += int(covered)
                results[level]["total"] += 1
                results[level]["widths"].append(width)
    
    # Compute final metrics
    final = {}
    for level in confidence_levels:
        if results[level]["total"] > 0:
            coverage = results[level]["covered"] / results[level]["total"]
            avg_width = np.mean(results[level]["widths"])
        else:
            coverage = 0.0
            avg_width = 0.0
        final[level] = (coverage, avg_width)
    
    return final


def find_boundary_peaks(
    boundary_probs: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 3,
) -> np.ndarray:
    """Find local maxima in boundary probability array."""
    peaks = []
    T = len(boundary_probs)
    
    for i in range(1, T - 1):
        if boundary_probs[i] < threshold:
            continue
        
        # Check if local maximum
        if boundary_probs[i] > boundary_probs[i - 1] and boundary_probs[i] >= boundary_probs[i + 1]:
            # Check minimum distance from previous peak
            if len(peaks) == 0 or i - peaks[-1] >= min_distance:
                peaks.append(i)
    
    return np.array(peaks)


# =============================================================================
# Linear CRF Baseline: Deriving Boundary Uncertainty from Position Marginals
# =============================================================================

def derive_boundary_probs_from_positions(
    position_probs: np.ndarray,
    method: str = "transition",
) -> np.ndarray:
    """
    Derive boundary probabilities from position-level class probabilities.
    
    This is what a linear CRF has to do - it doesn't have native segment-level
    uncertainty, so we derive it from position-level predictions.
    
    Args:
        position_probs: (T, C) position-level class probabilities
        method: How to derive boundary probability
            - "transition": P(boundary at t) = 1 - sum_c P(y_{t-1}=c) * P(y_t=c)
            - "entropy": Use entropy of position predictions
            - "max_diff": |max(P(y_t)) - max(P(y_{t-1}))|
            
    Returns:
        boundary_probs: (T,) derived boundary probabilities
    """
    T, C = position_probs.shape
    boundary_probs = np.zeros(T)
    
    if method == "transition":
        # P(boundary) ≈ 1 - P(same class as previous position)
        # This assumes independence, which is wrong but is what we have
        for t in range(1, T):
            # Probability of staying in same class
            p_same = np.sum(position_probs[t - 1] * position_probs[t])
            boundary_probs[t] = 1 - p_same
            
    elif method == "entropy":
        # High entropy = uncertainty = potential boundary
        for t in range(T):
            p = position_probs[t]
            p = p[p > 0]  # Avoid log(0)
            entropy = -np.sum(p * np.log(p + 1e-10))
            boundary_probs[t] = entropy / np.log(C)  # Normalize to [0, 1]
            
    elif method == "max_diff":
        # Change in most likely class
        for t in range(1, T):
            boundary_probs[t] = np.abs(
                position_probs[t].max() - position_probs[t - 1].max()
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return boundary_probs


# =============================================================================
# Main Evaluator Class
# =============================================================================

class CalibrationEvaluator:
    """
    Evaluate calibration of boundary uncertainty estimates.
    
    Compares semi-CRF native uncertainty vs linear CRF derived uncertainty.
    """
    
    def __init__(self, n_bins: int = 15, n_curve_points: int = 100):
        self.n_bins = n_bins
        self.n_curve_points = n_curve_points
    
    def evaluate(
        self,
        boundary_probs: Sequence[np.ndarray],
        true_boundaries: Sequence[np.ndarray],
        boundary_errors: Sequence[np.ndarray] | None = None,
    ) -> CalibrationResults:
        """
        Evaluate calibration metrics.
        
        Args:
            boundary_probs: List of (T,) arrays with boundary probabilities
            true_boundaries: List of (T,) boolean arrays with true boundary locations
            boundary_errors: Optional list of (T,) arrays with distance to nearest
                           true boundary at each position
                           
        Returns:
            CalibrationResults with all metrics
        """
        # Flatten all predictions
        all_probs = np.concatenate(boundary_probs)
        all_true = np.concatenate(true_boundaries).astype(float)
        
        # Compute ECE
        ece, bins, bin_acc, bin_conf, bin_counts = compute_ece(
            all_probs, all_true, self.n_bins
        )
        
        # Compute MCE
        mce = compute_mce(all_probs, all_true, self.n_bins)
        
        # Compute Brier score
        brier = compute_brier_score(all_probs, all_true)
        
        # Compute selective prediction curve
        coverage, accuracy = compute_selective_prediction_curve(
            all_probs, all_true, self.n_curve_points
        )
        auc = compute_auc_selective(coverage, accuracy)
        
        # Compute uncertainty-error correlation if errors provided
        if boundary_errors is not None:
            all_errors = np.concatenate(boundary_errors)
            uncertainties = 1 - all_probs  # High prob = low uncertainty
            corr, pval = compute_uncertainty_error_correlation(uncertainties, all_errors)
        else:
            corr, pval = 0.0, 1.0
        
        # Compute confidence interval metrics
        ci_results = evaluate_confidence_intervals(
            boundary_probs, true_boundaries, [0.50, 0.90, 0.95]
        )
        
        return CalibrationResults(
            ece=ece,
            ece_bins=bins,
            ece_accuracies=bin_acc,
            ece_confidences=bin_conf,
            ece_counts=bin_counts,
            mce=mce,
            coverage_thresholds=coverage,
            accuracy_at_coverage=accuracy,
            uncertainty_error_correlation=corr,
            uncertainty_error_pvalue=pval,
            ci_coverage_50=ci_results[0.50][0],
            ci_coverage_90=ci_results[0.90][0],
            ci_coverage_95=ci_results[0.95][0],
            ci_width_50=ci_results[0.50][1],
            ci_width_90=ci_results[0.90][1],
            ci_width_95=ci_results[0.95][1],
            brier_score=brier,
            auc_selective=auc,
        )


# =============================================================================
# Visualization
# =============================================================================

def plot_calibration_comparison(
    semicrf_results: CalibrationResults,
    linear_results: CalibrationResults,
    output_path: str | None = None,
):
    """
    Plot calibration comparison between semi-CRF and linear CRF.
    
    Creates a figure with:
    1. Reliability diagrams (calibration curves)
    2. Selective prediction curves
    3. Confidence interval coverage comparison
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Reliability diagram
    ax = axes[0]
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
    
    # Semi-CRF
    valid = semicrf_results.ece_counts > 0
    ax.plot(
        semicrf_results.ece_confidences[valid],
        semicrf_results.ece_accuracies[valid],
        'o-', color='#DC267F', label=f'Semi-CRF (ECE={semicrf_results.ece:.3f})',
        markersize=6
    )
    
    # Linear CRF
    valid = linear_results.ece_counts > 0
    ax.plot(
        linear_results.ece_confidences[valid],
        linear_results.ece_accuracies[valid],
        's-', color='#648FFF', label=f'Linear CRF (ECE={linear_results.ece:.3f})',
        markersize=6
    )
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 2. Selective prediction curve
    ax = axes[1]
    
    ax.plot(
        semicrf_results.coverage_thresholds,
        semicrf_results.accuracy_at_coverage,
        '-', color='#DC267F', label=f'Semi-CRF (AUC={semicrf_results.auc_selective:.3f})',
        linewidth=2
    )
    ax.plot(
        linear_results.coverage_thresholds,
        linear_results.accuracy_at_coverage,
        '-', color='#648FFF', label=f'Linear CRF (AUC={linear_results.auc_selective:.3f})',
        linewidth=2
    )
    
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.set_title('Selective Prediction')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence interval coverage
    ax = axes[2]
    
    ci_levels = [50, 90, 95]
    x = np.arange(len(ci_levels))
    width = 0.35
    
    semicrf_coverage = [
        semicrf_results.ci_coverage_50,
        semicrf_results.ci_coverage_90,
        semicrf_results.ci_coverage_95,
    ]
    linear_coverage = [
        linear_results.ci_coverage_50,
        linear_results.ci_coverage_90,
        linear_results.ci_coverage_95,
    ]
    
    bars1 = ax.bar(x - width/2, semicrf_coverage, width, label='Semi-CRF', color='#DC267F')
    bars2 = ax.bar(x + width/2, linear_coverage, width, label='Linear CRF', color='#648FFF')
    
    # Add expected coverage lines
    for i, level in enumerate(ci_levels):
        ax.hlines(level/100, i - 0.5, i + 0.5, colors='black', linestyles='--', alpha=0.5)
    
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('CI Coverage (dashed = expected)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}%' for l in ci_levels])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_examples(
    sequence_idx: int,
    semicrf_probs: np.ndarray,
    linear_probs: np.ndarray,
    true_boundaries: np.ndarray,
    output_path: str | None = None,
):
    """
    Plot example showing boundary uncertainty from semi-CRF vs linear CRF.
    
    This visualization shows why semi-CRF uncertainty is more interpretable:
    it gives you actual confidence intervals around boundary positions.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    T = len(semicrf_probs)
    x = np.arange(T)
    
    # Semi-CRF
    ax = axes[0]
    ax.fill_between(x, semicrf_probs, alpha=0.3, color='#DC267F')
    ax.plot(x, semicrf_probs, color='#DC267F', linewidth=1.5, label='Semi-CRF P(boundary)')
    
    # Mark true boundaries
    true_idx = np.where(true_boundaries)[0]
    ax.vlines(true_idx, 0, 1, colors='black', linestyles='--', alpha=0.7, label='True boundaries')
    
    ax.set_ylabel('P(boundary)')
    ax.set_title('Semi-CRF: Native Segment-Level Uncertainty')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Linear CRF
    ax = axes[1]
    ax.fill_between(x, linear_probs, alpha=0.3, color='#648FFF')
    ax.plot(x, linear_probs, color='#648FFF', linewidth=1.5, label='Linear CRF P(boundary)')
    ax.vlines(true_idx, 0, 1, colors='black', linestyles='--', alpha=0.7, label='True boundaries')
    
    ax.set_xlabel('Position')
    ax.set_ylabel('P(boundary)')
    ax.set_title('Linear CRF: Derived from Position Marginals')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Summary Printing
# =============================================================================

def print_calibration_comparison(
    semicrf_results: CalibrationResults,
    linear_results: CalibrationResults,
    model_names: tuple[str, str] = ("Semi-CRF", "Linear CRF"),
):
    """Print formatted comparison of calibration results."""
    
    print("\n" + "=" * 70)
    print("UNCERTAINTY CALIBRATION COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<40} {model_names[0]:>12} {model_names[1]:>12}     Δ")
    print("-" * 70)
    
    # ECE (lower is better)
    delta = semicrf_results.ece - linear_results.ece
    better = "✓" if delta < 0 else ""
    print(f"{'Expected Calibration Error (↓)':<40} {semicrf_results.ece:>12.4f} {linear_results.ece:>12.4f} {delta:>+8.4f} {better}")
    
    # MCE (lower is better)
    delta = semicrf_results.mce - linear_results.mce
    better = "✓" if delta < 0 else ""
    print(f"{'Maximum Calibration Error (↓)':<40} {semicrf_results.mce:>12.4f} {linear_results.mce:>12.4f} {delta:>+8.4f} {better}")
    
    # Brier score (lower is better)
    delta = semicrf_results.brier_score - linear_results.brier_score
    better = "✓" if delta < 0 else ""
    print(f"{'Brier Score (↓)':<40} {semicrf_results.brier_score:>12.4f} {linear_results.brier_score:>12.4f} {delta:>+8.4f} {better}")
    
    # AUC selective (higher is better)
    delta = semicrf_results.auc_selective - linear_results.auc_selective
    better = "✓" if delta > 0 else ""
    print(f"{'Selective Prediction AUC (↑)':<40} {semicrf_results.auc_selective:>12.4f} {linear_results.auc_selective:>12.4f} {delta:>+8.4f} {better}")
    
    # Uncertainty-error correlation (higher is better - uncertainty should predict errors)
    delta = semicrf_results.uncertainty_error_correlation - linear_results.uncertainty_error_correlation
    better = "✓" if delta > 0 else ""
    print(f"{'Uncertainty-Error Correlation (↑)':<40} {semicrf_results.uncertainty_error_correlation:>12.4f} {linear_results.uncertainty_error_correlation:>12.4f} {delta:>+8.4f} {better}")
    
    print("\nConfidence Interval Coverage (closer to nominal is better):")
    print("-" * 70)
    
    for level, (s_cov, s_width, l_cov, l_width) in [
        (50, (semicrf_results.ci_coverage_50, semicrf_results.ci_width_50,
              linear_results.ci_coverage_50, linear_results.ci_width_50)),
        (90, (semicrf_results.ci_coverage_90, semicrf_results.ci_width_90,
              linear_results.ci_coverage_90, linear_results.ci_width_90)),
        (95, (semicrf_results.ci_coverage_95, semicrf_results.ci_width_95,
              linear_results.ci_coverage_95, linear_results.ci_width_95)),
    ]:
        nominal = level / 100
        s_err = abs(s_cov - nominal)
        l_err = abs(l_cov - nominal)
        better = "✓" if s_err < l_err else ""
        print(f"  {level}% CI coverage (nominal={nominal:.2f}): "
              f"{model_names[0]}={s_cov:.3f} (width={s_width:.1f}), "
              f"{model_names[1]}={l_cov:.3f} (width={l_width:.1f}) {better}")
    
    print("\n" + "=" * 70)


# =============================================================================
# Utility: Compute boundary errors (distance to nearest true boundary)
# =============================================================================

def compute_boundary_errors(
    true_boundaries: np.ndarray,
) -> np.ndarray:
    """
    Compute distance to nearest true boundary for each position.
    
    This is used to correlate uncertainty with actual errors.
    """
    T = len(true_boundaries)
    true_idx = np.where(true_boundaries)[0]
    
    if len(true_idx) == 0:
        return np.full(T, T)  # Max distance
    
    errors = np.zeros(T)
    for i in range(T):
        errors[i] = np.min(np.abs(true_idx - i))
    
    return errors
