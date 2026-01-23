#!/usr/bin/env python3
"""
Integration Guide: Adding Calibration Evaluation to Task Benchmarks

This file shows how to integrate the calibration module with the existing
Gencode and TIMIT benchmarks. These are code snippets to be added to the
respective benchmark files, not a standalone script.

The key insight is that we need to:
1. Get boundary probabilities from the semi-CRF (native)
2. Derive boundary probabilities from the linear CRF (from position marginals)
3. Compare calibration metrics between them

IMPORTANT: MODEL REQUIREMENTS
=============================
To use calibration evaluation, the Semi-CRF model MUST use
UncertaintySemiMarkovCRFHead instead of SemiMarkovCRFHead:

    from torch_semimarkov import UncertaintySemiMarkovCRFHead

    self.crf = UncertaintySemiMarkovCRFHead(
        num_classes=num_classes,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
    )

UncertaintySemiMarkovCRFHead provides these methods required for calibration:
- compute_boundary_marginals(hidden, lengths) -> (batch, T) boundary probs
- compute_position_marginals(hidden, lengths) -> (batch, T, C) position probs

The linear CRF baseline (K=1) can use either class - boundary probabilities
are derived from position marginals using derive_boundary_probs_from_positions().
"""

import numpy as np
import torch
from torch import Tensor


# =============================================================================
# INTEGRATION FOR GENCODE BENCHMARK (gencode_exon_intron.py)
# =============================================================================

# Add these imports at the top of gencode_exon_intron.py:
# from calibration import (
#     CalibrationEvaluator,
#     derive_boundary_probs_from_positions,
#     compute_boundary_errors,
#     print_calibration_comparison,
#     plot_calibration_comparison,
# )


def evaluate_with_calibration_gencode(
    model,  # ExonIntronModel
    dataloader,
    device: torch.device,
    is_semicrf: bool = True,
):
    """
    Extended evaluation that includes calibration metrics.
    
    Add this function to gencode_exon_intron.py and call it after training
    to get calibration metrics alongside the standard metrics.
    """
    from calibration import (
        CalibrationEvaluator,
        derive_boundary_probs_from_positions,
        compute_boundary_errors,
    )
    
    model.eval()
    
    # Standard metrics collection (same as before)
    all_predictions = []
    all_targets = []
    all_pred_segments = []
    all_true_segments = []
    
    # Calibration data collection
    all_boundary_probs = []
    all_true_boundaries = []
    all_boundary_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            
            # Get hidden states from encoder
            hidden = model.encoder(sequence)
            
            # Get boundary probabilities
            if is_semicrf and hasattr(model.crf, 'compute_boundary_marginals'):
                # Native semi-CRF boundary probabilities
                # This uses UncertaintySemiMarkovCRFHead
                boundary_probs = model.crf.compute_boundary_marginals(hidden, lengths)
            else:
                # Linear CRF: derive from position marginals
                position_probs = model.crf.compute_position_marginals(hidden, lengths)
                boundary_probs = []
                for i in range(len(lengths)):
                    seq_len = lengths[i].item()
                    pos_p = position_probs[i, :seq_len].cpu().numpy()
                    bp = derive_boundary_probs_from_positions(pos_p, method="transition")
                    boundary_probs.append(bp)
                boundary_probs = boundary_probs  # List of arrays
            
            # Decode for standard metrics
            result = model.decode(sequence, lengths)
            
            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                
                # Standard metrics
                pred_labels = np.zeros(seq_len, dtype=np.int64)
                for seg in result.segments[i]:
                    pred_labels[seg.start:seg.end+1] = seg.label
                
                true_labels = labels[i, :seq_len].cpu().numpy()
                
                all_predictions.append(pred_labels)
                all_targets.append(true_labels)
                
                # Calibration metrics
                # True boundaries: where labels change
                true_bounds = np.zeros(seq_len, dtype=bool)
                for j in range(1, seq_len):
                    if true_labels[j] != true_labels[j-1]:
                        true_bounds[j] = True
                
                if isinstance(boundary_probs, torch.Tensor):
                    bp = boundary_probs[i, :seq_len].cpu().numpy()
                else:
                    bp = boundary_probs[i]
                
                all_boundary_probs.append(bp)
                all_true_boundaries.append(true_bounds)
                all_boundary_errors.append(compute_boundary_errors(true_bounds))
    
    # Compute calibration metrics
    evaluator = CalibrationEvaluator()
    calibration_results = evaluator.evaluate(
        boundary_probs=all_boundary_probs,
        true_boundaries=all_true_boundaries,
        boundary_errors=all_boundary_errors,
    )
    
    return calibration_results


def compare_calibration_gencode(
    data_dir,
    max_duration: int = 500,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Compare calibration between linear CRF and semi-CRF on Gencode benchmark.
    
    Call this after training both models.
    """
    from calibration import print_calibration_comparison, plot_calibration_comparison
    from torch.utils.data import DataLoader
    from pathlib import Path
    
    # This assumes you've already trained both models
    # In practice, you'd integrate this into the compare_models function
    
    device = torch.device(device)
    
    # Load test data
    test_dataset = GencodeDataset(Path(data_dir) / "test.jsonl")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Evaluate linear CRF (K=1)
    # linear_model = ... (your trained linear model)
    # linear_calibration = evaluate_with_calibration_gencode(
    #     linear_model, test_loader, device, is_semicrf=False
    # )
    
    # Evaluate semi-CRF (K=500)
    # semicrf_model = ... (your trained semi-CRF model)
    # semicrf_calibration = evaluate_with_calibration_gencode(
    #     semicrf_model, test_loader, device, is_semicrf=True
    # )
    
    # Print comparison
    # print_calibration_comparison(semicrf_calibration, linear_calibration)
    
    # Plot comparison
    # plot_calibration_comparison(
    #     semicrf_calibration, linear_calibration,
    #     output_path="calibration_comparison_gencode.pdf"
    # )
    
    pass  # Placeholder


# =============================================================================
# INTEGRATION FOR TIMIT BENCHMARK (timit_phoneme.py)
# =============================================================================

def evaluate_with_calibration_timit(
    model,  # TIMITModel
    dataloader,
    device: torch.device,
    is_semicrf: bool = True,
):
    """
    Extended evaluation that includes calibration metrics for TIMIT.
    
    Add this function to timit_phoneme.py.
    """
    from calibration import (
        CalibrationEvaluator,
        derive_boundary_probs_from_positions,
        compute_boundary_errors,
    )
    
    model.eval()
    
    all_boundary_probs = []
    all_true_boundaries = []
    all_boundary_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            
            hidden = model.encoder(features)
            
            if is_semicrf and hasattr(model.crf, 'compute_boundary_marginals'):
                boundary_probs = model.crf.compute_boundary_marginals(hidden, lengths)
            else:
                position_probs = model.crf.compute_position_marginals(hidden, lengths)
                boundary_probs = []
                for i in range(len(lengths)):
                    seq_len = lengths[i].item()
                    pos_p = position_probs[i, :seq_len].cpu().numpy()
                    bp = derive_boundary_probs_from_positions(pos_p, method="transition")
                    boundary_probs.append(bp)
            
            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                true_labels = labels[i, :seq_len].cpu().numpy()
                
                true_bounds = np.zeros(seq_len, dtype=bool)
                for j in range(1, seq_len):
                    if true_labels[j] != true_labels[j-1]:
                        true_bounds[j] = True
                
                if isinstance(boundary_probs, torch.Tensor):
                    bp = boundary_probs[i, :seq_len].cpu().numpy()
                else:
                    bp = boundary_probs[i]
                
                all_boundary_probs.append(bp)
                all_true_boundaries.append(true_bounds)
                all_boundary_errors.append(compute_boundary_errors(true_bounds))
    
    evaluator = CalibrationEvaluator()
    return evaluator.evaluate(
        boundary_probs=all_boundary_probs,
        true_boundaries=all_true_boundaries,
        boundary_errors=all_boundary_errors,
    )


# =============================================================================
# EXAMPLE: Modifying compare_models to include calibration
# =============================================================================

def compare_models_with_calibration(data_dir, **kwargs):
    """
    Example showing how to modify compare_models to include calibration.
    
    This is a template - copy the relevant parts into your existing
    compare_models function in each benchmark file.
    """
    from calibration import print_calibration_comparison, plot_calibration_comparison
    
    results = {}
    calibration_results = {}
    
    # Train linear CRF
    print("Training LINEAR CRF (K=1)")
    linear_model, linear_metrics = train_model(data_dir, model_type="linear", **kwargs)
    results["linear_crf"] = linear_metrics
    
    # Evaluate calibration for linear CRF
    # Note: For linear CRF, we derive boundary probs from position marginals
    linear_calibration = evaluate_with_calibration_gencode(  # or _timit
        linear_model, test_loader, device, is_semicrf=False
    )
    calibration_results["linear_crf"] = linear_calibration
    
    # Train semi-CRF  
    print("Training SEMI-CRF")
    semicrf_model, semicrf_metrics = train_model(data_dir, model_type="semicrf", **kwargs)
    results["semi_crf"] = semicrf_metrics
    
    # Evaluate calibration for semi-CRF
    # Note: Semi-CRF has native boundary probabilities
    semicrf_calibration = evaluate_with_calibration_gencode(  # or _timit
        semicrf_model, test_loader, device, is_semicrf=True
    )
    calibration_results["semi_crf"] = semicrf_calibration
    
    # Print standard metrics comparison
    # ... (existing code)
    
    # Print calibration comparison
    print_calibration_comparison(
        semicrf_calibration,
        linear_calibration,
        model_names=("Semi-CRF", "Linear CRF")
    )
    
    # Generate calibration plots
    plot_calibration_comparison(
        semicrf_calibration,
        linear_calibration,
        output_path="calibration_comparison.pdf"
    )
    
    return results, calibration_results


# =============================================================================
# IMPORTANT: Using UncertaintySemiMarkovCRFHead
# =============================================================================

"""
To get native boundary probabilities from the semi-CRF, you need to use
UncertaintySemiMarkovCRFHead instead of SemiMarkovCRFHead.

Change this in your model class:

    # Before
    from torch_semimarkov import SemiMarkovCRFHead
    self.crf = SemiMarkovCRFHead(...)
    
    # After
    from torch_semimarkov import UncertaintySemiMarkovCRFHead
    self.crf = UncertaintySemiMarkovCRFHead(...)

UncertaintySemiMarkovCRFHead inherits from SemiMarkovCRFHead and adds:
- compute_boundary_marginals(hidden, lengths) -> (batch, T) boundary probs
- compute_position_marginals(hidden, lengths) -> (batch, T, C) position probs
- compute_entropy_streaming(hidden, lengths) -> (batch,) entropy values

The linear CRF baseline (K=1) can still use SemiMarkovCRFHead, and we derive
boundary probabilities from position-level predictions using the
derive_boundary_probs_from_positions() function in calibration.py.
"""


# =============================================================================
# EXPECTED OUTPUT
# =============================================================================

EXAMPLE_OUTPUT = """
======================================================================
UNCERTAINTY CALIBRATION COMPARISON
======================================================================

Metric                                      Semi-CRF   Linear CRF        Δ
----------------------------------------------------------------------
Expected Calibration Error (↓)                0.0312       0.0847   -0.0535 ✓
Maximum Calibration Error (↓)                 0.0891       0.1523   -0.0632 ✓
Brier Score (↓)                               0.0234       0.0312   -0.0078 ✓
Selective Prediction AUC (↑)                  0.8934       0.8456   +0.0478 ✓
Uncertainty-Error Correlation (↑)             0.4521       0.2134   +0.2387 ✓

Confidence Interval Coverage (closer to nominal is better):
----------------------------------------------------------------------
  50% CI coverage (nominal=0.50): Semi-CRF=0.512 (width=8.3), Linear CRF=0.387 (width=12.1) ✓
  90% CI coverage (nominal=0.90): Semi-CRF=0.891 (width=23.4), Linear CRF=0.723 (width=31.2) ✓
  95% CI coverage (nominal=0.95): Semi-CRF=0.942 (width=31.2), Linear CRF=0.812 (width=42.5) ✓

======================================================================

Key insights:
- Semi-CRF has lower ECE: its confidence scores are more reliable
- Semi-CRF has higher uncertainty-error correlation: when it's uncertain, it's actually wrong
- Semi-CRF confidence intervals achieve closer to nominal coverage with smaller widths
- This means you can trust "boundary at position X ± Ybp (95% CI)" from semi-CRF
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nExample expected output:")
    print(EXAMPLE_OUTPUT)
