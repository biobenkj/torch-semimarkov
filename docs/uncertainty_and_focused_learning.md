# Uncertainty Quantification and Focused Learning

This guide covers uncertainty quantification methods for clinical applications where boundary confidence is critical, and focused learning techniques for active learning and curriculum learning workflows.

## Contents

- [Introduction](#introduction) — why uncertainty matters in clinical applications
- [Understanding boundary uncertainty](#understanding-boundary-uncertainty) — marginal probabilities and the forward-backward algorithm
- [Computing uncertainty at scale](#computing-uncertainty-at-scale) — streaming vs exact methods
- [API reference](#api-reference) — `UncertaintySemiMarkovCRFHead` methods
- [Focused learning](#focused-learning) — uncertainty-weighted loss for active learning
- [Clinical examples](#clinical-examples) — ECG, EEG, and genomics applications
- [Best practices](#best-practices) — when to use streaming vs exact methods
- [Troubleshooting](#troubleshooting) — common issues and solutions

## Introduction

### Why uncertainty matters for clinical applications

In clinical sequence segmentation, the cost of errors is not uniform:

- **Misidentifying a segment boundary** can change clinical interpretation (e.g., a gene boundary error changes protein predictions)
- **Off-by-one errors** compound when multiple annotations are combined
- **High-confidence errors** are more damaging than uncertain predictions that are flagged for review

Semi-Markov CRFs provide principled uncertainty estimates because they define a **distribution over segmentations**, not just point predictions. This enables:

1. **Boundary confidence**: P(boundary at position t) for each position
2. **Segment uncertainty**: Which regions need manual review?
3. **Active learning**: Sample high-uncertainty regions for labeling
4. **Curriculum learning**: Train on easy examples first

### The problem: Off-by-one errors and boundary ambiguity

Standard per-position classifiers predict a label for each position independently. This creates two problems:

1. **No segment-level reasoning**: The model doesn't explicitly consider segment durations or transitions
2. **No uncertainty about boundaries**: You get a single boundary position, not a probability distribution

With a semi-CRF, the model reasons about entire segments, and you can ask: "What's the probability that a segment boundary occurs within positions 100-110?"

## Understanding boundary uncertainty

### What are marginal probabilities?

Given a sequence, the semi-CRF defines a probability distribution over all possible segmentations. The **marginal probability** at a position is the probability that a segment boundary occurs there, summed over all segmentations:

```
P(boundary at position t) = Σ_{segmentations with boundary at t} P(segmentation)
```

This is computed efficiently via the forward-backward algorithm, which is built into the semi-CRF inference.

### Gradient-based marginals for streaming

For clinical-scale sequences (T >= 10K), materializing the full edge tensor for exact marginals would require terabytes of memory. Instead, we use **gradient-based marginals**:

The key insight is that the gradient of the log partition function with respect to the input scores encodes marginal information:

```
∂(log Z) / ∂(cum_scores[t, c]) ∝ P(label c used at position t)
```

This gradient is computed during the backward pass of the streaming algorithm, so it works at any sequence length with O(KC) memory.

### Streaming vs exact methods

| Method | Memory | Sequence Length | API |
|--------|--------|-----------------|-----|
| **Streaming** (gradient-based) | O(KC) | Any (T >= 10K+) | `use_streaming=True` |
| **Exact** (forward-backward) | O(T×K×C²) | T < 10K | `use_streaming=False` |

For clinical applications, **streaming is the default** because it scales to any sequence length.

## Computing uncertainty at scale

### Two-path approach

The `UncertaintySemiMarkovCRFHead` automatically selects the appropriate method based on sequence length and available resources:

```python
from torch_semimarkov import UncertaintySemiMarkovCRFHead

model = UncertaintySemiMarkovCRFHead(
    num_classes=5,
    max_duration=100,
    hidden_dim=64
)

# For clinical-scale sequences (T >= 10K), use streaming
boundary_probs = model.compute_boundary_marginals(
    hidden_states, lengths, use_streaming=True
)

# For short sequences (T < 10K), exact method is also available
# (but streaming works fine too)
boundary_probs = model.compute_boundary_marginals(
    hidden_states, lengths, use_streaming=False
)
```

### Memory comparison

For T=50,000 (a typical clinical recording), K=100, C=8:

| Component | Exact Method | Streaming Method |
|-----------|--------------|------------------|
| Edge tensor | 320 GB | Not materialized |
| Ring buffer | - | 6.4 KB |
| Cumulative scores | 1.6 MB | 1.6 MB |
| **Total** | **320 GB** | **< 2 MB** |

The streaming method is essential for clinical-scale data.

## API reference

### `UncertaintySemiMarkovCRFHead`

Extended CRF head with uncertainty quantification methods for clinical applications.

```python
from torch_semimarkov import UncertaintySemiMarkovCRFHead

model = UncertaintySemiMarkovCRFHead(
    num_classes=24,      # Number of segment labels
    max_duration=100,    # Maximum segment length
    hidden_dim=512       # Encoder hidden dimension
)
```

#### `compute_boundary_marginals()`

Compute P(boundary at position t) for each position.

```python
boundary_probs = model.compute_boundary_marginals(
    hidden_states,       # (batch, T, hidden_dim)
    lengths,             # (batch,) sequence lengths
    use_streaming=True,  # Use streaming method (required for T >= 10K)
    normalize=True       # Normalize to [0, 1] range
)
# Returns: (batch, T) tensor of boundary probabilities
```

**Interpretation:**
- High values (close to 1): Strong evidence for boundary at this position
- Low values (close to 0): Position is likely mid-segment
- Intermediate values: Boundary location is uncertain

#### `compute_position_marginals()`

Compute per-position label marginals P(label=c at position t).

```python
position_marginals = model.compute_position_marginals(
    hidden_states,  # (batch, T, hidden_dim)
    lengths         # (batch,) sequence lengths
)
# Returns: (batch, T, C) tensor of label probabilities
```

**Use cases:**
- Soft label assignments for downstream analysis
- Identifying positions with ambiguous class membership
- Generating label distributions for ensemble methods

#### `compute_entropy_streaming()`

Approximate entropy from marginal distribution (works at any scale).

```python
entropy = model.compute_entropy_streaming(
    hidden_states,  # (batch, T, hidden_dim)
    lengths         # (batch,) sequence lengths
)
# Returns: (batch,) tensor of entropy values
```

**Interpretation:**
- Higher entropy: More uncertainty about the segmentation
- Lower entropy: Model is confident in the segmentation

This is useful for:
- Flagging sequences that need manual review
- Curriculum learning (sort by entropy: easy → hard)
- Active learning (select high-entropy samples for labeling)

#### `compute_entropy_exact()`

Exact entropy via EntropySemiring (T < 10K only).

```python
entropy = model.compute_entropy_exact(
    hidden_states,  # (batch, T, hidden_dim)
    lengths         # (batch,) sequence lengths
)
# Returns: (batch,) tensor of exact entropy values
```

> **Warning**: This requires materializing the edge tensor. Will OOM for T > 10K!

#### `compute_loss_uncertainty_weighted()`

Uncertainty-weighted NLL loss for focused learning.

```python
loss = model.compute_loss_uncertainty_weighted(
    hidden_states,       # (batch, T, hidden_dim)
    lengths,             # (batch,) sequence lengths
    labels,              # (batch, T) per-position labels
    uncertainty_weight=1.0,           # Scale factor for weighting
    focus_mode="high_uncertainty",    # or "boundary_regions"
    reduction="mean"                  # "mean", "sum", or "none"
)
# Returns: Weighted NLL loss
```

**Focus modes:**
- `"high_uncertainty"`: Higher weight on samples with high entropy
- `"boundary_regions"`: Higher weight on samples with more boundary uncertainty

## Focused learning

### Uncertainty-weighted loss

The uncertainty-weighted loss scales the standard NLL loss by uncertainty:

```
weighted_loss = NLL × (1 + uncertainty_weight × uncertainty)
```

This causes the model to:
- Pay more attention to uncertain samples during training
- Learn more from ambiguous regions
- Converge faster on challenging examples

```python
# Standard training
loss = model.compute_loss(hidden_states, lengths, labels)

# Uncertainty-weighted training (for active learning)
loss = model.compute_loss_uncertainty_weighted(
    hidden_states, lengths, labels,
    uncertainty_weight=1.0,
    focus_mode="high_uncertainty"
)
```

### Curriculum learning

Sort samples by uncertainty (easy → hard) to stabilize training:

```python
# Compute entropy for all samples in a batch
entropy = model.compute_entropy_streaming(hidden_states, lengths)

# Sort indices by entropy (ascending = easy first)
sorted_indices = torch.argsort(entropy)

# Train on easy samples first, then progressively harder ones
for epoch in range(num_epochs):
    curriculum_fraction = min(1.0, 0.5 + epoch * 0.1)  # Start at 50%, grow
    num_samples = int(len(sorted_indices) * curriculum_fraction)

    batch_indices = sorted_indices[:num_samples]
    # Train on selected samples
```

### Active learning

Select the most uncertain samples for labeling:

```python
# Compute uncertainty for unlabeled pool
entropy = model.compute_entropy_streaming(unlabeled_hidden, unlabeled_lengths)

# Select top-k most uncertain for labeling
k = 100  # Budget for labeling
_, top_k_indices = torch.topk(entropy, k)

# Request labels for these samples
samples_to_label = unlabeled_data[top_k_indices]
```

This focuses labeling effort where it will most improve the model.

## Clinical examples

### ECG arrhythmia detection

For ECG recordings (250 Hz, 10s-60s), boundary precision around QRS complexes is critical:

```python
model = UncertaintySemiMarkovCRFHead(
    num_classes=5,      # N, V, S, F, Q beat types
    max_duration=100,   # ~400ms at 250Hz
    hidden_dim=64
)

# Get boundary uncertainty
boundary_probs = model.compute_boundary_marginals(hidden_states, lengths)

# Find positions with high boundary uncertainty (potential QRS)
high_uncertainty_mask = boundary_probs > 0.5

# Flag regions where boundary position is ambiguous
uncertain_regions = find_regions_with_spread_uncertainty(boundary_probs)
```

**Clinical use**: Flag beats where the QRS onset/offset is uncertain for cardiologist review.

### EEG sleep staging

For sleep studies (30s epochs over 4-12 hours), uncertainty at stage transitions matters:

```python
model = UncertaintySemiMarkovCRFHead(
    num_classes=5,      # Wake, N1, N2, N3, REM
    max_duration=300,   # 30s epochs at 10Hz feature rate
    hidden_dim=128
)

# Get boundary uncertainty at epoch transitions
boundary_probs = model.compute_boundary_marginals(hidden_states, lengths)

# Identify uncertain transitions (e.g., N1 ↔ N2)
transition_uncertainty = boundary_probs[:, epoch_boundaries]
```

**Clinical use**: Report confidence intervals for sleep stage transitions, not just point estimates.

### Genomics segmentation

For gene annotation (T = 10K-100K bp), boundary uncertainty affects protein predictions:

```python
model = UncertaintySemiMarkovCRFHead(
    num_classes=24,     # Gene structure categories
    max_duration=3000,  # Max exon/intron length
    hidden_dim=256
)

# Must use streaming for genomics-scale sequences
boundary_probs = model.compute_boundary_marginals(
    hidden_states, lengths, use_streaming=True
)

# Identify uncertain exon-intron boundaries
exon_intron_uncertainty = extract_boundary_uncertainty(
    boundary_probs, predicted_labels,
    transition_types=[(EXON, INTRON), (INTRON, EXON)]
)
```

**Clinical use**: Report confidence intervals for splice sites, propagate to downstream variant effect predictions.

## Best practices

### When to use streaming vs exact methods

**Use streaming (default) when:**
- T >= 10K (clinical sequences)
- Memory is constrained
- You need per-sample uncertainty (not global exact entropy)

**Use exact methods when:**
- T < 10K
- You need exact entropy values (not approximations)
- You're comparing against theoretical results

### Numerical stability

For long sequences (T > 100K):

1. **Always use float32** for cumulative scores:
   ```python
   cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float32)
   cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)
   ```

2. **Zero-center before cumsum** to prevent precision loss:
   ```python
   scores_centered = scores - scores.mean(dim=1, keepdim=True)
   cum_scores[:, 1:] = torch.cumsum(scores_centered.float(), dim=1)
   ```

### Interpreting uncertainty for clinical decision support

1. **Report confidence intervals, not point estimates**:
   ```python
   # Find positions where boundary probability is between 0.3 and 0.7
   uncertain_region = (boundary_probs > 0.3) & (boundary_probs < 0.7)
   # Report: "Boundary is between positions 100-110 (68% confidence)"
   ```

2. **Flag high-uncertainty predictions for review**:
   ```python
   entropy = model.compute_entropy_streaming(hidden_states, lengths)
   needs_review = entropy > entropy_threshold
   ```

3. **Propagate uncertainty to downstream analyses**:
   ```python
   # Use boundary distribution, not just argmax
   for position in range(T):
       boundary_prob = boundary_probs[0, position]
       # Weight downstream analyses by boundary confidence
   ```

## Troubleshooting

### OOM errors

**Problem**: Out of memory when computing marginals.

**Solution**: Use streaming method:
```python
# Don't do this for T >= 10K:
# boundary_probs = model.compute_boundary_marginals(hidden, lengths, use_streaming=False)

# Do this instead:
boundary_probs = model.compute_boundary_marginals(hidden, lengths, use_streaming=True)
```

### Numerical instability

**Problem**: NaN or Inf values in marginals or entropy.

**Solutions**:
1. Ensure cumulative scores are float32:
   ```python
   cum_scores = cum_scores.float()
   ```

2. Check for extreme input values:
   ```python
   assert torch.isfinite(hidden_states).all()
   ```

3. Use `normalize=True` for boundary marginals:
   ```python
   boundary_probs = model.compute_boundary_marginals(
       hidden, lengths, normalize=True
   )
   ```

### Marginals don't sum to expected value

**Problem**: Per-position marginals don't sum to 1 over classes.

**Clarification**: Boundary marginals represent P(boundary at position t), not P(label at position t). For per-position label distributions that sum to 1, use:

```python
position_marginals = model.compute_position_marginals(hidden, lengths)
# This sums to 1 over the class dimension
assert torch.allclose(position_marginals.sum(dim=-1), torch.ones(...))
```

### Exact methods fail for short sequences

**Problem**: `compute_entropy_exact` or `use_streaming=False` produces errors.

**Solution**: The exact methods require careful edge tensor construction. Some edge cases may produce NaN values. For production use, prefer streaming methods which are more robust:

```python
# Prefer this (always works):
entropy = model.compute_entropy_streaming(hidden, lengths)

# Over this (may fail for some inputs):
entropy = model.compute_entropy_exact(hidden, lengths)
```

## See also

- [Integration with upstream encoders](workflow_integration.md) — using semi-CRF with transformers and Mamba
- [Semirings](semirings.md) — LogSemiring, MaxSemiring, EntropySemiring
- [Streaming API](streaming_edge_api_roadmap.md) — technical details on O(KC) memory streaming
