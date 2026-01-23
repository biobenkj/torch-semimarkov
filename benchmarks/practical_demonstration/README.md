# Task Benchmarks: Semi-CRF vs Linear CRF

This directory contains task-level benchmarks demonstrating where Semi-CRFs provide meaningful improvements over linear CRFs. While the main `torch-semimarkov` benchmarks focus on computational efficiency (time, memory, throughput), these benchmarks address the question: **does duration modeling actually help?**

## The Core Argument

**Linear CRFs** model sequences as position-wise predictions with first-order Markov transition constraints:

$$P(y|x) \propto \exp\left(\sum_{t=1}^{T} \psi_{\text{emit}}(x_t, y_t) + \psi_{\text{trans}}(y_{t-1}, y_t)\right)$$

**Semi-CRFs** model *segments* as first-class objects with explicit duration distributions:

$$P(y|x) \propto \exp\left(\sum_{s \in \text{segments}} \psi_{\text{emit}}(x_{s}, c_s) + \psi_{\text{trans}}(c_{s-1}, c_s) + \psi_{\text{dur}}(c_s, d_s)\right)$$

The Semi-CRF should win when:

1. **Segment durations are informative** — characteristic lengths exist per class
2. **Segment-level coherence matters** — fragmented predictions are penalized
3. **The duration distribution is learnable** — sufficient segments per class

## Benchmarks

### 1. Gencode Exon/Intron Segmentation (`gencode_exon_intron.py`)

**Why this task?** Exons and introns have dramatically different, biologically meaningful length distributions:

| Feature | Median | 95th percentile | Max |
|---------|--------|-----------------|-----|
| Exon    | ~150bp | ~500bp          | ~10kb |
| Intron  | ~1kb   | ~10kb           | >100kb |

A linear CRF can't encode "exons are typically 50-500bp" — it only knows transition probabilities. A Semi-CRF explicitly penalizes implausible durations like a 3bp exon or a 10bp intron.

**Data:** Gencode GTF + reference genome, chromosome-based train/val/test split

**Expected results:**
- Position F1: Similar (both should get most positions right)
- Boundary F1: **Semi-CRF advantage** (better at exact junction detection)
- Segment F1: **Semi-CRF advantage** (fewer fragmented predictions)
- Duration calibration: **Semi-CRF advantage** (predicted lengths match true distribution)

### 2. TIMIT Phoneme Segmentation (`timit_phoneme.py`)

**Why this task?** This is the classic benchmark from the original Semi-CRF paper (Sarawagi & Cohen, 2004). Phonemes have characteristic durations:

| Phoneme type | Typical duration (frames @ 10ms) |
|--------------|----------------------------------|
| Stops (p, t, k) | 2-8 frames (20-80ms) |
| Vowels (aa, iy) | 5-15 frames (50-150ms) |
| Fricatives (s, sh) | 4-15 frames (40-150ms) |
| Silence | Variable (50-500ms) |

**Data:** TIMIT corpus (requires LDC license), standard train/test split

**Expected results:**
- Phone Error Rate: ~1-2% improvement (lower is better)
- Boundary F1: **Semi-CRF advantage**
- Historical precedent: Sarawagi & Cohen showed consistent gains

## Experimental Design

Both benchmarks use the **same encoder** with two CRF heads:

```
Encoder (BiLSTM or Mamba)
    │
    ├── Linear CRF (K=1)    ← This is just a special case!
    │
    └── Semi-CRF (K=500 or K=30)
```

**Key insight:** `torch-semimarkov` with `max_duration=1` degenerates to a standard linear CRF. This enables truly apples-to-apples comparison using identical code paths.

```python
# Linear CRF baseline
linear_model = SemiMarkovCRFHead(num_classes=C, max_duration=1, hidden_dim=H)

# Semi-CRF with duration modeling
semicrf_model = SemiMarkovCRFHead(num_classes=C, max_duration=500, hidden_dim=H)
```

Both models have identical:
- Encoder architecture
- Transition matrix parameterization
- Emission projection
- Optimizer and training schedule

The **only difference** is whether `K > 1` (duration modeling enabled).

## Metrics

### Position-level F1
Standard per-position classification accuracy. Both models should do similarly here.

### Boundary F1
Precision/recall of segment boundary detection:
- **Exact match** (tolerance=0): Did you nail the exact position?
- **Within tolerance** (tolerance=k): Did you get within k positions?

This is where Semi-CRFs should show clear advantage.

### Segment F1
A predicted segment is correct only if **all three** match: start position, end position, and label. This is the strictest metric.

### Duration Calibration
KL divergence between predicted and true segment length distributions per class. Lower is better. Semi-CRFs should produce segments with more realistic lengths.

### Phone Error Rate (TIMIT only)
Levenshtein edit distance between predicted and reference phone sequences, normalized by reference length. This is the standard TIMIT metric.

## Uncertainty Quantification Advantage

Beyond point predictions, Semi-CRFs provide a unique advantage in **uncertainty quantification**. The `calibration.py` module evaluates this systematically.

### What Linear CRFs Can Do

Linear CRFs can compute marginals via forward-backward:
- P(y_t = c | x) — position-level class probabilities
- P(y_{t-1} = c', y_t = c | x) — transition marginals

So you *can* derive boundary uncertainty by looking at where positions are likely to change class.

### What Semi-CRFs Give You That's Different

Semi-CRFs give you **segment-level** marginals:

$$P(\text{segment from } t \text{ to } t+k \text{ with label } c \mid x)$$

This is a joint distribution over (boundary position, duration, label). You can marginalize to get:

1. **P(boundary at t)** — probability of *any* segment ending at position t
2. **P(duration = k | label = c)** — learned duration distribution per class  
3. **Entropy over segmentations** — global uncertainty about the full parse

The key difference: Semi-CRF boundary probabilities come from a model that *explicitly reasons about segments as units*, not derived post-hoc from position-level predictions.

### Calibration Metrics

The `calibration.py` module evaluates:

| Metric | Description | Better |
|--------|-------------|--------|
| **ECE** | Expected Calibration Error: when model says 80% confident, is it right 80% of the time? | Lower |
| **MCE** | Maximum Calibration Error: worst-case miscalibration | Lower |
| **Brier Score** | Proper scoring rule for probabilistic predictions | Lower |
| **Selective Prediction AUC** | If we only predict high-confidence boundaries, does accuracy improve? | Higher |
| **Uncertainty-Error Correlation** | Does uncertainty predict actual mistakes? | Higher |
| **CI Coverage** | Does the 95% CI actually contain the true boundary 95% of the time? | Closer to nominal |

### Why This Matters

For genomics specifically:
- Variant effect prediction depends on whether a variant is in an exon or intron
- If the boundary is uncertain, you want to know *how* uncertain
- **"This splice site is at position 1523 ± 5bp (95% CI)"** is scientifically meaningful

Linear CRFs can't naturally express this—their boundary uncertainty is derived, not native.

### Using the Calibration Module

```python
from calibration import (
    CalibrationEvaluator,
    derive_boundary_probs_from_positions,
    print_calibration_comparison,
    plot_calibration_comparison,
)

# Semi-CRF: native boundary probabilities
semicrf_boundary_probs = model.crf.compute_boundary_marginals(hidden, lengths)

# Linear CRF: derive from position marginals
position_probs = model.crf.compute_position_marginals(hidden, lengths)
linear_boundary_probs = derive_boundary_probs_from_positions(position_probs)

# Evaluate calibration
evaluator = CalibrationEvaluator()
semicrf_results = evaluator.evaluate(semicrf_boundary_probs, true_boundaries)
linear_results = evaluator.evaluate(linear_boundary_probs, true_boundaries)

# Compare
print_calibration_comparison(semicrf_results, linear_results)
```

See `calibration_integration.py` for complete integration examples.

## Usage

### Quick validation (smoke test)

Before running the full benchmarks, verify the pipeline works end-to-end with synthetic data:

```bash
# Run all smoke tests (~6 seconds)
python smoke_test.py

# Verbose output for debugging
python smoke_test.py --verbose

# Run specific test suites
python smoke_test.py gencode        # GENCODE tests only
python smoke_test.py timit          # TIMIT tests only
python smoke_test.py calibration    # Calibration tests only
```

This validates:
- Library imports and core SemiMarkovCRFHead functionality
- Training pipelines for both benchmarks
- Evaluation metrics (position F1, boundary F1, PER)
- Calibration metrics (ECE, Brier score)
- Segment end conventions (inclusive end)

### Gencode benchmark

```bash
# 1. Download data
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz

# 2. Preprocess
python gencode/gencode_exon_intron.py preprocess \
    --gtf gencode.v44.annotation.gtf.gz \
    --fasta GRCh38.primary_assembly.genome.fa \
    --output-dir data/gencode/

# 3. Compare models
python gencode/gencode_exon_intron.py compare \
    --data-dir data/gencode/ \
    --max-duration 500 \
    --epochs 50
```

### TIMIT benchmark

```bash
# 1. Obtain TIMIT from LDC (requires license)

# 2. Preprocess
python timit/timit_phoneme.py preprocess \
    --timit-dir /path/to/TIMIT \
    --output-dir data/timit/

# 3. Compare models
python timit/timit_phoneme.py compare \
    --data-dir data/timit/ \
    --max-duration 30 \
    --epochs 50
```

### Unified runner with plotting

For running benchmarks with automatic figure generation:

```bash
# Run GENCODE benchmark with plotting
python run_benchmarks.py \
    --task gencode \
    --data-dir data/gencode/ \
    --output-dir results/gencode/ \
    --max-duration 500 \
    --epochs 50

# Run TIMIT benchmark with plotting
python run_benchmarks.py \
    --task timit \
    --data-dir data/timit/ \
    --output-dir results/timit/ \
    --max-duration 30 \
    --epochs 50

# Run both benchmarks
python run_benchmarks.py \
    --task all \
    --gencode-dir data/gencode/ \
    --timit-dir data/timit/ \
    --output-dir results/

# Regenerate plots from existing results
python run_benchmarks.py \
    --task plot \
    --json-path results/gencode/metrics.json \
    --output-dir results/gencode/figures/
```

This generates:
- `metrics.json` - Structured results for programmatic access
- `comparison_table.txt` - Text table for quick viewing
- `figures/comparison_bar.pdf` - Side-by-side bar chart
- `figures/boundary_tolerance.pdf` - Boundary F1 vs tolerance curve
- `figures/duration_kl.pdf` - Duration calibration (GENCODE only)

## Expected Output (Hypothetical)

The `compare` command will produce a table like the following. These numbers are illustrative of expected relative improvements, not measured results:

```
COMPARISON: Linear CRF vs Semi-CRF
============================================================
Metric                        Linear CRF        Semi-CRF            Δ
----------------------------------------------------------------------
position_f1_macro                 0.8234          0.8251       +0.0017
boundary_f1                       0.6123          0.6847       +0.0724  ← Key metric
segment_f1                        0.4521          0.5234       +0.0713  ← Key metric

Boundary F1 at different tolerances:
  tol=0                           0.6123          0.6847       +0.0724
  tol=1                           0.7234          0.7856       +0.0622
  tol=2                           0.7891          0.8312       +0.0421

Duration KL divergence (lower is better):
  intergenic                      0.8234          0.3421       ← Semi-CRF wins
  5UTR                            0.5123          0.2134       ← Semi-CRF wins
  CDS                             0.4521          0.1823       ← Semi-CRF wins
  3UTR                            0.5891          0.2512       ← Semi-CRF wins
  intron                          0.7234          0.2891       ← Semi-CRF wins
```

## Why This Matters for the Paper

The computational benchmarks in `benchmarks/` answer: "Is the streaming algorithm fast and memory-efficient?"

These task benchmarks answer three questions:
1. **Does the Semi-CRF produce better segmentations?** (boundary/segment F1)
2. **Are the uncertainty estimates trustworthy?** (calibration metrics)
3. **Can we give meaningful confidence intervals?** (CI coverage)

Together, they make the complete argument:
1. **Semi-CRFs are better** when segment durations matter (task benchmarks)
2. **Semi-CRF uncertainties are more calibrated** for actionable downstream use (calibration)
3. **torch-semimarkov makes them practical** at genome scale (computational benchmarks)

## References

- Sarawagi, S., & Cohen, W. W. (2004). Semi-Markov Conditional Random Fields for Information Extraction. *NeurIPS*.
- Lee, K. F., & Hon, H. W. (1989). Speaker-independent phone recognition using hidden Markov models. *IEEE TASSP*.
