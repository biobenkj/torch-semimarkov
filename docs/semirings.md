# Understanding Semirings

Semirings are a mathematical abstraction that lets you swap out the "arithmetic" used in dynamic programming (DP) algorithms. In `torch-semimarkov`, that means you can run the *same* forward/scan code to answer different questions (partition functions, Viterbi decoding, top-k, entropy, cross-entropy) just by changing the semiring.

## Contents

- [The intuition](#the-intuition) - why semirings exist
- [Available semirings](#available-semirings) - what's in the box
- [Choosing a semiring](#choosing-a-semiring) - which one to use when
- [Backend compatibility](#backend-compatibility) - GPU vs CPU support
- [Practical examples](#practical-examples) - code for common tasks
- [Advanced: Checkpoint semirings](#advanced-checkpoint-semirings) - memory-efficient gradients

## The intuition

### The problem: one scan, lots of questions

Dynamic programming (DP) is the "fill a table from left to right" trick: you solve small prefixes first, then reuse those results to solve longer prefixes.

In a semi-CRF, at each position `n` you consider **all segments that could end at `n`**. Concretely: you can end with a segment of length 1, or 2, or 3, ... up to your max segment length (subject to staying in-bounds). For each possible segment length `k` you:

1. take the best/total/whatever score you had at position `n-k`
2. **extend** it with the score of the new segment `edge[n-k, k]`
3. then **combine** all those candidate ways of ending at `n`

In code-shaped pseudocode, the scan looks like this:

```text
for n in 1..N:
    candidates = []
    for k in valid_lengths_ending_at(n):
        candidates.append( extend(beta[n-k], edge[n-k, k]) )
    beta[n] = combine(candidates)
```

The key observation is that the *shape* of this computation never changes.
What changes is what you mean by:

- **extend**: "how do I add the score of a new segment onto what I already have?"
- **combine**: "if there are multiple ways to end here, how do I merge them?"

Different questions correspond to different meanings:

| Question you want to answer | `combine` should do | `extend` should do |
| :--- | :--- | :--- |
| Total probability / partition function | soft-sum over candidates (`logsumexp`) | add segment scores (in log space) |
| Best path (Viterbi) | take the max | add segment scores |
| Top-k paths | keep the top-k candidates | add segment scores to each |
| Uncertainty (entropy) | entropy-aware sum | entropy-aware add |
| Distillation / model comparison (cross-entropy) | cross-entropy-aware sum | cross-entropy-aware add |

Without semirings, you'd write separate DP implementations for each row. With semirings, `torch-semimarkov` implements the scan once and you just swap the arithmetic.

### What a semiring gives you

A **semiring** is just a small bundle of "how to combine" rules:

- **\(\oplus\)** ("oplus"): the `combine` operator
- **\(\otimes\)** ("otimes"): the `extend` operator
- **`zero`**: the identity for \(\oplus\) (combining with it changes nothing)
- **`one`**: the identity for \(\otimes\) (extending with it changes nothing)

If you remember only one thing, make it this:

> Same DP scan, different semiring = different question answered.

#### (Optional) the same idea, written as math

If you like the compact version, the update step can be written as:

<details>
<summary>Show the math</summary>

$$
\beta[n] \,=\, \bigoplus_{k \in \mathcal{K}(n)} \left(\beta[n-k] \otimes \text{edge}[n-k, k]\right)
$$

</details>

## Available semirings

```python
from torch_semimarkov.semirings import (
    LogSemiring,           # Partition function, marginals
    MaxSemiring,           # Viterbi (best path)
    KMaxSemiring,          # Top-k paths
    EntropySemiring,       # Entropy of the path distribution
    CrossEntropySemiring,  # Cross-entropy / NLL-style objectives
    StdSemiring,           # Linear-space sum-product (mostly for debugging)
)
```

### LogSemiring (default)

**Use for:** training (NLL), computing marginals, normalization, uncertainty workflows that start from $\log Z$.

The log semiring computes the **log partition function**:

$$
Z = \sum_{\pi} \exp(\text{score}(\pi)), \qquad \log Z = \log \sum_{\pi} \exp(\text{score}(\pi))
$$

This is what you need for CRF-style likelihoods and marginals.

- $\oplus = \mathrm{logsumexp}$
- $\otimes = +$
- `zero` $= -\infty$
- `one` $= 0$

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

crf = SemiMarkov(LogSemiring)
log_Z, _ = crf.logpartition(edge, lengths=lengths)
# log_Z: log of sum(exp(score)) over all valid segmentations
```

### MaxSemiring

**Use for:** inference (Viterbi), getting the single best segmentation.

This swaps summation for maximization, giving the best score over all paths.

- $\oplus = \max$
- $\otimes = +$
- `zero` $= -\infty$
- `one` $= 0$

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import MaxSemiring

crf = SemiMarkov(MaxSemiring)
best_score, _ = crf.logpartition(edge, lengths=lengths)
```

Note: in many DP libraries (including this style of implementation), "marginals" under `MaxSemiring` become *hard* selections for the best path (0/1 indicators).

### KMaxSemiring

**Use for:** N-best lists, multiple hypotheses, "beam-like" DP.

Instead of keeping a single score per state, this keeps the top-$k$ scores per state, merging them as the DP progresses.

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import KMaxSemiring

crf = SemiMarkov(KMaxSemiring(k=5))
topk_scores, _ = crf.logpartition(edge, lengths=lengths)
# topk_scores: best 5 path scores (per batch item)
```

### EntropySemiring

**Use for:** uncertainty quantification, calibration/analysis, active learning signals.

The entropy semiring computes the entropy of the posterior over segmentations:

$$
H(P) = -\sum_{\pi} P(\pi)\,\log P(\pi)
$$

High entropy means the model is unsure (lots of plausible segmentations). Low entropy means the model is confident.

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import EntropySemiring

crf = SemiMarkov(EntropySemiring)
entropy, _ = crf.logpartition(edge, lengths=lengths)
```

### CrossEntropySemiring

**Use for:** cross-entropy style training objectives (including NLL as a special case), especially when you want the DP to directly produce a cross-entropy quantity.

Cross-entropy between a target distribution $Q$ over paths and the model distribution $P$ is:

$$
H(Q, P) = -\sum_{\pi} Q(\pi)\,\log P(\pi)
$$

A common special case is when $Q$ is a delta distribution on the gold path $\pi^\star$. Then:

$$
H(\delta_{\pi^\star}, P) = -\log P(\pi^\star) = \log Z - \mathrm{score}(\pi^\star)
$$

You can always compute this "by hand" with `LogSemiring` as `log_Z - gold_score`. The motivation for `CrossEntropySemiring` is that it can compute a cross-entropy quantity *directly in the semiring*, which can be convenient and sometimes more numerically stable depending on the target representation.

How to use it in `torch-semimarkov`:

- You pass **two sets of log edge potentials** (same shape as usual): one for *P* and one for *Q*.
- Pass them as a Python **list**: `[edge_P, edge_Q]` (a tuple will *not* be treated as a pair by the current helper code).
- `sum(...)` returns the cross-entropy in the third semiring component. If you want the tracked log-partitions too, request raw output.

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import CrossEntropySemiring

crf = SemiMarkov(CrossEntropySemiring)

# edge_P and edge_Q: both shaped (batch, N-1, K, C, C), in log-space
edge_P = ...
edge_Q = ...

# Cross-entropy H(P, Q) for each batch item (in nats)
h_pq = crf.sum([edge_P, edge_Q], lengths=lengths)

# If you want everything the semiring tracks:
raw = crf.sum([edge_P, edge_Q], lengths=lengths, _raw=True)  # shape (3, batch)
logZ_P, logZ_Q, h_pq = raw[0], raw[1], raw[2]
```

When is this useful?
- **Distillation / matching a teacher distribution**: let *P* be the teacher lattice distribution and *Q* be the student distribution; minimize `H(P,Q)`.
- **Comparing two models**: compute cross-entropy or combine with `KLDivergenceSemiring` depending on what you need.

If your target is a **single gold path** (a delta distribution), the usual CRF NLL is typically simpler to compute with `LogSemiring` as `log_Z - gold_score`.

### StdSemiring

**Use for:** debugging / sanity checks in linear space.

This is the usual sum-product semiring in real space. It's generally less numerically stable for long sequences than `LogSemiring`.

## Choosing a semiring

| Task | Semiring | What you get |
| --- | --- | --- |
| Training (CRF NLL) | `LogSemiring` | $\log Z$ (combine with gold score for loss) |
| Get marginal probabilities | `LogSemiring` | marginal probabilities over edges/segments |
| Predict best segmentation | `MaxSemiring` | best path score / hard selections |
| Multiple predictions | `KMaxSemiring(k=N)` | top-$N$ path scores |
| Measure uncertainty | `EntropySemiring` | entropy $H(P)$ |
| Cross-entropy between distributions | `CrossEntropySemiring` | cross-entropy $H(P,Q) = -\mathbb{E}_{x\sim P}[\log Q(x)]$ from `[edge_P, edge_Q]` |
| Debug in linear space | `StdSemiring` | real-space sums/products (can underflow/overflow) |

## Backend compatibility

`torch-semimarkov` typically has a "pure PyTorch" implementation and may have accelerated kernels (for example, Triton/CUDA) for some semirings. Accelerated support is usually best for the common scalar semirings and may fall back for structured semirings (entropy, top-k, etc.).

A typical situation looks like:

| Semiring | Pure PyTorch | Triton / CUDA kernel |
| --- | --- | --- |
| `LogSemiring` | yes | yes |
| `MaxSemiring` | yes | yes |
| `EntropySemiring` | yes | partial (often falls back) |
| `KMaxSemiring` | yes | partial (often falls back) |
| `CrossEntropySemiring` | yes | partial (often falls back) |

If an accelerated kernel doesn't support the semiring you chose, the library will fall back to the non-accelerated implementation.

## Practical examples

### Training with LogSemiring (classic CRF NLL)

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

crf = SemiMarkov(LogSemiring)

def compute_loss(edge, lengths, gold_edges):
    # 1) log-partition
    log_Z, _ = crf.logpartition(edge, lengths=lengths)

    # 2) score the gold path (this assumes gold_edges is a compatible 0/1 mask)
    gold_score = (edge * gold_edges).sum(dim=(1, 2, 3, 4))

    # 3) negative log-likelihood
    return (log_Z - gold_score).mean()
```

### Inference with MaxSemiring (Viterbi-style)

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import MaxSemiring

crf_max = SemiMarkov(MaxSemiring)

def predict(edge, lengths):
    # hard_marginals are typically 1 on edges used by the best path (and 0 otherwise)
    _, hard_marginals = crf_max.marginals(edge, lengths=lengths)
    labels, spans = SemiMarkov.from_parts(hard_marginals)
    return labels, spans
```

### Getting top-k hypotheses

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import KMaxSemiring

crf_k = SemiMarkov(KMaxSemiring(k=5))

def topk(edge, lengths):
    topk_scores, _ = crf_k.logpartition(edge, lengths=lengths)
    return topk_scores
```

### Measuring uncertainty (entropy)

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import EntropySemiring

crf_ent = SemiMarkov(EntropySemiring)

def normalized_entropy(edge, lengths):
    entropy, _ = crf_ent.logpartition(edge, lengths=lengths)
    return entropy / lengths.float()
```

### Keeping separate semirings for train vs decode

```python
import torch.nn as nn
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring, MaxSemiring

class MySemiCRFModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.crf_train = SemiMarkov(LogSemiring)
        self.crf_decode = SemiMarkov(MaxSemiring)

    def forward(self, x, lengths):
        edge = self.head(self.encoder(x))
        log_Z, _ = self.crf_train.logpartition(edge, lengths=lengths)
        return log_Z, edge

    def predict(self, x, lengths):
        edge = self.head(self.encoder(x))
        _, hard = self.crf_decode.marginals(edge, lengths=lengths)
        return SemiMarkov.from_parts(hard)
```

## Advanced: Checkpoint semirings

For large state spaces, the backward pass can get memory-hungry because DP needs saved intermediates for autograd. Checkpoint semirings trade compute for memory by recomputing some intermediates during the backward pass instead of storing everything.

```python
from torch_semimarkov.semirings.checkpoint import (
    CheckpointSemiring,
    CheckpointShardSemiring,
)
from torch_semimarkov.semirings import LogSemiring
from torch_semimarkov import SemiMarkov

# Wrap any base semiring with checkpointing
CheckpointedLog = CheckpointSemiring(LogSemiring)
crf = SemiMarkov(CheckpointedLog)

# Sharded version for even larger models / state spaces
ShardedLog = CheckpointShardSemiring(LogSemiring, num_shards=4)
crf_sharded = SemiMarkov(ShardedLog)
```

Use checkpoint semirings when:
- you're running out of GPU memory during training
- you have a large state space (for example, big label sets, long max segment length, or both)
- you're willing to trade extra compute for lower memory use

## Summary

- Semirings let you ask different questions with the same DP algorithm
- `LogSemiring` for training and marginals (most common)
- `MaxSemiring` for Viterbi decoding
- `KMaxSemiring` for top-k hypotheses
- `EntropySemiring` for uncertainty via entropy
- `CrossEntropySemiring` for distillation or model comparison (cross-entropy between two path distributions, $H(P,Q)$)
- Checkpoint semirings when memory is tight