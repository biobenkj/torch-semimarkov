# Understanding Semirings

Semirings are a mathematical abstraction that lets you swap out the "arithmetic" used in dynamic programming (DP) algorithms. This guide explains what they are, why they matter, and how to use them in `torch-semimarkov`.

## Contents

- [The Intuition](#the-intuition) — Why semirings exist
- [Available Semirings](#available-semirings) — What's in the box
- [Choosing a Semiring](#choosing-a-semiring) — Which one to use when
- [Backend Compatibility](#backend-compatibility) — GPU vs CPU support
- [Practical Examples](#practical-examples) — Code for common tasks
- [Advanced: Checkpoint Semirings](#advanced-checkpoint-semirings) — Memory-efficient gradients

## The Intuition

### The Problem: One Algorithm, Many Questions

Consider the forward algorithm for semi-CRFs. At its core, it is a dynamic programming recurrence that calculates a score for a position based on previous positions:

$$
\beta[n] = \bigoplus_{k} (\beta[n-k] \otimes \text{edge}[n-k, k])
$$

The interesting thing is that the **structure** of this recurrence (the scan) is always the same, but the **operations** ($\oplus$ and $\otimes$) depend on what question you are asking:

| Question | $\oplus$ (Combine) | $\otimes$ (Extend) | Identity |
| :--- | :--- | :--- | :--- |
| **Total Probability** (Partition) | `logsumexp` | `+` (add log-probs) | $-\infty$ |
| **Best Path** (Viterbi) | `max` | `+` (add scores) | $-\infty$ |
| **Top-K Paths** | `merge_topk` | `add_to_all` | `[]` |
| **Entropy** | `entropy_sum` | `entropy_add` | $0$ |

Without semirings, you would need to write a separate, complex scanning algorithm for each question. With semirings, `torch-semimarkov` implements the scan once and simply swaps the arithmetic operators.

### Why This Matters

You don't need to understand the underlying abstract algebra to use this package. The key insight is:

> **Different semirings answer different questions about the same model.**

Your edge potentials (the output of your neural network) stay the same. Your model architecture stays the same. You just swap the `Semiring` class to get different outputs.

## Available Semirings

```python
from torch_semimarkov.semirings import (
    LogSemiring,
    MaxSemiring,
    EntropySemiring,
    KMaxSemiring,
    CrossEntropySemiring,
    StdSemiring
)