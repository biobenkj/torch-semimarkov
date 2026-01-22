# API reference

For practical usage examples showing how to integrate these APIs with upstream encoders, see the [Integration guide](workflow_integration.md).

## SemiMarkovCRFHead (Recommended)

The high-level module for most use cases. Uses the streaming Triton kernel internally,
which is the recommended approach for both training and inference:

```python
from torch_semimarkov import SemiMarkovCRFHead

class SemiMarkovCRFHead(nn.Module):
    def __init__(
        self,
        num_classes: int,              # Number of label classes (C)
        max_duration: int,             # Maximum segment duration (K)
        hidden_dim: int = None,        # Optional: projection from encoder dim
        init_scale: float = 0.1,       # Parameter initialization scale
        duration_distribution: str = None,  # "learned", "geometric", "poisson", etc.
    ):
        """
        CRF head for Semi-Markov sequence labeling.

        Compatible with DDP - gradients sync automatically via standard PyTorch.
        Memory: O(KC) independent of sequence length T.
        """

    def forward(
        self,
        hidden_states,  # (batch, T, hidden_dim) or (batch, T, C)
        lengths,        # (batch,) sequence lengths
        use_triton=True,
    ) -> dict:
        """
        Compute partition function.

        Returns:
            dict with 'partition' (batch,) and 'cum_scores' (batch, T+1, C)
        """

    def compute_loss(
        self,
        hidden_states,
        lengths,
        labels,         # (batch, T) per-position labels
        use_triton=True,
        reduction="mean",
    ) -> Tensor:
        """Compute negative log-likelihood loss."""

    def decode(
        self,
        hidden_states,
        lengths,
        use_triton=True,
    ) -> Tensor:
        """Viterbi decoding - returns best score (batch,)."""

    def decode_with_traceback(
        self,
        hidden_states,
        lengths,
        max_traceback_length=10000,
    ) -> ViterbiResult:
        """Viterbi with path reconstruction. Returns (scores, segments)."""
```

**Example usage:**

```python
from torch_semimarkov import SemiMarkovCRFHead

# Create CRF head
crf = SemiMarkovCRFHead(
    num_classes=24,
    max_duration=100,
    hidden_dim=512,
    duration_distribution="geometric",  # or "learned", "poisson", etc.
)

# Encoder output
hidden = encoder(x)  # (batch, T, 512)
lengths = torch.full((batch,), T)

# Forward pass
result = crf(hidden, lengths)
partition = result['partition']  # (batch,)

# Training with labels
labels = torch.randint(0, 24, (batch, T))
loss = crf.compute_loss(hidden, lengths, labels)
loss.backward()

# Viterbi decoding with traceback
result = crf.decode_with_traceback(hidden, lengths)
for seg in result.segments[0]:
    print(f"[{seg.start}, {seg.end}] label={seg.label}")
```

## UncertaintySemiMarkovCRFHead

Extended CRF head with uncertainty quantification:

```python
from torch_semimarkov import UncertaintySemiMarkovCRFHead

class UncertaintySemiMarkovCRFHead(SemiMarkovCRFHead):
    """
    SemiMarkovCRFHead with uncertainty methods.

    Additional methods for boundary confidence and active learning.
    """

    def compute_boundary_marginals(
        self,
        hidden_states,
        lengths,
        use_streaming=True,
        normalize=True,
    ) -> Tensor:
        """
        P(boundary at position t) for each position.

        Returns: (batch, T) boundary probabilities
        """

    def compute_position_marginals(
        self,
        hidden_states,
        lengths,
    ) -> Tensor:
        """
        P(label=c at position t) for each position.

        Returns: (batch, T, C) label probabilities
        """

    def compute_entropy_streaming(
        self,
        hidden_states,
        lengths,
    ) -> Tensor:
        """
        Approximate entropy from marginals (works for T > 10K).

        Returns: (batch,) entropy estimates
        """

    def compute_loss_uncertainty_weighted(
        self,
        hidden_states,
        lengths,
        labels,
        uncertainty_weight=1.0,
        focus_mode="high_uncertainty",  # or "boundary_regions"
        use_triton=False,
        reduction="mean",
    ) -> Tensor:
        """
        Uncertainty-weighted loss for active learning.

        L_weighted = (1 + lambda * uncertainty) * NLL
        """
```

**Example usage:**

```python
from torch_semimarkov import UncertaintySemiMarkovCRFHead

model = UncertaintySemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)

# Boundary confidence for decision support
boundary_probs = model.compute_boundary_marginals(hidden, lengths)

# Uncertainty-weighted training for active learning
loss = model.compute_loss_uncertainty_weighted(hidden, lengths, labels)
```

## Streaming API (Recommended for Training)

The streaming API is the **recommended approach for training** on GPU. It uses Triton
kernels that compute edges on-the-fly, providing both memory efficiency and reliable
gradient computation.

For very long sequences (T = 100K - 400K+) where edge tensor cannot fit in memory:

```python
from torch_semimarkov import semi_crf_streaming_forward

def semi_crf_streaming_forward(
    cum_scores,       # (batch, T+1, C) cumulative projected scores
    transition,       # (C, C) or (K, C, C) transition matrix
    duration_bias,    # (K, C) duration bias
    lengths,          # (batch,) sequence lengths
    max_duration,     # K
    semiring="log",   # "log" (partition) or "max" (Viterbi)
    use_triton=True,  # Use Triton kernel if available
) -> Tensor:
    """
    Memory-efficient Semi-CRF forward with on-the-fly edge computation.

    Memory: O(T*C) for cum_scores vs O(T*K*C^2) for edge tensor
    - T=400K, K=3K, C=24: 38 MB vs 2.76 TB

    Returns:
        partition: (batch,) log partition function or Viterbi score
    """
```

**Example usage:**

```python
from torch_semimarkov import semi_crf_streaming_forward

# Pre-project features (outside kernel)
projected = hidden @ W_content  # (batch, T, C)
projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!

# Build cumulative scores
cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
cum_scores[:, 1:] = torch.cumsum(projected.float(), dim=1)

# Streaming forward (edges computed on-the-fly)
partition = semi_crf_streaming_forward(
    cum_scores, transition, duration_bias, lengths, max_duration
)
```

## Duration Distributions

Flexible parameterization for segment duration priors:

```python
from torch_semimarkov import (
    create_duration_distribution,
    LearnedDuration,
    GeometricDuration,
    NegativeBinomialDuration,
    PoissonDuration,
    UniformDuration,
    CallableDuration,
)

# Factory function
dur = create_duration_distribution(
    "geometric",      # or "learned", "negative_binomial", "poisson", "uniform"
    max_duration=100,
    num_classes=24,
    init_logit=-1.0,  # Distribution-specific kwargs
)
bias = dur()  # Returns (K, C) tensor
```

**Available distributions:**

| Distribution | Formula | Use case |
|--------------|---------|----------|
| `LearnedDuration` | Fully learned | Most flexible, default |
| `GeometricDuration` | P(k) ~ p(1-p)^(k-1) | Exponential decay, numerically stable |
| `NegativeBinomialDuration` | Generalizes geometric | Peaked distributions |
| `PoissonDuration` | P(k) ~ lambda^k/k! | Characteristic segment length |
| `UniformDuration` | P(k) = const | No duration preference |
| `CallableDuration` | User-defined | Full customization |

**Numerical stability note:**

`NegativeBinomialDuration` with very small `r` values (init_log_r < -10) can cause
numerical instability. A runtime warning is emitted when non-finite values are detected.
Use `GeometricDuration` as a stable alternative.

## SemiMarkov (Low-level API)

Low-level API with semiring abstraction for advanced use cases:

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring, MaxSemiring

class SemiMarkov(semiring):
    def logpartition(
        self,
        log_potentials,  # (batch, T-1, K, C, C) edge potentials
        lengths=None,    # (batch,) sequence lengths
        force_grad=False,
    ) -> Tuple[Tensor, List[Tensor], None]:
        """
        Compute log partition function using streaming linear scan.

        Memory: O(KC) via ring buffer, independent of sequence length T.

        Returns:
            v: (batch,) log partition values
            edges: list of edge marginals (for gradient computation)
            charts: None (streaming scan does not store charts)
        """

    def marginals(
        self,
        log_potentials,
        lengths=None,
    ) -> Tensor:
        """
        Compute edge marginals via backward pass.

        Returns: (batch, T-1, K, C, C) marginal probabilities
        """

    @staticmethod
    def hsmm(init, trans_z, trans_l, emission) -> Tensor:
        """
        Convert HSMM parameters to Semi-Markov edge potentials.

        Args:
            init: (C,) initial state distribution
            trans_z: (C, C) state transition matrix
            trans_l: (C, K) duration distribution per state
            emission: (batch, T, K, C) emission scores

        Returns:
            edge: (batch, T, K, C, C) edge potentials
        """
```

## Semirings

```python
from torch_semimarkov.semirings import (
    LogSemiring,      # Standard log-space (sum-product)
    MaxSemiring,      # Viterbi decoding (max-product)
    StdSemiring,      # Standard arithmetic
    KMaxSemiring,     # Top-k paths
    EntropySemiring,  # Entropy computation
)

from torch_semimarkov.semirings.checkpoint import (
    CheckpointSemiring,       # Gradient checkpointing
    CheckpointShardSemiring,  # Sharded checkpointing
)
```

## Triton Fused Scan (Inference Only)

Direct access to the Triton kernel for **inference only** when you have pre-computed edge tensors.

> **Note:** Even for inference, the streaming API is typically faster because computing
> edges on-the-fly from O(T×C) data is faster than loading O(T×K×C²) pre-computed edges.
> Use this API only when edge tensors are pre-computed from an external source.

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

def semi_crf_triton_forward(
    edge,            # (batch, T-1, K, C, C) potentials
    lengths,         # (batch,) sequence lengths
    use_triton=True,
    validate=False,  # Use float64 PyTorch for validation
    semiring="log",  # "log" (partition) or "max" (Viterbi)
    use_compile=True,
) -> Tensor:
    """
    Semi-Markov CRF forward scan optimized for inference.

    Primary use case: Fast forward pass with pre-computed edge tensors.
    For training, prefer the streaming API (semi_crf_streaming_forward).

    Returns:
        partition: (batch,) log partition function or Viterbi score
    """
```

**Performance comparison (forward-only, NVIDIA L40S):**

Even when the edge tensor fits in memory, streaming is faster:

| Configuration | triton_scan | streaming | Streaming Advantage |
|---------------|-------------|-----------|---------------------|
| K=100, batch=64 | 127ms, 14GB | 38ms, 6MB | 3.35× faster, 2,393× less memory |
| K=500, batch=32 | 330ms, 35GB | 224ms, 3MB | 1.48× faster, 11,795× less memory |

**Routing behavior:**

| Context | Execution Path | Recommendation |
|---------|----------------|----------------|
| `requires_grad=False` + CUDA | Custom Triton kernel | OK for inference if edges pre-exist |
| `requires_grad=True` + CUDA | `torch.compile` fallback | **Not recommended** - see warning |
| CPU or Triton unavailable | PyTorch reference | Use streaming API instead |

> **Warning: torch.compile limitations for training**
>
> The `torch.compile` path for backward passes has critical limitations at production scales:
> - **RecursionError**: Sequences longer than T≈1000 exceed Python's recursion limit in inductor
> - **OOM during backward**: Compiled graphs require 2×+ memory for gradient buffers
> - **Compilation time**: 20+ minutes for T=1000, essentially unusable
>
> For training, use `SemiMarkovCRFHead` or `semi_crf_streaming_forward`, which have
> hand-written Triton backward kernels (no compilation overhead).

**Example usage:**

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

# GPU inference: fast custom Triton kernel (primary use case)
partition = semi_crf_triton_forward(edge.cuda(), lengths.cuda())

# Viterbi score (max semiring)
viterbi = semi_crf_triton_forward(edge.cuda(), lengths.cuda(), semiring="max")
```

## Helper Types

```python
from torch_semimarkov import Segment, ViterbiResult

@dataclass
class Segment:
    """A single segment from Viterbi decoding."""
    start: int   # Start position (inclusive)
    end: int     # End position (inclusive)
    label: int   # Label class
    score: float # Segment score contribution

class ViterbiResult(NamedTuple):
    """Result from decode_with_traceback."""
    scores: Tensor              # (batch,) best scores
    segments: List[List[Segment]]  # Per-batch segment lists
```

## Memory Comparison

| Scenario | Edge tensor size | Streaming API size |
|----------|------------------|-------------------|
| T=1K, K=32, C=24 | 18 MB | 96 KB |
| T=10K, K=100, C=24 | 5.5 GB | 960 KB |
| T=400K, K=3K, C=24 | **2.76 TB** | 38 MB |

**Recommended usage:**

| Use case | API | Why |
|----------|-----|-----|
| Training (any T) | `SemiMarkovCRFHead` or `semi_crf_streaming_forward` | Hand-written backward kernels, no torch.compile |
| Inference (recommended) | `SemiMarkovCRFHead` or `semi_crf_streaming_forward` | Faster AND less memory than triton_scan |
| Inference with pre-existing edges | `semi_crf_triton_forward` | Only if edges already materialized externally |
| Very long sequences (T > 10K) | Streaming API | Edge tensor cannot fit in memory |

## See Also

- [Backends and Triton kernel](backends.md) - Detailed kernel behavior and options
- [Integration guide](workflow_integration.md) - End-to-end training examples
