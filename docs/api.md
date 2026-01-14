# API reference

For practical usage examples showing how to integrate these APIs with neural
network encoders, see the [Integration guide](workflow_integration.md).

## SemiMarkov

```python
class SemiMarkov(semiring):
    def logpartition(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,            # (batch,) sequence lengths
        use_linear_scan=None,    # Auto-select based on KC (>200 uses linear scan)
        use_vectorized=False,    # Use vectorized scan (O(TKC) memory, 2-3x faster)
        use_banded=False,        # Use banded matrix operations
        banded_perm="auto",      # Permutation strategy
        banded_bw_ratio=0.6,     # Bandwidth threshold
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute log partition function and backward pointers.

        Default uses streaming scan with O(KC) memory.

        Returns:
            v: (batch,) log partition values
            edges: edge marginals (for gradient computation)
            charts: intermediate DP tables
        """

    def _dp_scan_streaming(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,
        force_grad=False,
    ) -> Tuple[Tensor, List[Tensor], None]:
        """
        Streaming scan with O(KC) DP state (default).
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
    CheckpointShardSemiring,  # Sharded checkpointing (reduces O((KC)^3) peak)
)
```

## Triton fused streaming scan (~45x speedup)

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

def semi_crf_triton_forward(
    edge,                # (batch, T-1, K, C, C) potentials
    lengths,             # (batch,) sequence lengths
    use_triton=True,     # Use Triton kernel for inference
    validate=False,      # Use float64 PyTorch for validation
    semiring="log",      # "log" (partition) or "max" (Viterbi)
    use_compile=True,    # Use torch.compile for training
) -> Tensor:
    """
    Compute Semi-Markov CRF forward scan with hybrid optimization.

    Uses a hybrid approach for optimal performance:
    - Inference (requires_grad=False): Custom Triton kernel (~45x faster)
    - Training (requires_grad=True): torch.compile for efficient backward

    Returns:
        partition: (batch,) log partition function or Viterbi score
    """
```

**Hybrid routing:**

| Context | Execution Path |
|---------|----------------|
| `requires_grad=False` + CUDA | Custom Triton kernel |
| `requires_grad=True` + CUDA | `torch.compile` (automatic backward) |
| CPU or Triton unavailable | PyTorch reference |

**Example usage:**

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

# GPU inference: fast custom Triton kernel
partition = semi_crf_triton_forward(edge.cuda(), lengths.cuda())

# GPU training: torch.compile for efficient backward
edge_train = edge.cuda().requires_grad_(True)
partition = semi_crf_triton_forward(edge_train, lengths.cuda())
partition.sum().backward()

# Viterbi score (max semiring)
viterbi = semi_crf_triton_forward(edge.cuda(), lengths.cuda(), semiring="max")
```

See [Backends and Triton kernel](backends.md) for detailed behavior and options.
