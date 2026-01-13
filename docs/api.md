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

## Triton fused streaming scan (up to 45x speedup)

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

# Uses Triton automatically on CUDA, falls back to PyTorch on CPU
partition = semi_crf_triton_forward(edge.cuda(), lengths.cuda())
```
