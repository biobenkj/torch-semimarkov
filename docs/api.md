# API reference

## SemiMarkov

```python
class SemiMarkov(semiring):
    def logpartition(
        self,
        edge,                    # (batch, T-1, K, C, C) potentials
        lengths=None,            # (batch,) sequence lengths
        use_linear_scan=True,    # Use O(T) linear scan (default: auto-select)
        use_vectorized=True,     # Vectorize inner loops
        use_banded=False,        # Use banded matrix operations
        banded_perm="auto",      # Permutation strategy
        banded_bw_ratio=0.6,     # Bandwidth threshold
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute log partition function and backward pointers.

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
        Streaming scan with O(KC) DP state.
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

## Triton fused streaming scan

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

partition = semi_crf_triton_forward(edge, lengths, use_triton=True)
```
