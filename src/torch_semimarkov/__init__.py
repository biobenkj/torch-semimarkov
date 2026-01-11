"""
torch-semimarkov: Efficient Semi-Markov CRF Inference for PyTorch

This package provides optimized implementations of Semi-Markov CRF inference
algorithms, benchmarked and documented in:

    "Practical Semi-Markov CRF Inference for Genomic Sequence Annotation"
    Benjamin K. Johnson (2026)

Key finding: Memory, not time, is the binding constraint. Vectorized linear scan
is universally applicable across all genomic parameter regimes.

Original pytorch-struct: https://github.com/harvardnlp/pytorch-struct
License: MIT (see LICENSE file)

Modifications:
- Added use_linear_scan parameter for memory-efficient Semi-Markov CRF computation
- Added vectorized linear scan implementation (2-3x speedup)
- Added banded and block-triangular backends for experimentation
- Optimized for genomic segmentation use cases
"""

from .banded import BandedMatrix
from .banded_utils import (
    apply_permutation,
    measure_effective_bandwidth,
    rcm_ordering_from_adjacency,
    snake_ordering,
)
from .blocktriangular import BlockTriangularMatrix, block_triang_matmul
from .semimarkov import SemiMarkov

__version__ = "0.1.0"

__all__ = [
    "SemiMarkov",
    "BandedMatrix",
    "BlockTriangularMatrix",
    "block_triang_matmul",
    "measure_effective_bandwidth",
    "snake_ordering",
    "rcm_ordering_from_adjacency",
    "apply_permutation",
]
