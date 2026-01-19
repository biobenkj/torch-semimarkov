"""
torch-semimarkov: Efficient Semi-Markov CRF Inference for PyTorch

This package provides optimized implementations of Semi-Markov CRF inference
algorithms for genomic sequence annotation at chromosome scale (T=400K+).

Key Features:
- O(KC) memory streaming scan (independent of sequence length T)
- Golden Rule streaming API for on-the-fly edge computation
- Triton GPU acceleration for inference

Original pytorch-struct: https://github.com/harvardnlp/pytorch-struct
License: MIT (see LICENSE file)

Modifications:
- Added streaming linear scan with O(KC) memory
- Added Golden Rule streaming API for T=400K+ sequences
- Removed memory-intensive backends (banded, block-triangular)
- Optimized for genomic segmentation use cases
"""

from .semimarkov import SemiMarkov
from .streaming import (
    semi_crf_streaming_forward,
    SemiCRFStreaming,
    compute_edge_block_golden_rule,
)

__version__ = "0.2.0"

__all__ = [
    # Core API
    "SemiMarkov",
    # Streaming API (Golden Rule)
    "semi_crf_streaming_forward",
    "SemiCRFStreaming",
    "compute_edge_block_golden_rule",
]
