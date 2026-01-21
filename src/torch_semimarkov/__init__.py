"""
torch-semimarkov: Efficient Semi-Markov CRF Inference for PyTorch

This package provides optimized implementations of Semi-Markov CRF inference
algorithms for genomic sequence annotation at chromosome scale (T=400K+).

Key Features:
- O(KC) memory streaming scan (independent of sequence length T)
- Streaming API for on-the-fly edge computation via prefix-sum decomposition
- Triton GPU acceleration for inference

Original pytorch-struct: https://github.com/harvardnlp/pytorch-struct
License: MIT (see LICENSE file)

Modifications:
- Added streaming linear scan with O(KC) memory
- Added streaming API for T=400K+ sequences with on-the-fly edge computation
- Removed memory-intensive backends (banded, block-triangular)
- Optimized for genomic segmentation use cases
"""

from .duration import (
    CallableDuration,
    DurationDistribution,
    GeometricDuration,
    LearnedDuration,
    NegativeBinomialDuration,
    PoissonDuration,
    UniformDuration,
    create_duration_distribution,
)
from .nn import Segment, SemiMarkovCRFHead, ViterbiResult
from .semimarkov import SemiMarkov
from .streaming import (
    SemiCRFStreaming,
    compute_edge_block_streaming,
    semi_crf_streaming_forward,
)
from .uncertainty import UncertaintyMixin, UncertaintySemiMarkovCRFHead

__version__ = "0.2.0"

__all__ = [
    # Core API
    "SemiMarkov",
    # Streaming API
    "semi_crf_streaming_forward",
    "SemiCRFStreaming",
    "compute_edge_block_streaming",
    # Neural network modules
    "SemiMarkovCRFHead",
    "Segment",
    "ViterbiResult",
    # Uncertainty quantification
    "UncertaintyMixin",
    "UncertaintySemiMarkovCRFHead",
    # Duration distributions
    "DurationDistribution",
    "LearnedDuration",
    "GeometricDuration",
    "NegativeBinomialDuration",
    "PoissonDuration",
    "UniformDuration",
    "CallableDuration",
    "create_duration_distribution",
]
