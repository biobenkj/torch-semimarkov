"""
genbmm: Generalized Batch Matrix Multiplication with CUDA Support

This subpackage provides CUDA-accelerated generalized batch matrix multiplication
operations supporting multiple semiring modes (log, max, sample, product-max).

The CUDA extension (_C) is optional; pure PyTorch fallbacks are used when unavailable.
"""

from .genmul import logbmm, maxbmm, prodmaxbmm, samplebmm
from .sparse import BandedMatrix, banddiag

__all__ = ["logbmm", "maxbmm", "prodmaxbmm", "samplebmm", "BandedMatrix", "banddiag"]
