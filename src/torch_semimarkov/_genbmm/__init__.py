"""genbmm: Generalized Batch Matrix Multiplication with CUDA Support.

This subpackage provides CUDA-accelerated generalized batch matrix multiplication
operations supporting multiple semiring modes (log, max, sample, product-max).

The CUDA extension (_C) is optional; pure PyTorch fallbacks are used when unavailable.

Production-ready exports:

- ``logbmm``: Log-space matrix multiplication (logsumexp reduction)
- ``maxbmm``: Max-space matrix multiplication (max reduction)
- ``BandedMatrix``: Banded matrix representation
- ``banddiag``: Banded diagonal extraction

Experimental exports (not intended for production use):

- ``samplebmm``: Sampling-based matrix multiplication
- ``prodmaxbmm``: Product-max matrix multiplication
"""

from .genmul import logbmm, maxbmm, prodmaxbmm, samplebmm
from .sparse import BandedMatrix, banddiag

__all__ = ["logbmm", "maxbmm", "prodmaxbmm", "samplebmm", "BandedMatrix", "banddiag"]
