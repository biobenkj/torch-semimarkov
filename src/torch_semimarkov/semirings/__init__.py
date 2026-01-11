"""
Semiring operations for structured prediction.

Vendored from pytorch-struct with optimizations.
"""

from .checkpoint import CheckpointSemiring, CheckpointShardSemiring
from .semirings import (
    CrossEntropySemiring,
    EntropySemiring,
    KLDivergenceSemiring,
    KMaxSemiring,
    LogSemiring,
    MaxSemiring,
    StdSemiring,
)

__all__ = [
    "LogSemiring",
    "MaxSemiring",
    "StdSemiring",
    "KMaxSemiring",
    "EntropySemiring",
    "CrossEntropySemiring",
    "KLDivergenceSemiring",
    "CheckpointSemiring",
    "CheckpointShardSemiring",
]
