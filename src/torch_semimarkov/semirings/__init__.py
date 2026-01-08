"""
Semiring operations for structured prediction.

Vendored from pytorch-struct with optimizations.
"""

from .semirings import (
    LogSemiring,
    MaxSemiring,
    StdSemiring,
    KMaxSemiring,
    EntropySemiring,
    CrossEntropySemiring,
)
from .checkpoint import CheckpointSemiring, CheckpointShardSemiring

__all__ = [
    "LogSemiring",
    "MaxSemiring",
    "StdSemiring",
    "KMaxSemiring",
    "EntropySemiring",
    "CrossEntropySemiring",
    "CheckpointSemiring",
    "CheckpointShardSemiring",
]
