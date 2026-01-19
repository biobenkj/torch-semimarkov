"""
Tests for semiring implementations.

Verifies semiring operations produce correct results and maintain
invariants like associativity and identity elements.
"""

import torch

from torch_semimarkov.semirings import (
    CrossEntropySemiring,
    EntropySemiring,
    KLDivergenceSemiring,
    KMaxSemiring,
    LogSemiring,
    StdSemiring,
)
from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring


def test_kmax_semiring_roundtrip():
    torch.manual_seed(0)
    k = 3
    semiring = KMaxSemiring(k)
    x = torch.randn(2, 4)

    converted = semiring.convert(x)
    assert converted.shape == (k, 2, 4)

    roundtrip = semiring.unconvert(converted)
    assert torch.allclose(roundtrip, x)


def test_entropy_semiring_sum_shape():
    torch.manual_seed(1)
    x = torch.randn(3, 5)

    converted = EntropySemiring.convert(x)
    summed = EntropySemiring.sum(converted, dim=-1)
    entropy = EntropySemiring.unconvert(summed)

    assert summed.shape == (2, 3)
    assert entropy.shape == (3,)
    assert torch.isfinite(entropy).all()


def test_kl_divergence_semiring_zero_for_equal():
    torch.manual_seed(2)
    logp = torch.randn(2, 6)
    converted = KLDivergenceSemiring.convert([logp, logp])
    summed = KLDivergenceSemiring.sum(converted, dim=-1)
    kl = KLDivergenceSemiring.unconvert(summed)

    assert torch.isfinite(kl).all()
    assert kl.abs().max().item() < 1e-3


def test_cross_entropy_semiring_non_negative():
    torch.manual_seed(3)
    logp = torch.randn(2, 6)
    converted = CrossEntropySemiring.convert([logp, logp])
    summed = CrossEntropySemiring.sum(converted, dim=-1)
    ce = CrossEntropySemiring.unconvert(summed)

    assert torch.isfinite(ce).all()
    assert ce.min().item() > -1e-4


def test_logsemiring_matmul():
    """Test LogSemiring matrix multiplication."""
    torch.manual_seed(4)
    batch, n = 2, 4
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    result = LogSemiring.matmul(a, b)

    assert result.shape == (batch, n, n)
    assert torch.isfinite(result).all()


def test_stdsemiring_matmul():
    """Test StdSemiring matrix multiplication."""
    torch.manual_seed(5)
    batch, n = 2, 4
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    result = StdSemiring.matmul(a, b)

    assert result.shape == (batch, n, n)
    assert torch.isfinite(result).all()


def test_checkpoint_shard_semiring_matches_logsemiring():
    torch.manual_seed(6)
    batch, n = 2, 3
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=128)
    expected = LogSemiring.matmul(a, b)
    actual = ShardedLog.matmul(a, b)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_logsemiring_sum():
    """Test LogSemiring sum (logsumexp)."""
    torch.manual_seed(7)
    x = torch.randn(3, 4, 5)

    result = LogSemiring.sum(x, dim=-1)
    expected = torch.logsumexp(x, dim=-1)

    assert torch.allclose(result, expected)


def test_logsemiring_identity():
    """Test LogSemiring identity elements."""
    # zero: additive identity (should be -inf or very negative)
    # one: multiplicative identity (should be 0 in log space)
    assert LogSemiring.zero < -1e4  # -100000.0
    assert LogSemiring.one == 0.0
