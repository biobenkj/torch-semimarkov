import torch

from torch_semimarkov.banded import BandedMatrix
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


def test_logsemiring_banded_matmul_matches_banded():
    torch.manual_seed(4)
    batch, n = 1, 5
    lu, ld = 1, 1
    fill = -1e9

    a_dense = torch.randn(batch, n, n)
    b_dense = torch.randn(batch, n, n)
    a_band = BandedMatrix.from_dense(a_dense, lu, ld, fill)

    out = LogSemiring.matmul(a_band, b_dense)
    expected = BandedMatrix.from_dense(b_dense, lu, ld, fill).multiply_log(a_band.transpose())

    assert isinstance(out, BandedMatrix)
    assert torch.allclose(out.data, expected.data)


def test_stdsemiring_banded_matmul_matches_banded():
    torch.manual_seed(5)
    batch, n = 1, 4
    lu, ld = 1, 1
    fill = 0.0

    a_dense = torch.randn(batch, n, n)
    b_dense = torch.randn(batch, n, n)
    a_band = BandedMatrix.from_dense(a_dense, lu, ld, fill)

    out = StdSemiring.matmul(a_band, b_dense)
    expected = BandedMatrix.from_dense(b_dense, lu, ld, fill).multiply(a_band.transpose())

    assert isinstance(out, BandedMatrix)
    assert torch.allclose(out.data, expected.data)


def test_checkpoint_shard_semiring_matches_logsemiring():
    torch.manual_seed(6)
    batch, n = 2, 3
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=128)
    expected = LogSemiring.matmul(a, b)
    actual = ShardedLog.matmul(a, b)

    assert torch.allclose(actual, expected, atol=1e-6)
