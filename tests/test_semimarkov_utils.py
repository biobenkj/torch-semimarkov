import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def test_sum_matches_logpartition():
    torch.manual_seed(0)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    sm = SemiMarkov(LogSemiring)
    v, _, _ = sm.logpartition(edge, lengths=lengths, use_linear_scan=True)
    total = sm.sum(edge, lengths=lengths)

    assert torch.allclose(total, LogSemiring.unconvert(v))


def test_marginals_shape_and_bounds():
    torch.manual_seed(1)
    batch, T, K, C = 2, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    sm = SemiMarkov(LogSemiring)
    marginals = sm.marginals(edge, lengths=lengths)

    assert marginals.shape == edge.shape
    assert torch.isfinite(marginals).all()
    assert (marginals >= -1e-4).all()
    assert (marginals <= 1.0 + 1e-4).all()


def test_to_parts_from_parts_roundtrip():
    sequence = torch.tensor([[0, -1, 1, -1, -1, 2]], dtype=torch.long)
    C, K = 3, 4

    edge = SemiMarkov.to_parts(sequence, extra=(C, K))
    recovered, extra = SemiMarkov.from_parts(edge)

    assert extra == (C, K)
    assert torch.equal(recovered, sequence)
