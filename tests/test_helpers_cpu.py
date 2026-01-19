import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.helpers import _Struct
from torch_semimarkov.semirings import LogSemiring


def test_get_dimension_list_sets_grad():
    struct = _Struct(LogSemiring)
    a = torch.zeros(1, 2, 3)
    b = torch.ones(1, 2, 3)

    dims = struct._get_dimension([a, b])

    assert dims == a.shape
    assert a.requires_grad is True
    assert b.requires_grad is True


def test_make_chart_force_grad_respects_potentials():
    struct = _Struct(LogSemiring)

    potentials = torch.zeros(2, 3)
    chart = struct._make_chart(1, (2, 3), potentials, force_grad=True)
    assert chart[0].requires_grad is True

    potentials_with_grad = torch.zeros(2, 3, requires_grad=True)
    chart_with_grad = struct._make_chart(1, (2, 3), potentials_with_grad, force_grad=True)
    assert chart_with_grad[0].requires_grad is False


def test_raw_sum_and_marginals_shapes():
    torch.manual_seed(0)
    batch, T, K, C = 1, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)
    struct = SemiMarkov(LogSemiring)

    v_raw = struct.sum(edge, lengths=lengths, _raw=True)
    assert v_raw.shape == (1, batch)

    marg_raw = struct.marginals(edge, lengths=lengths, _raw=True)
    assert marg_raw.shape == (1, batch, T - 1, K, C, C)
