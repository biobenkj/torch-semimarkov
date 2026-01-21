import torch

from torch_semimarkov.banded import BandedMatrix
from torch_semimarkov.banded_utils import (
    apply_permutation,
    measure_effective_bandwidth,
    rcm_ordering_from_adjacency,
    snake_ordering,
)


def test_measure_effective_bandwidth_dense():
    n = 5
    adj = torch.zeros(n, n)
    adj[0, 2] = 1.0
    adj[4, 1] = 1.0

    bw = measure_effective_bandwidth(adj, fill_value=0.0)

    assert bw == 3


def test_measure_effective_bandwidth_banded():
    data = torch.zeros(1, 4, 4)
    bm = BandedMatrix(data, lu=2, ld=1, fill=0.0)

    bw = measure_effective_bandwidth(bm)

    assert bw == 2


def test_snake_ordering():
    K, C = 4, 2
    perm = snake_ordering(K, C)

    expected = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.long)
    assert torch.equal(perm, expected)


def test_rcm_ordering_from_adjacency():
    n = 6
    adj = torch.eye(n)
    perm, used = rcm_ordering_from_adjacency(adj)

    if used:
        assert perm.numel() == n
        assert torch.equal(torch.sort(perm).values, torch.arange(n))
    else:
        assert torch.equal(perm, torch.arange(n))


def test_apply_permutation_matches_manual():
    torch.manual_seed(0)
    n = 4
    potentials = torch.randn(2, n, n)
    perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)

    permuted = apply_permutation(potentials, perm)
    expected = potentials.index_select(-1, perm).index_select(-2, perm)

    assert torch.allclose(permuted, expected)
