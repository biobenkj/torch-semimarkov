import pytest
import torch

import torch_semimarkov.blocktriangular as bt
from torch_semimarkov.blocktriangular import (
    BlockTriangularMatrix,
    block_triang_matmul,
    clear_structure_cache,
)
from torch_semimarkov.semirings import LogSemiring


def _make_dense(batch, K, C, span, fill):
    torch.manual_seed(0)
    dense = torch.full((batch, K * C, K * C), fill)
    for k1 in range(K):
        for k2 in range(K):
            if k1 + k2 > span:
                continue
            i = slice(k1 * C, (k1 + 1) * C)
            j = slice(k2 * C, (k2 + 1) * C)
            dense[:, i, j] = torch.randn(batch, C, C)
    return dense


def _mask_output_blocks(dense, K, C, span, fill):
    masked = dense.clone()
    for k1 in range(K):
        for k3 in range(K):
            if k1 + k3 <= span:
                continue
            i = slice(k1 * C, (k1 + 1) * C)
            j = slice(k3 * C, (k3 + 1) * C)
            masked[:, i, j] = fill
    return masked


def test_blocktriangular_roundtrip():
    batch, K, C, span = 2, 3, 2, 2
    fill = LogSemiring.zero.item()

    dense = _make_dense(batch, K, C, span, fill)
    bt_matrix = BlockTriangularMatrix.from_dense(dense, K, C, span)
    roundtrip = bt_matrix.to_dense(semiring=LogSemiring)

    assert torch.allclose(roundtrip, dense)


def test_block_triang_matmul_matches_dense():
    batch, K, C, span = 1, 3, 2, 2
    fill = LogSemiring.zero.item()

    left_dense = _make_dense(batch, K, C, span, fill)
    right_dense = _make_dense(batch, K, C, span, fill)

    left_bt = BlockTriangularMatrix.from_dense(left_dense, K, C, span)
    right_bt = BlockTriangularMatrix.from_dense(right_dense, K, C, span)

    out_bt = block_triang_matmul(left_bt, right_bt, LogSemiring, span)
    out_dense = out_bt.to_dense(semiring=LogSemiring)

    expected = torch.logsumexp(
        left_dense.unsqueeze(-1) + right_dense.unsqueeze(-3),
        dim=2,
    )
    expected = _mask_output_blocks(expected, K, C, span, fill)

    assert torch.allclose(out_dense, expected, atol=1e-4)


def test_block_triang_matmul_mismatched_mask_raises():
    batch, K, C, span = 1, 3, 2, 2
    fill = LogSemiring.zero.item()

    dense = _make_dense(batch, K, C, span, fill)
    mask_left = torch.ones(K, K, dtype=torch.bool)
    mask_right = torch.zeros(K, K, dtype=torch.bool)

    left_bt = BlockTriangularMatrix.from_dense(dense, K, C, span, duration_mask=mask_left)
    right_bt = BlockTriangularMatrix.from_dense(dense, K, C, span, duration_mask=mask_right)

    with pytest.raises(ValueError):
        block_triang_matmul(left_bt, right_bt, LogSemiring, span)


def test_structure_cache_clear():
    clear_structure_cache()
    assert len(bt._STRUCTURE_CACHE) == 0

    batch, K, C, span = 1, 3, 2, 2
    fill = LogSemiring.zero.item()
    dense = _make_dense(batch, K, C, span, fill)
    left_bt = BlockTriangularMatrix.from_dense(dense, K, C, span)
    right_bt = BlockTriangularMatrix.from_dense(dense, K, C, span)

    _ = block_triang_matmul(left_bt, right_bt, LogSemiring, span)

    assert len(bt._STRUCTURE_CACHE) > 0
