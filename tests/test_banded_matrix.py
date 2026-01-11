import torch

from torch_semimarkov.banded import BandedMatrix


def _band_mask(n, lu, ld):
    row = torch.arange(n).view(n, 1)
    col = torch.arange(n).view(1, n)
    return (col - row <= lu) & (row - col <= ld)


def _apply_band_fill(dense, lu, ld, fill):
    n = dense.size(-1)
    mask = _band_mask(n, lu, ld).to(dense.device)
    fill_tensor = torch.full_like(dense, fill)
    return torch.where(mask.unsqueeze(0), dense, fill_tensor)


def test_banded_from_dense_roundtrip():
    torch.manual_seed(0)
    batch, n = 2, 6
    lu, ld = 2, 1
    fill = 0.0

    mask = _band_mask(n, lu, ld)
    values = torch.randn(batch, n, n)
    dense = torch.where(mask.unsqueeze(0), values, torch.full_like(values, fill))

    bm = BandedMatrix.from_dense(dense, lu, ld, fill)
    roundtrip = bm.to_dense()

    assert torch.allclose(roundtrip, dense)


def test_banded_transpose_matches_dense():
    torch.manual_seed(1)
    batch, n = 1, 5
    lu, ld = 1, 2
    fill = 0.0

    mask = _band_mask(n, lu, ld)
    values = torch.randn(batch, n, n)
    dense = torch.where(mask.unsqueeze(0), values, torch.full_like(values, fill))

    bm = BandedMatrix.from_dense(dense, lu, ld, fill)
    transposed = bm.transpose().to_dense()
    expected = dense.transpose(-1, -2)

    assert torch.allclose(transposed, expected)


def test_band_shift_updates_data_and_metadata():
    torch.manual_seed(2)
    batch, n = 1, 4
    lu, ld = 2, 2
    fill = -0.5
    data = torch.randn(batch, n, lu + ld + 1)
    bm = BandedMatrix(data.clone(), lu, ld, fill)

    shifted = bm.band_shift(1)
    pad = torch.full((batch, n, 1), fill, device=data.device, dtype=data.dtype)
    expected_data = torch.cat([data[:, :, 1:], pad], dim=-1)

    assert shifted.lu == lu + 1
    assert shifted.ld == ld - 1
    assert torch.allclose(shifted.data, expected_data)

    shifted_neg = bm.band_shift(-1)
    pad_neg = torch.full((batch, n, 1), fill, device=data.device, dtype=data.dtype)
    expected_neg = torch.cat([pad_neg, data[:, :, :-1]], dim=-1)

    assert shifted_neg.lu == lu - 1
    assert shifted_neg.ld == ld + 1
    assert torch.allclose(shifted_neg.data, expected_neg)


def test_col_shift_updates_data_and_metadata():
    torch.manual_seed(3)
    batch, n = 1, 5
    lu, ld = 1, 1
    fill = 1.0
    data = torch.randn(batch, n, lu + ld + 1)
    bm = BandedMatrix(data.clone(), lu, ld, fill)

    shifted = bm.col_shift(1)
    pad = torch.full((batch, 1, data.size(2)), fill, device=data.device, dtype=data.dtype)
    expected = torch.cat([data[:, 1:, :], pad], dim=1)

    assert shifted.lu == lu - 1
    assert shifted.ld == ld + 1
    assert torch.allclose(shifted.data, expected)

    shifted_neg = bm.col_shift(-1)
    pad_neg = torch.full((batch, 1, data.size(2)), fill, device=data.device, dtype=data.dtype)
    expected_neg = torch.cat([pad_neg, data[:, :-1, :]], dim=1)

    assert shifted_neg.lu == lu + 1
    assert shifted_neg.ld == ld - 1
    assert torch.allclose(shifted_neg.data, expected_neg)


def test_banded_multiply_log_matches_dense():
    torch.manual_seed(4)
    batch, n = 1, 5
    lu, ld = 1, 1
    fill = -1e9

    mask = _band_mask(n, lu, ld)
    a_values = torch.randn(batch, n, n)
    b_values = torch.randn(batch, n, n)
    dense_a = torch.where(mask.unsqueeze(0), a_values, torch.full_like(a_values, fill))
    dense_b = torch.where(mask.unsqueeze(0), b_values, torch.full_like(b_values, fill))

    a_banded = BandedMatrix.from_dense(dense_a, lu, ld, fill)
    b_banded = BandedMatrix.from_dense(dense_b, lu, ld, fill)

    out_banded = a_banded.multiply_log(b_banded)
    out_dense = out_banded.to_dense()

    expected = torch.logsumexp(
        dense_a.unsqueeze(-1) + dense_b.unsqueeze(-3),
        dim=2,
    )
    expected = _apply_band_fill(expected, out_banded.lu, out_banded.ld, fill)

    assert torch.allclose(out_dense, expected, atol=1e-5)
