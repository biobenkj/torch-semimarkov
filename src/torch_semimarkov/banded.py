r"""Lightweight banded matrix helpers (CPU/PyTorch only).

This module provides a pure-PyTorch implementation of banded matrix operations
that mirrors the public interface of ``genbmm.BandedMatrix``. It enables
prototyping banded Semi-Markov algorithms without requiring the genbmm CUDA
extension.

.. note::
    This implementation is intentionally simple and slower than CUDA kernels.
    For production GPU workloads, use the genbmm CUDA extension when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BandedMatrix:
    r"""Banded matrix representation for memory-efficient sparse operations.

    Stores only the non-zero diagonals of a banded matrix in a compact format.
    Supports log-semiring and max-semiring matrix multiplication.

    Args:
        data (Tensor): Banded data of shape :math:`(\text{batch}, n, \text{lu}+\text{ld}+1)`.
        lu (int): Number of upper diagonals (above main diagonal).
        ld (int): Number of lower diagonals (below main diagonal).
        fill (float, optional): Fill value for out-of-band elements. Default: ``0.0``

    Attributes:
        width (int): Total bandwidth (lu + ld + 1).

    Examples::

        >>> # Create banded matrix from dense
        >>> dense = torch.randn(2, 10, 10)
        >>> banded = BandedMatrix.from_dense(dense, lu=2, ld=2)
        >>> banded.data.shape
        torch.Size([2, 10, 5])
        >>> # Convert back to dense
        >>> reconstructed = banded.to_dense()

    See Also:
        :func:`bandedlogbmm`: Convenience function for log-semiring banded matmul
    """

    data: torch.Tensor  # shape: (batch, n, lu+ld+1)
    lu: int
    ld: int
    fill: float = 0.0

    def __post_init__(self):
        assert self.data.dim() == 3, "BandedMatrix expects (batch, n, width)"
        assert self.data.size(-1) == self.lu + self.ld + 1, (
            "Width must equal lu + ld + 1 " f"(got {self.data.size(-1)} vs {self.lu + self.ld + 1})"
        )
        # cache width
        self.width = self.data.size(-1)

    @classmethod
    def from_dense(cls, dense: torch.Tensor, lu: int, ld: int, fill: float = 0.0):
        r"""from_dense(dense, lu, ld, fill=0.0) -> BandedMatrix

        Extract a banded view from a dense square matrix.

        Args:
            dense (Tensor): Dense matrix of shape :math:`(\text{batch}, n, n)`.
            lu (int): Number of upper diagonals to extract.
            ld (int): Number of lower diagonals to extract.
            fill (float, optional): Fill value for padding. Default: ``0.0``

        Returns:
            BandedMatrix: Banded representation with data shape
            :math:`(\text{batch}, n, \text{lu}+\text{ld}+1)`.
        """
        assert dense.dim() == 3, "Expected (batch, n, n) dense input"
        batch, n, _ = dense.shape
        width = lu + ld + 1
        band = torch.full((batch, n, width), fill, device=dense.device, dtype=dense.dtype)
        for diag_offset in range(-ld, lu + 1):
            idx = diag_offset + ld  # column index in band storage
            if diag_offset >= 0:
                # Upper diagonal: copy elements (i, i+diag_offset)
                if n - diag_offset > 0:
                    i = torch.arange(0, n - diag_offset, device=dense.device)
                    band[:, i, idx] = dense[:, i, i + diag_offset]
            else:
                # Lower diagonal: copy elements (j, j+diag_offset) where j >= -diag_offset
                if -diag_offset < n:
                    j = torch.arange(-diag_offset, n, device=dense.device)
                    band[:, j, idx] = dense[:, j, j + diag_offset]
        return cls(band, lu, ld, fill)

    def to_dense(self) -> torch.Tensor:
        r"""to_dense() -> Tensor

        Expand banded representation to dense square matrix.

        Returns:
            Tensor: Dense matrix of shape :math:`(\text{batch}, n, n)`.
        """
        batch, n, _ = self.data.shape
        dense = torch.full((batch, n, n), self.fill, device=self.data.device, dtype=self.data.dtype)
        for diag_offset in range(-self.ld, self.lu + 1):
            idx = diag_offset + self.ld
            if diag_offset >= 0:
                # Upper diagonal
                if n - diag_offset > 0:
                    i = torch.arange(0, n - diag_offset, device=self.data.device)
                    dense[:, i, i + diag_offset] = self.data[:, i, idx]
            else:
                # Lower diagonal
                if -diag_offset < n:
                    j = torch.arange(-diag_offset, n, device=self.data.device)
                    dense[:, j, j + diag_offset] = self.data[:, j, idx]
        return dense

    def transpose(self) -> BandedMatrix:
        r"""transpose() -> BandedMatrix

        Band-aware matrix transpose.

        Swaps upper and lower bandwidths while transposing the underlying data.

        Returns:
            BandedMatrix: Transposed banded matrix with ``lu`` and ``ld`` swapped.
        """
        batch, n, _ = self.data.shape
        new_lu, new_ld = self.ld, self.lu
        new_width = new_lu + new_ld + 1
        new_data = torch.full(
            (batch, n, new_width),
            self.fill,
            device=self.data.device,
            dtype=self.data.dtype,
        )

        for diag_offset in range(-new_ld, new_lu + 1):
            src_offset = -diag_offset
            src_idx = src_offset + self.ld
            dst_idx = diag_offset + new_ld

            if src_idx < 0 or src_idx >= self.data.size(-1):
                continue

            if diag_offset >= 0:
                i = torch.arange(0, n - diag_offset, device=self.data.device)
                new_data[:, i, dst_idx] = self.data[:, i + diag_offset, src_idx]
            else:
                i = torch.arange(-diag_offset, n, device=self.data.device)
                new_data[:, i, dst_idx] = self.data[:, i + diag_offset, src_idx]

        return BandedMatrix(new_data, new_lu, new_ld, self.fill)

    def band_shift(self, t: int) -> BandedMatrix:
        """Shift the band offsets (positive shifts upward)."""
        if t == 0:
            return self
        pad = torch.full(
            (self.data.size(0), self.data.size(1), abs(t)),
            self.fill,
            device=self.data.device,
            dtype=self.data.dtype,
        )
        if t > 0:
            shifted = torch.cat([self.data[:, :, t:], pad], dim=-1)
            lu, ld = self.lu + t, self.ld - t
        else:
            shifted = torch.cat([pad, self.data[:, :, :t]], dim=-1)
            lu, ld = self.lu + t, self.ld - t
        return BandedMatrix(shifted, lu, ld, self.fill)

    def col_shift(self, t: int) -> BandedMatrix:
        """Shift columns (time) with band metadata update."""
        if t == 0:
            return self
        pad = torch.full(
            (self.data.size(0), abs(t), self.data.size(2)),
            self.fill,
            device=self.data.device,
            dtype=self.data.dtype,
        )
        if t > 0:
            shifted = torch.cat([self.data[:, t:, :], pad], dim=1)
            return BandedMatrix(shifted, self.lu - t, self.ld + t, self.fill)
        shifted = torch.cat([pad, self.data[:, :t, :]], dim=1)
        return BandedMatrix(shifted, self.lu - t, self.ld + t, self.fill)

    def _new(self, lu: int, ld: int) -> torch.Tensor:
        width = lu + ld + 1
        return torch.full(
            (self.data.size(0), self.data.size(1), width),
            self.fill,
            device=self.data.device,
            dtype=self.data.dtype,
        )

    def _multiply_template(self, other: BandedMatrix, reduce_fn) -> BandedMatrix:
        """
        Generic banded multiplication template.
        reduce_fn takes (values, dim) -> reduction result (e.g., logsumexp or max).
        """
        assert self.data.shape[1] == other.data.shape[1], "Time dimension mismatch"
        n = self.data.shape[1]
        lu_out = self.lu + other.lu
        ld_out = self.ld + other.ld
        out_data = self._new(lu_out, ld_out)

        # Loop over output rows/cols within band; use vectorized torch ops where possible
        for i in range(n):
            for band_col in range(out_data.size(-1)):
                j = i + (band_col - ld_out)
                if j < 0 or j >= n:
                    continue

                # valid k indices within both bands
                # row i in self has diagonals from i-ld .. i+lu
                k_min = max(0, i - self.ld, j - other.lu)
                k_max = min(n - 1, i + self.lu, j + other.ld)
                if k_min > k_max:
                    # No valid k indices, output stays at fill value
                    continue

                # Create range - handle edge case where k_min == k_max
                if k_min == k_max:
                    k_range = torch.tensor([k_min], device=self.data.device)
                else:
                    k_range = torch.arange(k_min, k_max + 1, device=self.data.device)

                # fetch banded values
                a_idx = (k_range - i) + self.ld  # offsets into self.data
                b_idx = (j - k_range) + other.ld  # offsets into other.data
                a_vals = self.data[:, i, a_idx]
                b_vals = other.data[:, k_range, b_idx]
                vals = a_vals + b_vals
                out_data[:, i, band_col] = reduce_fn(vals, dim=-1)

        return BandedMatrix(out_data, lu_out, ld_out, self.fill)

    def multiply(self, other: BandedMatrix) -> BandedMatrix:
        r"""multiply(other) -> BandedMatrix

        Standard (sum-product) banded matrix multiplication.

        Args:
            other (BandedMatrix): Right operand with compatible dimensions.

        Returns:
            BandedMatrix: Product with bandwidth ``lu_out = lu1 + lu2``, ``ld_out = ld1 + ld2``.
        """
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.sum(x, dim=dim))

    def multiply_log(self, other: BandedMatrix) -> BandedMatrix:
        r"""multiply_log(other) -> BandedMatrix

        Log-semiring banded matrix multiplication using logsumexp.

        Computes :math:`C_{ij} = \log \sum_k \exp(A_{ik} + B_{kj})` efficiently
        using the banded structure.

        Args:
            other (BandedMatrix): Right operand with compatible dimensions.

        Returns:
            BandedMatrix: Product in log-space.
        """
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.logsumexp(x, dim=dim))

    def multiply_max(self, other: BandedMatrix) -> BandedMatrix:
        r"""multiply_max(other) -> BandedMatrix

        Max-semiring banded matrix multiplication (Viterbi).

        Computes :math:`C_{ij} = \max_k (A_{ik} + B_{kj})` efficiently
        using the banded structure.

        Args:
            other (BandedMatrix): Right operand with compatible dimensions.

        Returns:
            BandedMatrix: Max-product result.
        """
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.max(x, dim=dim)[0])


def bandedlogbmm(a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
    r"""bandedlogbmm(a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld) -> Tensor

    Log-semiring banded batch matrix multiplication.

    Convenience function mirroring the genbmm API surface for drop-in compatibility.

    Args:
        a (Tensor): Left operand banded data of shape :math:`(\text{batch}, n, a\_lu + a\_ld + 1)`.
        a_lu (int): Upper bandwidth of ``a``.
        a_ld (int): Lower bandwidth of ``a``.
        b (Tensor): Right operand banded data.
        b_lu (int): Upper bandwidth of ``b``.
        b_ld (int): Lower bandwidth of ``b``.
        o_lu (int): Expected output upper bandwidth (unused, for API compatibility).
        o_ld (int): Expected output lower bandwidth (unused, for API compatibility).

    Returns:
        Tensor: Banded data of the log-semiring product.
    """
    return BandedMatrix(a, a_lu, a_ld).multiply_log(BandedMatrix(b, b_lu, b_ld)).data
