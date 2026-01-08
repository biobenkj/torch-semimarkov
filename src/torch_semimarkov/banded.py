"""
Lightweight banded matrix helpers (CPU/PyTorch only).

This mirrors the public interface of genbmm.BandedMatrix enough to let us
prototype a banded Semi-Markov path without requiring the genbmm CUDA
extension. The implementation is intentionally simple and slower than the
CUDA kernels; it is meant for correctness scaffolding and future wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BandedMatrix:
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
        """
        Extract a banded view from a dense square matrix.
        """
        assert dense.dim() == 3, "Expected (batch, n, n) dense input"
        batch, n, _ = dense.shape
        width = lu + ld + 1
        band = torch.full((batch, n, width), fill, device=dense.device, dtype=dense.dtype)
        for diag_offset in range(-ld, lu + 1):
            idx = diag_offset + lu  # column index in band storage
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
        """Expand to dense square matrix (batch, n, n)."""
        batch, n, _ = self.data.shape
        dense = torch.full((batch, n, n), self.fill, device=self.data.device, dtype=self.data.dtype)
        for diag_offset in range(-self.ld, self.lu + 1):
            idx = diag_offset + self.lu
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

    def transpose(self) -> "BandedMatrix":
        """Band-aware transpose (swap upper/lower bandwidths)."""
        # flip diagonals and swap lu/ld
        flipped = torch.flip(self.data, dims=[-1])
        return BandedMatrix(flipped, self.ld, self.lu, self.fill)

    def band_shift(self, t: int) -> "BandedMatrix":
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

    def col_shift(self, t: int) -> "BandedMatrix":
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

    def _multiply_template(self, other: "BandedMatrix", reduce_fn) -> "BandedMatrix":
        """
        Generic banded multiplication template.
        reduce_fn takes (values, dim) -> reduction result (e.g., logsumexp or max).
        """
        assert self.data.shape[1] == other.data.shape[1], "Time dimension mismatch"
        n = self.data.shape[1]
        lu_out = self.lu + other.ld
        ld_out = self.ld + other.lu
        out_data = self._new(lu_out, ld_out)

        # Loop over output rows/cols within band; use vectorized torch ops where possible
        for i in range(n):
            for band_col in range(out_data.size(-1)):
                j = i + (band_col - lu_out)
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
                a_idx = (k_range - i) + self.lu  # offsets into self.data
                b_idx = (j - k_range) + other.lu  # offsets into other.data
                a_vals = self.data[:, i, a_idx]
                b_vals = other.data[:, j, b_idx]
                vals = a_vals + b_vals
                out_data[:, i, band_col] = reduce_fn(vals, dim=-1)

        return BandedMatrix(out_data, lu_out, ld_out, self.fill)

    def multiply(self, other: "BandedMatrix") -> "BandedMatrix":
        """Standard (sum-product) banded matmul."""
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.sum(x, dim=dim))

    def multiply_log(self, other: "BandedMatrix") -> "BandedMatrix":
        """Log-semiring banded matmul (logsumexp)."""
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.logsumexp(x, dim=dim))

    def multiply_max(self, other: "BandedMatrix") -> "BandedMatrix":
        """Max-semiring banded matmul (Viterbi)."""
        return self._multiply_template(other, reduce_fn=lambda x, dim: torch.max(x, dim=dim)[0])


# simple factory to mirror genbmm API surface
def bandedlogbmm(a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
    return BandedMatrix(a, a_lu, a_ld).multiply_log(BandedMatrix(b, b_lu, b_ld)).data
