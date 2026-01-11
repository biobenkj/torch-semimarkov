"""
Utilities for banded Semi-Markov experiments:
- Measure effective bandwidth
- Simple duration permutations (snake)
- Optional Reverse Cuthill-McKee (RCM) ordering when SciPy is available
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    _has_scipy = True
except Exception:
    _has_scipy = False

try:
    from .banded import BandedMatrix
except Exception:
    BandedMatrix = None  # type: ignore


def measure_effective_bandwidth(
    adj: torch.Tensor,
    fill_value: Optional[float] = None,
) -> int:
    """
    Compute the maximum |i-j| where adjacency is non-fill.

    Works on:
    - Dense square torch.Tensor (batchless). If 3D, operates per batch and returns max across batch.
    - BandedMatrix (returns max(lu, ld)).
    """
    if BandedMatrix is not None and isinstance(adj, BandedMatrix):
        return max(adj.lu, adj.ld)

    if not torch.is_tensor(adj):
        raise TypeError("adj must be a torch.Tensor or BandedMatrix")

    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    if adj.dim() != 3 or adj.size(1) != adj.size(2):
        raise ValueError("adj must be (batch, n, n) or (n, n)")

    batch, n, _ = adj.shape
    # Determine fill_value if not provided
    if fill_value is None:
        # Heuristic: use +inf or -inf if present, else 0
        if torch.isinf(adj).any():
            fill_value = float("inf") if (adj == float("inf")).any() else float("-inf")
        else:
            fill_value = 0.0

    max_bw = 0
    for b in range(batch):
        mask = adj[b] != fill_value
        rows, cols = torch.nonzero(mask, as_tuple=True)
        if rows.numel() == 0:
            continue
        diff = torch.abs(rows - cols)
        max_bw = max(max_bw, int(diff.max().item()))
    return max_bw


def snake_ordering(K: int, C: int) -> torch.Tensor:
    """
    Simple low-high interleaving over durations (k), then labels (c) in ascending order.
    Returns a permutation over flattened states (k-major: idx = k*C + c).
    """
    ks = []
    low, high = 0, K - 1
    toggle = True
    while low <= high:
        if toggle:
            ks.append(low)
            low += 1
        else:
            ks.append(high)
            high -= 1
        toggle = not toggle
    perm = []
    for k in ks:
        for c in range(C):
            perm.append(k * C + c)
    return torch.tensor(perm, dtype=torch.long)


def rcm_ordering_from_adjacency(adj: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Reverse Cuthill-McKee ordering for a square dense adjacency.

    Returns (perm, used_scipy). If SciPy is unavailable, returns identity and False.
    """
    if not _has_scipy:
        n = adj.size(0)
        return torch.arange(n, dtype=torch.long), False

    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adj must be (n, n)")

    # Treat as undirected: symmetrize
    adj_sym = adj.cpu()
    adj_sym = torch.logical_or(adj_sym != 0, adj_sym.t() != 0).to(torch.int32)
    mat = csr_matrix(adj_sym.numpy())
    perm_np = reverse_cuthill_mckee(mat, symmetric_mode=True)
    perm = torch.from_numpy(perm_np.astype("int64"))
    return perm, True


def apply_permutation(potentials: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Permute flattened state dimension of potentials shaped (..., K*C, K*C).
    """
    return potentials.index_select(-1, perm).index_select(-2, perm)
