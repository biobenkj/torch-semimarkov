r"""Utilities for banded Semi-Markov experiments.

This module provides tools for analyzing and optimizing banded matrix structures:

- Bandwidth measurement for sparsity analysis
- State permutation strategies (snake ordering, RCM)
- Permutation application utilities

.. note::
    Reverse Cuthill-McKee ordering requires SciPy to be installed. The module
    gracefully falls back to identity permutations when SciPy is unavailable.
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
    r"""measure_effective_bandwidth(adj, fill_value=None) -> int

    Compute the effective bandwidth of an adjacency matrix.

    The effective bandwidth is :math:`\max_{(i,j): A_{ij} \neq \text{fill}} |i - j|`,
    i.e., the maximum distance from the diagonal of any non-fill entry.

    Args:
        adj (Tensor or BandedMatrix): Adjacency matrix of shape :math:`(n, n)` or
            :math:`(\text{batch}, n, n)`. For BandedMatrix, returns ``max(lu, ld)``.
        fill_value (float, optional): Value representing empty/non-edges.
            Default: ``None`` (auto-detected from inf values or 0).

    Returns:
        int: Maximum distance from diagonal of any non-fill entry.

    Examples::

        >>> adj = torch.eye(5)
        >>> measure_effective_bandwidth(adj)
        0
        >>> adj[0, 4] = 1  # Add off-diagonal entry
        >>> measure_effective_bandwidth(adj)
        4
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
    r"""snake_ordering(K, C) -> Tensor

    Generate a snake ordering permutation for duration-label state space.

    Creates a low-high interleaving over durations, keeping labels in ascending
    order within each duration. This often reduces bandwidth for Semi-Markov
    structures where adjacent durations have similar transition patterns.

    Args:
        K (int): Number of durations.
        C (int): Number of labels/classes.

    Returns:
        Tensor: Permutation tensor of shape :math:`(K \cdot C,)` with dtype ``torch.long``.

    Examples::

        >>> perm = snake_ordering(4, 2)
        >>> perm  # Interleaves durations: 0, 3, 1, 2
        tensor([0, 1, 6, 7, 2, 3, 4, 5])
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
    r"""rcm_ordering_from_adjacency(adj) -> Tuple[Tensor, bool]

    Compute Reverse Cuthill-McKee ordering to minimize bandwidth.

    RCM is a graph-based permutation algorithm that reorders vertices to
    minimize the bandwidth of the adjacency matrix. The input is symmetrized
    before computing the ordering.

    Args:
        adj (Tensor): Square adjacency matrix of shape :math:`(n, n)`.

    Returns:
        Tuple[Tensor, bool]: A tuple containing:

        - **perm** (Tensor): Permutation tensor of shape :math:`(n,)`.
        - **used_scipy** (bool): ``True`` if SciPy was used, ``False`` if
          SciPy was unavailable (returns identity permutation).

    .. note::
        Requires SciPy for the actual RCM computation. Falls back to identity
        permutation when SciPy is not installed.
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
    r"""apply_permutation(potentials, perm) -> Tensor

    Apply a permutation to both dimensions of a matrix.

    Permutes the last two dimensions of the input tensor using the same
    permutation, maintaining matrix structure.

    Args:
        potentials (Tensor): Input tensor of shape :math:`(..., K \cdot C, K \cdot C)`.
        perm (Tensor): Permutation indices of shape :math:`(K \cdot C,)`.

    Returns:
        Tensor: Permuted tensor with same shape as input.

    Examples::

        >>> mat = torch.randn(2, 8, 8)
        >>> perm = snake_ordering(4, 2)
        >>> permuted = apply_permutation(mat, perm)
        >>> permuted.shape
        torch.Size([2, 8, 8])
    """
    return potentials.index_select(-1, perm).index_select(-2, perm)
