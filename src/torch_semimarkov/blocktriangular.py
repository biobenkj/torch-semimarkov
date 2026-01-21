r"""Block-triangular representation for Semi-Markov duration blocks.

This module provides a sparse block representation that exploits the duration
constraint :math:`k_1 + k_2 \leq \text{span}` in Semi-Markov binary tree algorithms.

Instead of materializing full :math:`(K \cdot C, K \cdot C)` matrices, only the
:math:`\frac{K(K+1)}{2} \cdot C \cdot C` blocks satisfying the constraint are stored.
This significantly reduces memory for mid-level tree nodes.

.. note::
    This module is intended for experimentation and correctness validation.
    Structure metadata is cached per ``(K, span, duration_mask_key, device)`` for
    efficient repeated calls with the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Module-level cache for structural metadata
# Key: (K, span, duration_mask_key, device)
# Value: dict with 'e_block_indices', 'triplets', 'e_ptr', 'sorted_idx'
_STRUCTURE_CACHE = {}


@dataclass
class BlockTriangularMatrix:
    r"""Block-triangular matrix over duration states.

    Stores a sparse representation of :math:`(K \cdot C, K \cdot C)` matrices where
    only blocks :math:`(k_1, k_2)` satisfying the duration constraint are materialized.

    Args:
        values (Tensor): Block values of shape :math:`(B, \text{num\_blocks}, C, C)`.
        block_indices (Tensor): Block coordinates of shape :math:`(\text{num\_blocks}, 2)`
            where each row is :math:`(k_1, k_2)`.
        K (int): Number of duration states.
        C (int): Number of labels/classes.
        duration_mask_key (tuple, optional): Hashable key for the duration mask used
            during construction. Used for structure caching. Default: ``None``

    Attributes:
        device: Device of the values tensor.
        batch_size (int): Batch dimension size.

    Examples::

        >>> dense = torch.randn(2, 12, 12)  # K=4, C=3
        >>> bt = BlockTriangularMatrix.from_dense(dense, K=4, C=3, span=4)
        >>> bt.values.shape
        torch.Size([2, 10, 3, 3])  # Only 10 blocks satisfy k1+k2 <= 4
    """

    values: torch.Tensor
    block_indices: torch.Tensor
    K: int
    C: int
    duration_mask_key: Optional[tuple[int, ...]] = None

    @property
    def device(self):
        return self.values.device

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @classmethod
    def from_dense(
        cls,
        dense: torch.Tensor,
        K: int,
        C: int,
        span: int,
        duration_mask: Optional[torch.Tensor] = None,
    ) -> BlockTriangularMatrix:
        r"""from_dense(dense, K, C, span, duration_mask=None) -> BlockTriangularMatrix

        Compress a dense matrix into block-triangular form.

        Extracts only blocks :math:`(k_1, k_2)` where :math:`k_1 + k_2 \leq \text{span}`
        (and optionally passing the duration mask).

        Args:
            dense (Tensor): Dense matrix of shape :math:`(B, N, N)` where :math:`N = K \cdot C`.
            K (int): Number of duration states.
            C (int): Number of labels/classes.
            span (int): Maximum span length for the duration constraint.
            duration_mask (Tensor, optional): Boolean mask of shape :math:`(K, K)` for
                additional block filtering. Default: ``None``

        Returns:
            BlockTriangularMatrix: Sparse block representation.
        """
        B, N, N2 = dense.shape
        assert N == N2 == K * C, f"Expected N = K*C, got N={N}, K*C={K*C}"
        device = dense.device
        mask_key = _duration_mask_key(duration_mask)

        block_indices = []
        blocks = []

        for k1 in range(K):
            for k2 in range(K):
                if k1 + k2 > span:
                    continue
                if duration_mask is not None and not bool(duration_mask[k1, k2]):
                    continue

                i_start = k1 * C
                j_start = k2 * C
                block = dense[:, i_start : i_start + C, j_start : j_start + C]
                block_indices.append((k1, k2))
                blocks.append(block)

        if not blocks:
            values = dense.new_zeros(B, 0, C, C)
            block_indices_tensor = dense.new_zeros(0, 2, dtype=torch.long, device=device)
            return cls(
                values=values,
                block_indices=block_indices_tensor,
                K=K,
                C=C,
                duration_mask_key=mask_key,
            )

        values = torch.stack(blocks, dim=1)  # [B, num_blocks, C, C]
        block_indices_tensor = dense.new_tensor(block_indices, dtype=torch.long, device=device)
        return cls(
            values=values,
            block_indices=block_indices_tensor,
            K=K,
            C=C,
            duration_mask_key=mask_key,
        )

    def to_dense(self, semiring=None) -> torch.Tensor:
        r"""to_dense(semiring=None) -> Tensor

        Expand back to dense matrix.

        Args:
            semiring (optional): If provided, absent blocks are filled with
                ``semiring.zero.item()``. Otherwise filled with ``0.0``.

        Returns:
            Tensor: Dense matrix of shape :math:`(B, N, N)` where :math:`N = K \cdot C`.

        .. note::
            For log-space operations, pass the semiring to ensure absent blocks
            are filled with ``-inf`` instead of ``0.0``.
        """
        B = self.values.shape[0]
        N = self.K * self.C

        fill_value = semiring.zero.item() if semiring is not None else 0.0
        dense = self.values.new_full((B, N, N), fill_value=fill_value)
        for b, (k1, k2) in enumerate(self.block_indices.tolist()):
            i_start = k1 * self.C
            j_start = k2 * self.C
            dense[:, i_start : i_start + self.C, j_start : j_start + self.C] = self.values[:, b]
        return dense


def _build_block_index(
    block_indices: torch.Tensor, K: int, duration_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Map (k1, k2) -> block index.
    """
    device = block_indices.device
    idx = block_indices.new_full((K, K), -1, device=device)
    if block_indices.numel() == 0:
        return idx
    for b, (k1, k2) in enumerate(block_indices.tolist()):
        idx[k1, k2] = b
    if duration_mask is not None:
        # Ensure masked-out blocks are treated as invalid even if present in block_indices
        idx = idx.masked_fill(~duration_mask, -1)
    return idx


def _build_triplets(
    idxC: torch.Tensor,
    idxD: torch.Tensor,
    K: int,
    span: int,
    c_zero_idx: int,
    d_zero_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build result block indices and contributing block triplets:
      E_{k1,k3} = sum_k2 C_{k1,k2} @ D_{k2,k3}
    """
    device = idxC.device
    e_block_indices = []
    triplets = []

    for k1 in range(K):
        for k3 in range(K):
            if k1 + k3 > span:
                continue
            # Always materialize the output block (even if there are no contributors)
            e_idx = len(e_block_indices)
            e_block_indices.append((k1, k3))
            # Limit k2 search space using span constraints to avoid scanning impossible pairs
            max_k2 = min(K - 1, span - k1, span - k3)
            if max_k2 < 0:
                max_k2 = -1  # no valid k2 values, but keep e_block_indices entry
            for k2 in range(max_k2 + 1):
                c_b = int(idxC[k1, k2].item())
                d_b = int(idxD[k2, k3].item())

                if c_b < 0:
                    c_b = c_zero_idx  # implicit zero block
                if d_b < 0:
                    d_b = d_zero_idx  # implicit zero block

                triplets.append((e_idx, c_b, d_b))

    if e_block_indices:
        e_block_indices_tensor = torch.tensor(e_block_indices, dtype=torch.long, device=device)
    else:
        e_block_indices_tensor = torch.zeros(0, 2, dtype=torch.long, device=device)

    if triplets:
        triplets_tensor = torch.tensor(triplets, dtype=torch.long, device=device)
    else:
        triplets_tensor = torch.zeros(0, 3, dtype=torch.long, device=device)
    return e_block_indices_tensor, triplets_tensor


def _duration_mask_key(duration_mask: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    """
    Convert duration_mask to a hashable key.

    Returns None if mask is None, otherwise returns a tuple of flat indices where mask is True.
    """
    if duration_mask is None:
        return None
    # Convert to tuple of indices where True
    indices = torch.nonzero(duration_mask, as_tuple=False).flatten().tolist()
    return tuple(indices)


def _duration_mask_from_key(
    duration_mask_key: Optional[tuple[int, ...]],
    K: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Reconstruct a duration mask tensor from its cached key representation.
    """
    if duration_mask_key is None:
        return None
    if len(duration_mask_key) == 0:
        return torch.zeros(K, K, dtype=torch.bool, device=device)
    key_tensor = torch.tensor(duration_mask_key, dtype=torch.long, device=device)
    if key_tensor.numel() % 2 != 0:
        raise ValueError(f"Invalid duration_mask_key length: {len(duration_mask_key)}")
    coords = key_tensor.view(-1, 2)
    mask = torch.zeros(K, K, dtype=torch.bool, device=device)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _get_or_build_structure(
    C_bt: BlockTriangularMatrix,
    D_bt: BlockTriangularMatrix,
    K: int,
    span: int,
) -> dict:
    """
    Get cached structural metadata or build it if not cached.

    Returns dict with keys: e_block_indices, triplets, e_ptr, sorted_idx

    The structure only depends on (K, span, duration_mask_key, device), not on actual values,
    so we can cache and reuse it across matmul calls with the same configuration.
    """
    device = C_bt.device
    # Both operands must have been built with the same duration mask
    if C_bt.duration_mask_key != D_bt.duration_mask_key:
        raise ValueError(
            "C_bt and D_bt must be built with the same duration_mask; "
            f"got {C_bt.duration_mask_key} vs {D_bt.duration_mask_key}"
        )

    mask_key = C_bt.duration_mask_key
    cache_key = (K, span, mask_key, str(device))

    if cache_key in _STRUCTURE_CACHE:
        return _STRUCTURE_CACHE[cache_key]

    duration_mask = _duration_mask_from_key(mask_key, K, device)

    # Build structure from scratch
    idxC = _build_block_index(C_bt.block_indices, K, duration_mask=duration_mask)
    idxD = _build_block_index(D_bt.block_indices, K, duration_mask=duration_mask)

    c_zero_idx = C_bt.block_indices.shape[0]
    d_zero_idx = D_bt.block_indices.shape[0]

    e_block_indices, triplets = _build_triplets(
        idxC, idxD, K, span, c_zero_idx=c_zero_idx, d_zero_idx=d_zero_idx
    )

    num_e_blocks = e_block_indices.shape[0]
    num_triplets = triplets.shape[0]

    if num_e_blocks == 0:
        structure = {
            "e_block_indices": e_block_indices,
            "triplets": triplets,
            "e_ptr": torch.zeros(1, dtype=torch.long, device=device),
            "sorted_idx": torch.zeros(0, dtype=torch.long, device=device),
            "c_zero_idx": c_zero_idx,
            "d_zero_idx": d_zero_idx,
        }
        _STRUCTURE_CACHE[cache_key] = structure
        return structure

    if num_triplets == 0:
        sorted_idx = torch.zeros(0, dtype=torch.long, device=device)
        e_ptr = torch.zeros(num_e_blocks + 1, dtype=torch.long, device=device)
    else:
        # Sort triplets by e_idx for CSR accumulation
        e_indices = triplets[:, 0]
        sorted_idx = torch.argsort(e_indices)

        # Build CSR pointers
        e_indices_sorted = e_indices[sorted_idx]
        e_ptr = torch.searchsorted(
            e_indices_sorted.contiguous(),
            torch.arange(num_e_blocks + 1, device=device, dtype=e_indices.dtype),
        )

    structure = {
        "e_block_indices": e_block_indices,
        "triplets": triplets,
        "e_ptr": e_ptr,
        "sorted_idx": sorted_idx,
        "c_zero_idx": c_zero_idx,
        "d_zero_idx": d_zero_idx,
    }

    _STRUCTURE_CACHE[cache_key] = structure
    return structure


def block_triang_matmul(
    C_bt: BlockTriangularMatrix,
    D_bt: BlockTriangularMatrix,
    semiring,
    span: int,
    debug: bool = False,
) -> BlockTriangularMatrix:
    r"""block_triang_matmul(C_bt, D_bt, semiring, span, debug=False) -> BlockTriangularMatrix

    Block-triangular semiring matrix multiplication.

    Computes :math:`E = C \otimes D` where :math:`\otimes` denotes semiring matrix
    multiplication, exploiting the block-triangular sparsity pattern.

    .. math::
        E_{k_1, k_3} = \bigoplus_{k_2} C_{k_1, k_2} \otimes D_{k_2, k_3}

    Args:
        C_bt (BlockTriangularMatrix): Left operand.
        D_bt (BlockTriangularMatrix): Right operand.
        semiring: Semiring defining the algebraic operations.
        span (int): Maximum span length for output constraint.
        debug (bool, optional): Print debug information. Default: ``False``

    Returns:
        BlockTriangularMatrix: Product matrix with blocks satisfying :math:`k_1 + k_3 \leq \text{span}`.

    .. warning::
        Input matrices must have invalid blocks filled with ``semiring.zero.item()``
        (not ``0.0``) to prevent contamination in log-space arithmetic.

    .. note::
        Structure metadata is cached per ``(K, span, duration_mask_key, device)`` for
        efficient repeated calls with the same configuration.
    """
    assert C_bt.K == D_bt.K, "K mismatch"
    assert C_bt.C == D_bt.C, "C mismatch"

    K, C_dim = C_bt.K, C_bt.C
    B = C_bt.batch_size

    if debug:
        print("\n=== block_triang_matmul DEBUG ===")
        print(f"K={K}, C={C_dim}, B={B}, span={span}")
        print(f"C_bt blocks: {C_bt.block_indices.tolist()}")
        print(f"D_bt blocks: {D_bt.block_indices.tolist()}")

    # Get cached structure or build it (OPTIMIZATION: caches based on
    # (K, span, duration_mask_key, device))
    # This avoids rebuilding triplets/CSR structure on every call
    structure = _get_or_build_structure(C_bt, D_bt, K, span)
    e_block_indices = structure["e_block_indices"]
    triplets = structure["triplets"]
    e_ptr = structure["e_ptr"]
    sorted_idx = structure["sorted_idx"]
    c_zero_idx = structure["c_zero_idx"]
    d_zero_idx = structure["d_zero_idx"]

    if debug:
        cache_key = (K, span, C_bt.duration_mask_key, str(C_bt.device))
        is_cached = cache_key in _STRUCTURE_CACHE
        print(
            f"\nStructure cache: {'HIT' if is_cached else 'MISS'} (cache size: {len(_STRUCTURE_CACHE)})"
        )
        print(f"Result blocks (k1,k3): {e_block_indices.tolist()}")
        print("Triplets (e_idx, c_idx, d_idx):")
        for t in triplets.tolist():
            e_idx, c_idx, d_idx = t
            k1_k2 = C_bt.block_indices[c_idx].tolist()
            k2_k3 = D_bt.block_indices[d_idx].tolist()
            k1_k3 = e_block_indices[e_idx].tolist()
            print(f"  {t} -> C{k1_k2} @ D{k2_k3} contributes to E{k1_k3}")

    if e_block_indices.numel() == 0:
        if debug:
            print("No valid result blocks - returning empty matrix")
        values = C_bt.values.new_zeros(B, 0, C_dim, C_dim)
        return BlockTriangularMatrix(values=values, block_indices=e_block_indices, K=K, C=C_dim)

    num_triplets = triplets.shape[0]
    num_e_blocks = e_block_indices.shape[0]

    if debug:
        print(f"\nnum_triplets={num_triplets}, num_e_blocks={num_e_blocks}")

    e_indices = triplets[:, 0]
    c_indices = triplets[:, 1]
    d_indices = triplets[:, 2]

    if c_zero_idx != C_bt.values.shape[1] or d_zero_idx != D_bt.values.shape[1]:
        raise ValueError("Cached zero index does not match block count for inputs")

    zero_block_left = C_bt.values.new_full((B, 1, C_dim, C_dim), fill_value=semiring.zero.item())
    zero_block_right = D_bt.values.new_full((B, 1, C_dim, C_dim), fill_value=semiring.zero.item())

    C_values_padded = torch.cat([C_bt.values, zero_block_left], dim=1)
    D_values_padded = torch.cat([D_bt.values, zero_block_right], dim=1)

    left_blocks = C_values_padded[:, c_indices]  # [B, T, C, C]
    right_blocks = D_values_padded[:, d_indices]  # [B, T, C, C]

    BT = B * num_triplets
    left_bt = left_blocks.reshape(BT, C_dim, C_dim)
    right_bt = right_blocks.reshape(BT, C_dim, C_dim)

    # CRITICAL BUG FIX: Must convert to semiring representation before calling semiring.matmul
    # Original bug: Called semiring.matmul directly on raw tensors, which gave incorrect results
    # for LogSemiring (and other semirings). The semiring expects to work with converted tensors.
    left_sem = semiring.convert(left_bt)
    right_sem = semiring.convert(right_bt)

    prod_sem = semiring.matmul(left_sem, right_sem)  # [ssize, B*T, C, C]
    prod_bt = semiring.unconvert(prod_sem)  # [B*T, C, C]
    prod = prod_bt.reshape(B, num_triplets, C_dim, C_dim)

    if debug:
        print(f"\nProduct blocks computed, shape: {prod.shape}")
        print(f"Sample product values (batch 0, triplet 0):\n{prod[0, 0]}")

    # OPTIMIZATION: Use cached sorted indices and CSR pointers
    # sorted_idx and e_ptr were computed once and cached in _get_or_build_structure
    prod_sorted = prod[:, sorted_idx]  # [B, T, C, C]

    if debug:
        e_indices_sorted = e_indices[sorted_idx]
        print("\n=== CSR accumulation structure (from cache) ===")
        print(
            f"Sorted e_indices: {e_indices_sorted.tolist()[:20]}{'...' if num_triplets > 20 else ''}"
        )
        print(f"e_ptr (CSR pointers): {e_ptr.tolist()}")
        for e in range(min(5, num_e_blocks)):
            start, end = e_ptr[e].item(), e_ptr[e + 1].item()
            print(f"  Block {e}: triplets [{start}:{end}] ({end-start} contributions)")

    # Initialize result blocks with semiring zero (must cover all e_blocks, even if num_triplets=0)
    e_values = C_bt.values.new_full(
        (B, num_e_blocks, C_dim, C_dim), fill_value=semiring.zero.item()
    )

    if debug:
        print(f"\nInitialized e_values with semiring.zero = {semiring.zero.item()}")
        print(f"e_values shape: {e_values.shape}")
        print("\n=== Accumulating contributions (vectorized) ===")

    # Accumulate contributions per result block using vectorized semiring.sum
    # This replaces the Python loop over triplets with a loop over result blocks,
    # dramatically reducing overhead (num_e_blocks << num_triplets typically)
    for e in range(num_e_blocks):
        start, end = e_ptr[e].item(), e_ptr[e + 1].item()
        if start == end:
            # No contributions to this block (shouldn't happen with current triplet building)
            continue

        # Get all triplet products contributing to block e: [B, num_contrib, C, C]
        contrib = prod_sorted[:, start:end]

        if debug and e < 3:
            print(f"\nBlock {e}: reducing {end-start} contributions")
            print(f"  contrib shape: {contrib.shape}")
            if end - start == 1:
                print("  Single contribution (no reduction needed)")
            else:
                print("  Using semiring.sum over dim=1")

        # Reduce with semiring.sum: [B, num_contrib, C, C] -> [B, C, C]
        # StdSemiring: torch.sum(contrib, dim=1)
        # LogSemiring: torch.logsumexp(contrib, dim=1)
        # MaxSemiring: torch.max(contrib, dim=1)[0]
        if end - start == 1:
            # Single contribution - no reduction needed
            e_values[:, e] = contrib[:, 0]
        else:
            # Multiple contributions - use vectorized semiring.sum
            e_values[:, e] = semiring.sum(contrib, dim=1)

        if debug and e < 3:
            print(f"  Result e_values[0, {e}]:\n{e_values[0, e]}")

    if debug:
        print("\n=== Final result ===")
        print(f"Result block indices: {e_block_indices.tolist()}")
        print(f"Result values shape: {e_values.shape}")
        for i in range(min(3, num_e_blocks)):
            print(f"Block {i} ({e_block_indices[i].tolist()}):\n{e_values[0, i]}")

    return BlockTriangularMatrix(
        values=e_values,
        block_indices=e_block_indices,
        K=K,
        C=C_dim,
        duration_mask_key=C_bt.duration_mask_key,
    )


def clear_structure_cache():
    r"""clear_structure_cache()

    Clear the structure metadata cache.

    Call this to free memory after processing many different ``(K, span)``
    configurations, or to ensure fresh structure computation in tests.
    """
    _STRUCTURE_CACHE.clear()


def get_structure_cache_info() -> dict:
    r"""get_structure_cache_info() -> dict

    Get information about the structure cache.

    Returns:
        dict: Dictionary with keys:

        - **size** (int): Number of cached configurations.
        - **keys** (list): List of ``(K, span, mask_key, device)`` cache keys.
    """
    return {
        "size": len(_STRUCTURE_CACHE),
        "keys": list(_STRUCTURE_CACHE.keys()),
    }


__all__ = [
    "BlockTriangularMatrix",
    "block_triang_matmul",
    "clear_structure_cache",
    "get_structure_cache_info",
]
