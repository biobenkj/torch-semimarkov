"""
Test that streaming linear scan produces correct partition functions.

This is critical for ensuring the core algorithm works correctly.
"""

import pytest
import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return {
        "T": 32,
        "K": 6,
        "C": 3,
        "B": 2,
    }


@pytest.fixture
def edge_and_lengths(small_config):
    """Create random edge potentials."""
    torch.manual_seed(42)
    T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]
    edge = torch.randn(B, T - 1, K, C, C)
    lengths = torch.full((B,), T, dtype=torch.long)
    return edge, lengths


def test_streaming_produces_finite_values(edge_and_lengths):
    """Verify streaming scan produces finite log partition values."""
    edge, lengths = edge_and_lengths
    struct = SemiMarkov(LogSemiring)

    edge_copy = edge.clone().detach().requires_grad_(True)
    v, _, _ = struct._dp_scan_streaming(edge_copy, lengths, force_grad=True)

    # Check values are finite
    assert torch.isfinite(v).all(), "Partition function contains non-finite values"

    # Check backward produces finite gradients
    v.sum().backward()
    assert torch.isfinite(edge_copy.grad).all(), "Gradients contain non-finite values"


def test_streaming_variable_lengths(small_config):
    """Verify streaming scan handles variable sequence lengths correctly."""
    torch.manual_seed(123)
    T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]

    edge = torch.randn(B, T - 1, K, C, C)
    # Variable lengths: some shorter than T
    lengths = torch.tensor([T, T - 5][:B], dtype=torch.long)

    struct = SemiMarkov(LogSemiring)

    edge_copy = edge.clone().detach().requires_grad_(True)
    v, _, _ = struct._dp_scan_streaming(edge_copy, lengths, force_grad=True)

    # Check values are finite
    assert torch.isfinite(v).all(), "Variable length partition contains non-finite values"

    # Check different lengths give different results
    # v has shape (ssize, batch) where ssize=1 for LogSemiring
    assert v[0, 0] != v[0, 1], "Different lengths should give different partition functions"


def test_streaming_short_sequences():
    """Verify streaming scan handles edge cases with short sequences."""
    torch.manual_seed(456)
    struct = SemiMarkov(LogSemiring)

    # Test with sequence length close to K
    for T in [4, 8, 12]:
        K, C, B = 6, 3, 2
        edge = torch.randn(B, T - 1, K, C, C)
        lengths = torch.full((B,), T, dtype=torch.long)

        edge_copy = edge.clone().detach().requires_grad_(True)
        v, _, _ = struct._dp_scan_streaming(edge_copy, lengths, force_grad=True)

        assert torch.isfinite(v).all(), f"Short sequence T={T} has non-finite values"


def test_streaming_batch_consistency():
    """Verify same input gives same output regardless of batch position."""
    torch.manual_seed(789)
    T, K, C = 32, 6, 3
    struct = SemiMarkov(LogSemiring)

    # Create single input
    edge_single = torch.randn(1, T - 1, K, C, C)
    lengths_single = torch.full((1,), T, dtype=torch.long)

    v_single, _, _ = struct._dp_scan_streaming(
        edge_single.clone().detach().requires_grad_(True),
        lengths_single,
    )

    # Create batch with duplicates
    edge_batch = edge_single.expand(4, -1, -1, -1, -1).clone()
    lengths_batch = torch.full((4,), T, dtype=torch.long)

    v_batch, _, _ = struct._dp_scan_streaming(
        edge_batch.clone().detach().requires_grad_(True),
        lengths_batch,
    )

    # All batch elements should have same partition function
    max_diff = (v_batch - v_single).abs().max().item()
    assert max_diff < 1e-5, f"Batch elements differ: max diff = {max_diff}"


def test_logpartition_api(edge_and_lengths):
    """Verify the main logpartition API works correctly."""
    edge, lengths = edge_and_lengths
    struct = SemiMarkov(LogSemiring)

    edge_copy = edge.clone().detach().requires_grad_(True)
    v, potentials, _ = struct.logpartition(edge_copy, lengths=lengths)

    # Check return types
    assert isinstance(v, torch.Tensor), "Partition should be a tensor"
    assert isinstance(potentials, list), "Potentials should be a list"
    # v has shape (ssize, batch) where ssize=1 for LogSemiring
    assert v.shape[-1] == edge.shape[0], "Partition should have batch in last dim"

    # Check finite
    assert torch.isfinite(v).all(), "Partition contains non-finite values"


if __name__ == "__main__":
    # Quick manual test
    torch.manual_seed(42)
    T, K, C, B = 32, 6, 3, 2
    edge = torch.randn(B, T - 1, K, C, C)
    lengths = torch.full((B,), T, dtype=torch.long)

    struct = SemiMarkov(LogSemiring)

    edge_copy = edge.clone().detach().requires_grad_(True)
    v, _, _ = struct._dp_scan_streaming(edge_copy, lengths, force_grad=True)

    print(f"Partition: {v}")
    print(f"Finite: {torch.isfinite(v).all()}")

    v.sum().backward()
    print(f"Grad finite: {torch.isfinite(edge_copy.grad).all()}")
