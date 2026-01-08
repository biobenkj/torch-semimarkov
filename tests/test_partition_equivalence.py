"""
Test that all linear scan implementations produce equivalent partition functions.

This is critical for ensuring refactors don't break correctness.
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


def test_streaming_matches_vectorized(edge_and_lengths):
    """Verify streaming scan produces same log partition as vectorized scan."""
    edge, lengths = edge_and_lengths
    struct = SemiMarkov(LogSemiring)

    # Vectorized (current default)
    edge_v = edge.clone().detach().requires_grad_(True)
    v_vec, _, _ = struct._dp_standard_vectorized(edge_v, lengths, force_grad=True)

    # Streaming (new implementation)
    edge_s = edge.clone().detach().requires_grad_(True)
    v_stream, _, _ = struct._dp_scan_streaming(edge_s, lengths, force_grad=True)

    # Forward pass should match
    max_diff = (v_vec - v_stream).abs().max().item()
    assert max_diff < 1e-4, f"Forward pass mismatch: max diff = {max_diff}"

    # Backward pass should also match
    v_vec.sum().backward()
    v_stream.sum().backward()

    grad_diff = (edge_v.grad - edge_s.grad).abs().max().item()
    assert grad_diff < 1e-4, f"Backward pass mismatch: max grad diff = {grad_diff}"


def test_streaming_matches_original(edge_and_lengths):
    """Verify streaming scan produces same log partition as original (non-vectorized) scan."""
    edge, lengths = edge_and_lengths
    struct = SemiMarkov(LogSemiring)

    # Original (reference implementation)
    edge_o = edge.clone().detach().requires_grad_(True)
    v_orig, _, _ = struct._dp_standard(edge_o, lengths, force_grad=True)

    # Streaming
    edge_s = edge.clone().detach().requires_grad_(True)
    v_stream, _, _ = struct._dp_scan_streaming(edge_s, lengths, force_grad=True)

    max_diff = (v_orig - v_stream).abs().max().item()
    assert max_diff < 1e-4, f"Forward pass mismatch with original: max diff = {max_diff}"


def test_streaming_variable_lengths(small_config):
    """Verify streaming scan handles variable sequence lengths correctly."""
    torch.manual_seed(123)
    T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]

    edge = torch.randn(B, T - 1, K, C, C)
    # Variable lengths: some shorter than T
    lengths = torch.tensor([T, T - 5, T - 10, T][:B], dtype=torch.long)

    struct = SemiMarkov(LogSemiring)

    # Vectorized
    edge_v = edge.clone().detach().requires_grad_(True)
    v_vec, _, _ = struct._dp_standard_vectorized(edge_v, lengths, force_grad=True)

    # Streaming
    edge_s = edge.clone().detach().requires_grad_(True)
    v_stream, _, _ = struct._dp_scan_streaming(edge_s, lengths, force_grad=True)

    max_diff = (v_vec - v_stream).abs().max().item()
    assert max_diff < 1e-4, f"Variable length mismatch: max diff = {max_diff}"


def test_streaming_short_sequences():
    """Verify streaming scan handles edge cases with short sequences."""
    torch.manual_seed(456)
    struct = SemiMarkov(LogSemiring)

    # Test with sequence length close to K
    for T in [4, 8, 12]:
        K, C, B = 6, 3, 2
        edge = torch.randn(B, T - 1, K, C, C)
        lengths = torch.full((B,), T, dtype=torch.long)

        edge_v = edge.clone().detach().requires_grad_(True)
        v_vec, _, _ = struct._dp_standard_vectorized(edge_v, lengths, force_grad=True)

        edge_s = edge.clone().detach().requires_grad_(True)
        v_stream, _, _ = struct._dp_scan_streaming(edge_s, lengths, force_grad=True)

        max_diff = (v_vec - v_stream).abs().max().item()
        assert max_diff < 1e-4, f"Short sequence T={T} mismatch: max diff = {max_diff}"


if __name__ == "__main__":
    # Quick manual test
    torch.manual_seed(42)
    T, K, C, B = 32, 6, 3, 2
    edge = torch.randn(B, T - 1, K, C, C)
    lengths = torch.full((B,), T, dtype=torch.long)

    struct = SemiMarkov(LogSemiring)

    edge_v = edge.clone().detach().requires_grad_(True)
    v_vec, _, _ = struct._dp_standard_vectorized(edge_v, lengths, force_grad=True)

    edge_s = edge.clone().detach().requires_grad_(True)
    v_stream, _, _ = struct._dp_scan_streaming(edge_s, lengths, force_grad=True)

    print(f"Vectorized: {v_vec}")
    print(f"Streaming:  {v_stream}")
    print(f"Max diff:   {(v_vec - v_stream).abs().max().item():.2e}")

    v_vec.sum().backward()
    v_stream.sum().backward()
    print(f"Grad diff:  {(edge_v.grad - edge_s.grad).abs().max().item():.2e}")
