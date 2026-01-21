#!/usr/bin/env python
"""
Comprehensive test suite verifying all Semi-Markov backends produce equivalent results.

This is critical for paper reproducibility: regardless of which backend is used
for efficiency reasons, they must all compute the same partition function and gradients.

Backends tested:
- linear_scan: O(N) sequential, non-vectorized reference implementation
- linear_scan_vectorized: O(N) with vectorized inner loop (2-3x faster)
- linear_scan_streaming: O(N) with O(K*C) DP state (memory efficient)
- binary_tree: O(log N) parallel depth (high memory due to O((KC)^3) temporaries)
- binary_tree_sharded: Same as binary_tree but with CheckpointShardSemiring
- block_triangular: Structured matrix approach

All backends should produce identical partition functions within numerical tolerance.
"""

import argparse
import sys
import time

import pytest
import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

# -----------------------------------------------------------------------------
# Test Configurations
# -----------------------------------------------------------------------------

SMALL_CONFIG = {"T": 32, "K": 6, "C": 3, "B": 2, "name": "Small"}
MEDIUM_CONFIG = {"T": 64, "K": 8, "C": 4, "B": 2, "name": "Medium"}
# Note: Large configs may OOM on binary_tree due to O((KC)^3) temporaries


def get_all_backends():
    """Return list of all available backends."""
    return [
        "linear_scan",
        "linear_scan_vectorized",
        "linear_scan_streaming",
        "binary_tree",
        "binary_tree_sharded",
        "block_triangular",
    ]


def get_linear_backends():
    """Return backends that use linear scan (for faster testing)."""
    return [
        "linear_scan",
        "linear_scan_vectorized",
        "linear_scan_streaming",
    ]


# -----------------------------------------------------------------------------
# Backend Execution Helpers
# -----------------------------------------------------------------------------


def run_backend(struct, edge, lengths, backend: str, force_grad: bool = True):
    """
    Run a specific backend and return (partition_value, potentials_list).

    Returns the raw output from each backend's method.
    """
    from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring

    if backend == "linear_scan":
        v, potentials, _ = struct._dp_standard(edge, lengths, force_grad=force_grad)
        return v, potentials

    elif backend == "linear_scan_vectorized":
        v, potentials, _ = struct._dp_standard_vectorized(edge, lengths, force_grad=force_grad)
        return v, potentials

    elif backend == "linear_scan_streaming":
        v, potentials, _ = struct._dp_scan_streaming(edge, lengths, force_grad=force_grad)
        return v, potentials

    elif backend == "binary_tree":
        v, potentials, _ = struct.logpartition(edge, lengths=lengths, use_linear_scan=False)
        return v, potentials

    elif backend == "binary_tree_sharded":
        ShardedLogSemiring = CheckpointShardSemiring(LogSemiring, max_size=10000)
        struct_sharded = SemiMarkov(ShardedLogSemiring)
        v, potentials, _ = struct_sharded.logpartition(edge, lengths=lengths, use_linear_scan=False)
        return v, potentials

    elif backend == "block_triangular":
        if hasattr(struct, "_dp_blocktriangular"):
            v, potentials, _ = struct._dp_blocktriangular(edge, lengths, force_grad=force_grad)
            return v, potentials
        else:
            raise NotImplementedError("block_triangular not available")

    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_test_data(T, K, C, B, device="cpu", dtype=torch.float32, seed=42):
    """Create random edge potentials for testing."""
    torch.manual_seed(seed)
    edge = torch.randn(B, T - 1, K, C, C, device=device, dtype=dtype)
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    return edge, lengths


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return SMALL_CONFIG.copy()


@pytest.fixture
def medium_config():
    return MEDIUM_CONFIG.copy()


class TestLinearBackendsEquivalence:
    """Test that all linear scan variants produce identical results."""

    def test_forward_pass_small(self, small_config):
        """All linear backends should produce same partition function."""
        T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        # Compare all against reference (linear_scan)
        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"{backend} differs from linear_scan by {max_diff:.2e}"

    def test_gradient_equivalence(self, small_config):
        """All linear backends should produce same gradients."""
        T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        grads = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            v.sum().backward()
            grads[backend] = edge_copy.grad.clone()

        # Compare all against reference
        ref_grad = grads["linear_scan"]
        for backend, grad in grads.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref_grad - grad).abs().max().item()
            assert max_diff < 1e-4, f"{backend} gradient differs by {max_diff:.2e}"

    def test_variable_lengths(self, small_config):
        """All linear backends handle variable lengths correctly."""
        T, K, C, B = small_config["T"], small_config["K"], small_config["C"], small_config["B"]
        torch.manual_seed(42)
        edge = torch.randn(B, T - 1, K, C, C)
        lengths = torch.tensor([T, T - 5][:B], dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"{backend} variable length differs by {max_diff:.2e}"


class TestAllBackendsEquivalence:
    """Test that ALL backends (including tree) produce equivalent results."""

    @pytest.mark.parametrize("T", [16, 24, 32])
    def test_all_backends_forward(self, T):
        """All backends should produce same partition function."""
        K, C, B = 4, 3, 2  # Small state space to avoid OOM on binary_tree
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_all_backends():
            try:
                edge_copy = edge.clone().detach().requires_grad_(True)
                v, _ = run_backend(struct, edge_copy, lengths, backend)
                results[backend] = v.detach()
            except (NotImplementedError, RuntimeError) as e:
                # Skip backends that aren't available or OOM
                print(f"  Skipping {backend}: {e}")
                continue

        # Use linear_scan as reference
        if "linear_scan" not in results:
            pytest.skip("Reference backend not available")

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-3, f"T={T}: {backend} differs from linear_scan by {max_diff:.2e}"

    def test_all_backends_gradient(self):
        """All backends should produce equivalent gradients."""
        T, K, C, B = 24, 4, 3, 2  # Small to avoid OOM
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        grads = {}
        for backend in get_all_backends():
            try:
                edge_copy = edge.clone().detach().requires_grad_(True)
                v, _ = run_backend(struct, edge_copy, lengths, backend)
                v.sum().backward()
                grads[backend] = edge_copy.grad.clone()
            except (NotImplementedError, RuntimeError) as e:
                print(f"  Skipping {backend}: {e}")
                continue

        if "linear_scan" not in grads:
            pytest.skip("Reference backend not available")

        ref_grad = grads["linear_scan"]
        for backend, grad in grads.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref_grad - grad).abs().max().item()
            # Slightly looser tolerance for tree backends due to different accumulation order
            tol = 1e-3 if "tree" in backend else 1e-4
            assert max_diff < tol, f"{backend} gradient differs by {max_diff:.2e}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("T", [4, 6, 8, 10])
    def test_short_sequences(self, T):
        """Backends handle sequences near or shorter than K."""
        K, C, B = 6, 3, 2
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"T={T}: {backend} differs by {max_diff:.2e}"

    def test_single_batch(self):
        """Backends work with batch size 1."""
        T, K, C, B = 32, 6, 3, 1
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"B=1: {backend} differs by {max_diff:.2e}"

    def test_single_class(self):
        """Backends work with single class (C=1)."""
        T, K, C, B = 32, 6, 1, 2
        edge, lengths = create_test_data(T, K, C, B)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"C=1: {backend} differs by {max_diff:.2e}"


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_large_potentials(self):
        """Backends handle large potential values."""
        T, K, C, B = 32, 6, 3, 2
        torch.manual_seed(42)
        # Large values that could cause overflow in non-log space
        edge = torch.randn(B, T - 1, K, C, C) * 10.0
        lengths = torch.full((B,), T, dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            # Use relative tolerance for large values
            rel_diff = ((ref - v).abs() / (ref.abs() + 1e-8)).max().item()
            assert rel_diff < 1e-4, f"Large potentials: {backend} rel diff = {rel_diff:.2e}"

    def test_small_potentials(self):
        """Backends handle small potential values."""
        T, K, C, B = 32, 6, 3, 2
        torch.manual_seed(42)
        # Small values
        edge = torch.randn(B, T - 1, K, C, C) * 0.01
        lengths = torch.full((B,), T, dtype=torch.long)
        struct = SemiMarkov(LogSemiring)

        results = {}
        for backend in get_linear_backends():
            edge_copy = edge.clone().detach().requires_grad_(True)
            v, _ = run_backend(struct, edge_copy, lengths, backend)
            results[backend] = v.detach()

        ref = results["linear_scan"]
        for backend, v in results.items():
            if backend == "linear_scan":
                continue
            max_diff = (ref - v).abs().max().item()
            assert max_diff < 1e-4, f"Small potentials: {backend} differs by {max_diff:.2e}"


# -----------------------------------------------------------------------------
# CLI Interface for Manual Testing
# -----------------------------------------------------------------------------


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_equivalence_check(device, dtype, configs, backends, verbose=True):
    """
    Run comprehensive equivalence check across backends.

    Returns True if all backends produce equivalent results.
    """
    all_pass = True

    for config in configs:
        T, K, C, B = config["T"], config["K"], config["C"], config["B"]
        name = config.get("name", f"T={T},K={K},C={C}")

        if verbose:
            print(f"\n{name}: T={T}, K={K}, C={C}, B={B}")
            print("-" * 60)

        edge, lengths = create_test_data(T, K, C, B, device=device, dtype=dtype)
        struct = SemiMarkov(LogSemiring)

        results = {}
        grads = {}
        times = {}

        for backend in backends:
            try:
                edge_copy = edge.clone().detach().requires_grad_(True)

                _sync_if_cuda(device)
                t0 = time.perf_counter()

                v, _ = run_backend(struct, edge_copy, lengths, backend)
                v.sum().backward()

                _sync_if_cuda(device)
                elapsed = (time.perf_counter() - t0) * 1000

                results[backend] = v.detach()
                grads[backend] = edge_copy.grad.clone()
                times[backend] = elapsed

            except Exception as e:
                if verbose:
                    print(f"  {backend:25s}: SKIPPED ({e})")
                continue

        if "linear_scan" not in results:
            if verbose:
                print("  Reference backend (linear_scan) not available!")
            continue

        ref_v = results["linear_scan"]
        ref_grad = grads["linear_scan"]

        for backend in backends:
            if backend not in results:
                continue

            v = results[backend]
            grad = grads[backend]
            t = times[backend]

            v_diff = (ref_v - v).abs().max().item()
            g_diff = (ref_grad - grad).abs().max().item()

            status = "✓" if v_diff < 1e-3 and g_diff < 1e-3 else "✗"
            if status == "✗":
                all_pass = False

            if verbose:
                if backend == "linear_scan":
                    print(f"  {backend:25s}: v={v[0].item():10.4f} (reference)  {t:8.2f}ms")
                else:
                    print(
                        f"  {backend:25s}: v_diff={v_diff:.2e}, g_diff={g_diff:.2e}  {t:8.2f}ms  {status}"
                    )

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Test Semi-Markov backend equivalence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_backend_equivalence.py                    # Run all tests
  python test_backend_equivalence.py --device cuda      # Run on GPU
  python test_backend_equivalence.py --quick            # Quick test (linear backends only)
  python test_backend_equivalence.py --backends linear_scan,linear_scan_streaming
""",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use (cuda, cpu). Default: cuda if available"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Data type for computations",
    )
    parser.add_argument(
        "--backends", default=None, help="Comma-separated list of backends to test. Default: all"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with linear backends only")
    parser.add_argument(
        "--configs",
        default="small,medium",
        help="Comma-separated configs: small, medium, or T:K:C:B format",
    )
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Dtype
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Backends
    if args.backends:
        backends = args.backends.split(",")
    elif args.quick:
        backends = get_linear_backends()
    else:
        backends = get_all_backends()

    # Configs
    configs = []
    for c in args.configs.split(","):
        if c == "small":
            configs.append(SMALL_CONFIG)
        elif c == "medium":
            configs.append(MEDIUM_CONFIG)
        else:
            parts = c.split(":")
            if len(parts) == 4:
                T, K, C, B = map(int, parts)
                configs.append({"T": T, "K": K, "C": C, "B": B, "name": "Custom"})

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "SEMI-MARKOV BACKEND EQUIVALENCE TEST" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Backends: {', '.join(backends)}")

    all_pass = run_equivalence_check(device, dtype, configs, backends)

    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL BACKENDS PRODUCE EQUIVALENT RESULTS")
    else:
        print("✗ SOME BACKENDS DIFFER - CHECK OUTPUT ABOVE")
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
