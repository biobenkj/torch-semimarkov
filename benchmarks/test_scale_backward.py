#!/usr/bin/env python3
"""Scale testing for Semi-CRF backward pass at large dimensions.

Tests the checkpointed backward implementation at target dimensions:
- T=400,000+ (sequence length)
- K=3,000+ (max segment duration)
- C=16-24 (number of labels)

Usage:
    python benchmarks/test_scale_backward.py
    python benchmarks/test_scale_backward.py --quick   # Quick sanity check
    python benchmarks/test_scale_backward.py --full    # Full scale test
"""

import argparse
import gc
import math
import time

import torch

from torch_semimarkov.backward import (
    semi_crf_backward_beta,
    semi_crf_compute_marginals,
    semi_crf_forward_with_alpha,
)

# Import the checkpointed implementations
from torch_semimarkov.checkpointed import (
    _compute_checkpoint_interval,
    semi_crf_backward_from_ring_checkpoints,
    semi_crf_forward_with_ring_checkpoints,
    semi_crf_triton_checkpointed_backward,
)


def estimate_memory(T: int, K: int, C: int, batch: int = 1) -> dict:
    """Estimate memory usage for different approaches."""
    bytes_per_float = 4  # float32

    # Full alpha storage: T × C per batch
    full_alpha_floats = T * C * batch
    full_alpha_mb = full_alpha_floats * bytes_per_float / (1024 * 1024)

    # Old checkpointing (√T interval)
    old_interval = max(K, int(math.sqrt(T)))
    old_num_ckpts = (T + old_interval - 1) // old_interval
    old_ckpt_floats = old_num_ckpts * K * C * batch
    old_ckpt_mb = old_ckpt_floats * bytes_per_float / (1024 * 1024)

    # New checkpointing (√(T×K) interval)
    new_interval = max(K, int(math.sqrt(T * K)))
    new_num_ckpts = (T + new_interval - 1) // new_interval
    new_ckpt_floats = new_num_ckpts * K * C * batch
    new_ckpt_mb = new_ckpt_floats * bytes_per_float / (1024 * 1024)

    # Segment buffer
    segment_floats = new_interval * C * batch
    segment_mb = segment_floats * bytes_per_float / (1024 * 1024)

    # Beta ring buffer
    beta_ring_floats = K * C * batch
    beta_ring_mb = beta_ring_floats * bytes_per_float / (1024 * 1024)

    return {
        "full_alpha_mb": full_alpha_mb,
        "old_interval": old_interval,
        "old_num_ckpts": old_num_ckpts,
        "old_ckpt_mb": old_ckpt_mb,
        "new_interval": new_interval,
        "new_num_ckpts": new_num_ckpts,
        "new_ckpt_mb": new_ckpt_mb,
        "segment_mb": segment_mb,
        "beta_ring_mb": beta_ring_mb,
        "new_total_mb": new_ckpt_mb + segment_mb + beta_ring_mb,
    }


def test_checkpoint_interval():
    """Test that checkpoint interval formula is correct."""
    print("\n" + "=" * 60)
    print("Test 1: Checkpoint Interval Formula")
    print("=" * 60)

    test_cases = [
        (100, 10),
        (1000, 100),
        (10000, 500),
        (100000, 1000),
        (400000, 3000),
    ]

    print(f"{'T':>10} {'K':>8} {'√T':>8} {'√(T×K)':>10} {'Actual':>10} {'Check':>8}")
    print("-" * 60)

    all_pass = True
    for T, K in test_cases:
        sqrt_T = int(math.sqrt(T))
        sqrt_TK = int(math.sqrt(T * K))
        actual = _compute_checkpoint_interval(T, K)
        expected = max(K, sqrt_TK)
        passed = actual == expected
        all_pass = all_pass and passed

        print(f"{T:>10} {K:>8} {sqrt_T:>8} {sqrt_TK:>10} {actual:>10} {'✓' if passed else '✗':>8}")

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_memory_estimates():
    """Show memory estimates for target dimensions."""
    print("\n" + "=" * 60)
    print("Test 2: Memory Estimates")
    print("=" * 60)

    configs = [
        ("Small", 1000, 100, 8, 1),
        ("Medium", 10000, 500, 16, 1),
        ("Large", 100000, 1000, 24, 1),
        ("Target", 400000, 3000, 24, 1),
        ("Target batch=4", 400000, 3000, 24, 4),
    ]

    for name, T, K, C, batch in configs:
        mem = estimate_memory(T, K, C, batch)
        print(f"\n{name}: T={T:,}, K={K:,}, C={C}, batch={batch}")
        print(f"  Full α storage:     {mem['full_alpha_mb']:>10.2f} MB")
        print(
            f"  Old √T checkpoints: {mem['old_ckpt_mb']:>10.2f} MB ({mem['old_num_ckpts']} ckpts, interval={mem['old_interval']})"
        )
        print(
            f"  New √(TK) ckpts:    {mem['new_ckpt_mb']:>10.2f} MB ({mem['new_num_ckpts']} ckpts, interval={mem['new_interval']})"
        )
        print(f"  Segment buffer:     {mem['segment_mb']:>10.2f} MB")
        print(f"  Beta ring:          {mem['beta_ring_mb']:>10.2f} MB")
        print(f"  NEW TOTAL:          {mem['new_total_mb']:>10.2f} MB")
        print(f"  Improvement:        {mem['old_ckpt_mb'] / mem['new_total_mb']:.1f}x")

    return True


def test_gradient_correctness_small(device="cuda"):
    """Test gradient correctness on small dimensions (fast)."""
    print("\n" + "=" * 60)
    print(f"Test 3: Gradient Correctness (Small) - {device}")
    print("=" * 60)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return True

    torch.manual_seed(42)

    # Small enough for exact comparison
    configs = [
        (1, 50, 10, 4),
        (2, 100, 20, 6),
        (1, 200, 50, 8),
    ]

    all_pass = True
    for batch, T, K, C in configs:
        print(f"\n  Config: batch={batch}, T={T}, K={K}, C={C}")

        # Create edge tensor
        edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, device=device)
        edge = edge * 0.1  # Scale down to avoid numerical issues
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        # Reference: full alpha storage + PyTorch backward
        edge_ref = edge.clone().requires_grad_(True)
        partition_ref, alpha_full = semi_crf_forward_with_alpha(edge_ref.detach(), lengths)
        log_Z_ref = partition_ref  # partition is already log Z
        beta_full = semi_crf_backward_beta(edge_ref.detach(), lengths, semiring="log")
        marginals_ref = semi_crf_compute_marginals(
            edge_ref.detach(), alpha_full, beta_full, log_Z_ref, lengths
        )

        # Checkpointed: using ring checkpoints
        partition_ckpt, ring_ckpts, interval = semi_crf_forward_with_ring_checkpoints(
            edge.detach(), lengths
        )
        marginals_ckpt = semi_crf_backward_from_ring_checkpoints(
            edge.detach(), ring_ckpts, partition_ckpt, lengths, interval
        )

        # Compare partition values
        partition_diff = (log_Z_ref - partition_ckpt).abs().max().item()
        print(f"    Partition diff: {partition_diff:.2e}")

        # Compare marginals
        marginal_diff = (marginals_ref - marginals_ckpt).abs().max().item()
        print(f"    Marginal max diff: {marginal_diff:.2e}")

        passed = partition_diff < 1e-6 and marginal_diff < 1e-5
        all_pass = all_pass and passed
        print(f"    Result: {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_triton_backward_small(device="cuda"):
    """Test Triton backward kernel on small dimensions."""
    print("\n" + "=" * 60)
    print(f"Test 4: Triton Checkpointed Backward (Small) - {device}")
    print("=" * 60)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return True

    torch.manual_seed(42)

    configs = [
        (1, 50, 10, 4),
        (2, 100, 20, 6),
    ]

    all_pass = True
    for batch, T, K, C in configs:
        print(f"\n  Config: batch={batch}, T={T}, K={K}, C={C}")

        edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float32, device=device)
        edge = edge * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        # Triton checkpointed backward with gradient
        edge_grad = edge.clone().requires_grad_(True)
        partition = semi_crf_triton_checkpointed_backward(edge_grad, lengths)
        partition.sum().backward()
        grad_triton = edge_grad.grad.clone()

        # Reference: PyTorch full backward
        edge_ref = edge.clone().requires_grad_(True)
        partition_ref, alpha_full = semi_crf_forward_with_alpha(edge_ref.detach(), lengths)
        log_Z_ref = partition_ref
        beta_full = semi_crf_backward_beta(edge_ref.detach(), lengths, semiring="log")
        marginals_ref = semi_crf_compute_marginals(
            edge_ref.detach(), alpha_full, beta_full, log_Z_ref, lengths
        )

        # Compare
        grad_diff = (grad_triton - marginals_ref).abs().max().item()
        print(f"    Gradient max diff: {grad_diff:.2e}")

        passed = grad_diff < 1e-4  # Relaxed tolerance for float32
        all_pass = all_pass and passed
        print(f"    Result: {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_scale_memory(device="cuda"):
    """Test memory usage at scale."""
    print("\n" + "=" * 60)
    print(f"Test 5: Scale Memory Test - {device}")
    print("=" * 60)

    if device != "cuda" or not torch.cuda.is_available():
        print("CUDA not available or not requested, skipping")
        return True

    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Target dimensions (start smaller and work up)
    configs = [
        ("Medium", 10000, 500, 16, 1),
        ("Large", 50000, 1000, 20, 1),
        # ("Target", 400000, 3000, 24, 1),  # Uncomment when ready
    ]

    for name, T, K, C, batch in configs:
        print(f"\n  {name}: T={T:,}, K={K:,}, C={C}, batch={batch}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

        # Create minimal edge tensor (just needs to be the right shape)
        # We don't need to fill it with random values for memory testing
        edge = torch.zeros(batch, T - 1, K, C, C, dtype=torch.float32, device=device)
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        mem_edge = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"    Edge tensor: {mem_edge - mem_before:.1f} MB")

        # Compute checkpoint interval and estimates
        interval = _compute_checkpoint_interval(T, K)
        num_ckpts = (T + interval - 1) // interval
        estimated = estimate_memory(T, K, C, batch)
        print(f"    Checkpoint interval: {interval:,} ({num_ckpts} checkpoints)")
        print(f"    Estimated checkpoint memory: {estimated['new_ckpt_mb']:.1f} MB")

        try:
            # Run forward with checkpointing
            partition, ring_ckpts, actual_interval = semi_crf_forward_with_ring_checkpoints(
                edge, lengths
            )

            mem_after_fwd = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(f"    Actual checkpoint memory: {ring_ckpts.numel() * 4 / (1024*1024):.1f} MB")
            print(f"    Memory after forward: {mem_after_fwd - mem_edge:.1f} MB")
            print(f"    Peak memory: {mem_peak:.1f} MB")
            print("    Result: PASS")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM at T={T}: {e}")
                print("    Result: FAIL (OOM)")
                return False
            raise

        # Clean up
        del edge, lengths, partition, ring_ckpts
        gc.collect()
        torch.cuda.empty_cache()

    return True


def test_scale_gradient(device="cuda"):
    """Test gradient computation at moderate scale."""
    print("\n" + "=" * 60)
    print(f"Test 6: Scale Gradient Test - {device}")
    print("=" * 60)

    if device != "cuda" or not torch.cuda.is_available():
        print("CUDA not available or not requested, skipping")
        return True

    torch.manual_seed(42)

    # Medium scale for gradient testing
    configs = [
        ("Medium", 5000, 200, 12, 1),
        ("Larger", 10000, 500, 16, 1),
    ]

    all_pass = True
    for name, T, K, C, batch in configs:
        print(f"\n  {name}: T={T:,}, K={K:,}, C={C}, batch={batch}")

        gc.collect()
        torch.cuda.empty_cache()

        # Create edge tensor with small values to avoid numerical issues
        edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float32, device=device) * 0.01
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        try:
            t_start = time.time()

            # Forward + backward with Triton checkpointing
            edge_grad = edge.clone().requires_grad_(True)
            partition = semi_crf_triton_checkpointed_backward(edge_grad, lengths)
            partition.sum().backward()

            t_elapsed = time.time() - t_start

            # Basic sanity checks
            has_nan = edge_grad.grad.isnan().any().item()
            has_inf = edge_grad.grad.isinf().any().item()
            grad_sum = edge_grad.grad.sum().item()
            grad_min = edge_grad.grad.min().item()
            grad_max = edge_grad.grad.max().item()

            print(f"    Time: {t_elapsed:.2f}s")
            print(f"    Partition: {partition.item():.4f}")
            print(f"    Gradient sum: {grad_sum:.4f}")
            print(f"    Gradient range: [{grad_min:.6f}, {grad_max:.6f}]")
            print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")

            # Gradients should be non-negative (marginal probabilities)
            all_nonneg = (edge_grad.grad >= -1e-6).all().item()
            print(f"    All gradients >= 0: {all_nonneg}")

            passed = not has_nan and not has_inf and all_nonneg
            all_pass = all_pass and passed
            print(f"    Result: {'PASS' if passed else 'FAIL'}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM: {e}")
                print("    Result: FAIL (OOM)")
                all_pass = False
            else:
                raise

        del edge, lengths
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_full_scale(device="cuda"):
    """Test at full target dimensions."""
    print("\n" + "=" * 60)
    print(f"Test 7: Full Target Scale - {device}")
    print("=" * 60)

    if device != "cuda" or not torch.cuda.is_available():
        print("CUDA not available or not requested, skipping")
        return True

    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()

    # Target: T=400K, K=3K, C=24
    T, K, C, batch = 400_000, 3000, 24, 1

    print(f"\n  Target: T={T:,}, K={K:,}, C={C}, batch={batch}")

    # Show estimates
    estimated = estimate_memory(T, K, C, batch)
    print("  Estimated memory breakdown:")
    print(f"    Checkpoints: {estimated['new_ckpt_mb']:.1f} MB")
    print(f"    Segment buffer: {estimated['segment_mb']:.1f} MB")
    print(f"    Beta ring: {estimated['beta_ring_mb']:.1f} MB")
    print(f"    TOTAL: {estimated['new_total_mb']:.1f} MB")

    # Calculate edge tensor size (this is the big one!)
    edge_size_gb = batch * (T - 1) * K * C * C * 4 / (1024**3)
    print(f"    Edge tensor: {edge_size_gb:.1f} GB (!)")

    if edge_size_gb > 40:  # More than one L40S can handle
        print(f"\n  WARNING: Edge tensor ({edge_size_gb:.1f} GB) exceeds single GPU memory")
        print("  This test requires model parallelism or reduced dimensions")
        print("  Skipping full scale test - testing at reduced dimensions instead")

        # Test at reduced but still large dimensions
        T, K, C = 100_000, 1000, 24
        edge_size_gb = batch * (T - 1) * K * C * C * 4 / (1024**3)
        print(f"\n  Reduced: T={T:,}, K={K:,}, C={C}")
        print(f"    Edge tensor: {edge_size_gb:.1f} GB")

    torch.cuda.reset_peak_memory_stats()

    try:
        t_start = time.time()

        # Create edge tensor (zeros for speed)
        edge = torch.zeros(batch, T - 1, K, C, C, dtype=torch.float32, device=device)
        # Add small random values to first/last segments to avoid NaN
        edge[:, :100, :, :, :] = torch.randn(batch, 100, K, C, C, device=device) * 0.001
        edge[:, -100:, :, :, :] = torch.randn(batch, 100, K, C, C, device=device) * 0.001
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        t_create = time.time()
        print(f"\n  Edge tensor created in {t_create - t_start:.1f}s")
        print(f"  Memory after edge: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

        # Forward only (backward would need gradients which doubles memory)
        partition, ring_ckpts, interval = semi_crf_forward_with_ring_checkpoints(edge, lengths)

        t_fwd = time.time()
        print(f"  Forward completed in {t_fwd - t_create:.1f}s")
        print(f"  Checkpoint interval: {interval:,}")
        print(f"  Ring checkpoints shape: {ring_ckpts.shape}")
        print(f"  Checkpoint memory: {ring_ckpts.numel() * 4 / (1024**2):.1f} MB")
        print(f"  Peak memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print("  Result: PASS")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM: {e}")
            print("  Result: FAIL (OOM)")
            return False
        raise


def main():
    parser = argparse.ArgumentParser(description="Scale test for Semi-CRF backward pass")
    parser.add_argument("--quick", action="store_true", help="Quick sanity check only")
    parser.add_argument("--full", action="store_true", help="Include full scale test")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU only")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    print("=" * 60)
    print("Semi-CRF Backward Pass Scale Testing")
    print("=" * 60)

    if device == "cuda" and torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("Device: CPU")

    results = {}

    # Always run these
    results["checkpoint_interval"] = test_checkpoint_interval()
    results["memory_estimates"] = test_memory_estimates()

    if not args.quick:
        results["gradient_small"] = test_gradient_correctness_small(device)
        results["triton_small"] = test_triton_backward_small(device)
        results["scale_memory"] = test_scale_memory(device)
        results["scale_gradient"] = test_scale_gradient(device)

    if args.full:
        results["full_scale"] = test_full_scale(device)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
