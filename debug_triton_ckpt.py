#!/usr/bin/env python3
"""Debug script to trace gradient differences in Triton checkpointed backward kernel."""

import torch
from torch_semimarkov.triton_backward import (
    semi_crf_forward_with_ring_checkpoints,
    semi_crf_backward_from_ring_checkpoints,
    launch_triton_checkpointed_backward_kernel,
    NEG_INF,
)

def debug_alpha_recomputation():
    """Debug alpha recomputation in forward kernel by comparing segment by segment."""
    torch.manual_seed(5400)

    batch, T, K, C = 8, 64, 12, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ALPHA RECOMPUTATION DEBUG")
    print("=" * 60)

    edge = torch.randn(batch, T - 1, K, C, C, device=device)
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    # Get ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    print(f"Checkpoint interval: {interval}")
    print(f"Ring checkpoints shape: {ring_checkpoints.shape}")
    num_checkpoints = ring_checkpoints.shape[1]

    # Now manually recompute alpha for each segment using PyTorch
    # and compare with what the checkpoint provides

    # Full forward pass to get ground truth alpha
    alpha_full = torch.full((batch, T, C), NEG_INF, device=device, dtype=edge.dtype)
    alpha_full[:, 0, :] = 0.0

    for t in range(1, T):
        k_eff = min(K - 1, t)
        scores_all = []
        for k in range(1, k_eff + 1):
            start = t - k
            alpha_prev = alpha_full[:, start, :]
            edge_k = edge[:, start, k, :, :]
            scores = alpha_prev.unsqueeze(-2) + edge_k
            scores_all.append(scores)
        scores_stacked = torch.stack(scores_all, dim=1)
        scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
        alpha_full[:, t, :] = torch.logsumexp(scores_over_src, dim=1)

    print("\nGround truth alpha stats:")
    print(f"  Range: [{alpha_full.min():.4f}, {alpha_full.max():.4f}]")

    # Check each checkpoint
    print("\nCheckpoint verification:")
    for ckpt_idx in range(num_checkpoints):
        ckpt_pos = ckpt_idx * interval
        if ckpt_pos >= T:
            continue

        # Get alpha[ckpt_pos] from ground truth
        alpha_gt = alpha_full[:, ckpt_pos, :]  # (batch, C)

        # Get alpha[ckpt_pos] from checkpoint ring buffer
        ring_slot = ckpt_pos % K
        alpha_ckpt = ring_checkpoints[:, ckpt_idx, ring_slot, :]  # (batch, C)

        diff = (alpha_gt - alpha_ckpt).abs().max()
        print(f"  Checkpoint {ckpt_idx} (pos {ckpt_pos}): max_diff={diff:.6f}")

        if diff > 1e-3:
            print(f"    WARNING: Large difference!")
            print(f"    Ground truth: {alpha_gt[0, :4]}")
            print(f"    Checkpoint: {alpha_ckpt[0, :4]}")

def debug_gradient_mismatch():
    """Debug gradient mismatch with detailed output."""
    torch.manual_seed(5400)

    # Use the failing test parameters
    batch, T, K, C = 8, 64, 12, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("GRADIENT MISMATCH DEBUG")
    print("=" * 60)
    print(f"Testing with batch={batch}, T={T}, K={K}, C={C}")
    print(f"Device: {device}")

    edge = torch.randn(batch, T - 1, K, C, C, device=device)
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    # Compute forward with ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    print(f"Checkpoint interval: {interval}")
    print(f"Ring checkpoints shape: {ring_checkpoints.shape}")
    print(f"Partition (log_Z) range: [{partition.min():.2f}, {partition.max():.2f}]")

    # PyTorch reference
    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    print(f"\nPyTorch gradient stats:")
    print(f"  Shape: {grad_pytorch.shape}")
    print(f"  Range: [{grad_pytorch.min():.6f}, {grad_pytorch.max():.6f}]")
    print(f"  Sum: {grad_pytorch.sum():.6f}")
    print(f"  Has NaN: {torch.isnan(grad_pytorch).any()}")
    print(f"  Has Inf: {torch.isinf(grad_pytorch).any()}")

    if device == "cuda":
        # Triton kernel
        grad_triton = launch_triton_checkpointed_backward_kernel(
            edge, ring_checkpoints, partition, lengths, interval
        )

        print(f"\nTriton gradient stats:")
        print(f"  Shape: {grad_triton.shape}")
        print(f"  Range: [{grad_triton.min():.6f}, {grad_triton.max():.6f}]")
        print(f"  Sum: {grad_triton.sum():.6f}")
        print(f"  Has NaN: {torch.isnan(grad_triton).any()}")
        print(f"  Has Inf: {torch.isinf(grad_triton).any()}")

        # Compare
        diff = (grad_triton - grad_pytorch).abs()
        print(f"\nDifference stats:")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")

        # Find where the max difference occurs
        max_idx = torch.argmax(diff.view(-1))
        max_idx_tuple = torch.unravel_index(max_idx, diff.shape)
        print(f"  Max diff at index: {[x.item() for x in max_idx_tuple]}")
        print(f"  PyTorch value: {grad_pytorch[max_idx_tuple].item():.6f}")
        print(f"  Triton value: {grad_triton[max_idx_tuple].item():.6f}")

        # Check by segment
        num_checkpoints = ring_checkpoints.shape[1]
        print(f"\nPer-segment analysis (checkpoint_interval={interval}):")
        for ckpt_idx in range(num_checkpoints):
            seg_start = ckpt_idx * interval
            seg_end = min((ckpt_idx + 1) * interval, T - 1)  # T-1 because edge has T-1 positions

            if seg_start >= T - 1:
                continue

            seg_diff = diff[:, seg_start:seg_end, :, :, :]
            seg_pytorch = grad_pytorch[:, seg_start:seg_end, :, :, :]
            seg_triton = grad_triton[:, seg_start:seg_end, :, :, :]

            print(f"  Segment {ckpt_idx} (t={seg_start}-{seg_end-1}):")
            print(f"    PyTorch range: [{seg_pytorch.min():.6f}, {seg_pytorch.max():.6f}]")
            print(f"    Triton range: [{seg_triton.min():.6f}, {seg_triton.max():.6f}]")
            print(f"    Max diff: {seg_diff.max():.6f}")

        # Check by k value
        print(f"\nPer-duration analysis:")
        for k in range(1, K):
            k_diff = diff[:, :, k, :, :]
            k_pytorch = grad_pytorch[:, :, k, :, :]
            k_triton = grad_triton[:, :, k, :, :]
            print(f"  k={k}: PyTorch [{k_pytorch.min():.4f}, {k_pytorch.max():.4f}], "
                  f"Triton [{k_triton.min():.4f}, {k_triton.max():.4f}], max_diff={k_diff.max():.4f}")
    else:
        print("\nSkipping Triton kernel (no CUDA)")


def test_smaller_shapes():
    """Test smaller shapes that are known to pass, to verify the debug approach works."""
    torch.manual_seed(5100)

    # These should pass
    batch, T, K, C = 4, 20, 5, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SMALLER SHAPES TEST (should pass)")
    print("=" * 60)
    print(f"Testing with batch={batch}, T={T}, K={K}, C={C}")

    edge = torch.randn(batch, T - 1, K, C, C, device=device)
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    if device == "cuda":
        grad_triton = launch_triton_checkpointed_backward_kernel(
            edge, ring_checkpoints, partition, lengths, interval
        )

        diff = (grad_triton - grad_pytorch).abs()
        print(f"Max diff: {diff.max():.6f}")
        print(f"PASS" if diff.max() < 1e-3 else "FAIL")


if __name__ == "__main__":
    test_smaller_shapes()
    print("\n")
    debug_alpha_recomputation()
    print("\n")
    debug_gradient_mismatch()
