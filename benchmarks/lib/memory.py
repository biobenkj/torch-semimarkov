"""Memory estimation and skip logic for benchmarks."""

from __future__ import annotations

import math


def bytes_to_gb(b: int) -> float:
    """Convert bytes to GB with 3 decimal places."""
    return round(b / (1024**3), 3)


def estimate_memory_breakdown(T: int, K: int, C: int, B: int, backend: str) -> dict[str, float]:
    """
    Estimate memory breakdown by category (in GB).

    Categories:
    - potentials: Edge potential tensor (B, T-1, K, C, C) for edge-tensor backends,
                  or cumulative scores for streaming backends
    - dp_state: Forward/backward DP tables
    - workspace: Intermediate computation tensors
    - autograd: Saved tensors for backward pass

    Backends:
    - binary_tree: O((KC)^2 * log(T)) - binary tree matmul intermediates
    - banded: Similar to binary_tree with banded reduction
    - block_triangular: O(K(K+1)/2 * C^2 * log(T)) - exploits sparsity
    - linear_scan: O(T*K*C) - full DP table
    - linear_scan_vectorized: O(T*K*C) - same as linear_scan
    - linear_scan_streaming: O(K*C) - ring buffer only
    - triton_streaming: O(T*C) cumulative scores + O(K*C) ring buffer
    """
    float_bytes = 4  # float32
    N = T - 1  # number of positions
    KC = K * C

    # ----------------------------------------
    # Edge-tensor backends: potentials = (B, T-1, K, C, C)
    # ----------------------------------------
    if backend in (
        "binary_tree",
        "binary_tree_sharded",
        "banded",
        "block_triangular",
        "linear_scan",
        "linear_scan_vectorized",
        "linear_scan_streaming",
    ):
        potentials_bytes = B * N * K * C * C * float_bytes

        if backend == "binary_tree":
            # Binary tree: O((KC)^2) per matmul, log2(N) levels
            # Plus intermediate results at each level
            log_levels = max(1, int(math.ceil(math.log2(N))))
            # Matmul workspace: (KC, KC) intermediate
            matmul_workspace = KC * KC * float_bytes
            # DP state: accumulator at each level
            dp_state_bytes = B * KC * KC * float_bytes * log_levels
            workspace_bytes = matmul_workspace * B
            # Autograd: saves all intermediate results for backward
            autograd_bytes = potentials_bytes + dp_state_bytes

        elif backend == "binary_tree_sharded":
            # Same algorithm as binary_tree but using CheckpointShardSemiring
            # which splits the O((KC)^3) matmul into smaller shards
            # This reduces peak memory at the cost of more serial computation
            log_levels = max(1, int(math.ceil(math.log2(N))))
            shard_size = 10000  # default shard size from checkpoint.py
            # DP state: same as binary_tree
            dp_state_bytes = B * KC * KC * float_bytes * log_levels
            # Workspace is reduced because we shard the matmul
            # Instead of (KC)^2 all at once, we do it in chunks
            workspace_bytes = B * min(KC * KC, shard_size) * float_bytes
            # Autograd still needs to save inputs but recomputes forward in backward
            autograd_bytes = potentials_bytes + dp_state_bytes // 2  # ~half of non-sharded

        elif backend == "banded":
            # Banded: similar to binary_tree but with bandwidth reduction
            # Assume effective bandwidth ~K for banded optimization
            log_levels = max(1, int(math.ceil(math.log2(N))))
            bandwidth = K  # approximate bandwidth
            dp_state_bytes = B * KC * bandwidth * float_bytes * log_levels
            workspace_bytes = bandwidth * KC * float_bytes * B
            autograd_bytes = potentials_bytes + dp_state_bytes

        elif backend == "block_triangular":
            # Block triangular: exploits k1 + k2 <= span sparsity
            # Uses K(K+1)/2 blocks of size (C, C) instead of full (KC, KC)
            log_levels = max(1, int(math.ceil(math.log2(N))))
            num_blocks = K * (K + 1) // 2
            block_size = C * C * float_bytes
            dp_state_bytes = B * num_blocks * block_size * log_levels
            workspace_bytes = num_blocks * block_size * B
            autograd_bytes = potentials_bytes + dp_state_bytes

        elif backend == "linear_scan":
            # Linear scan: O(T*K*C) for full DP table (non-vectorized)
            # Stores alpha[t, k, c] for all t
            dp_state_bytes = B * N * K * C * float_bytes
            workspace_bytes = B * K * C * C * float_bytes  # transition workspace
            autograd_bytes = potentials_bytes + dp_state_bytes

        elif backend == "linear_scan_vectorized":
            # Same memory as linear_scan, just faster computation
            dp_state_bytes = B * N * K * C * float_bytes
            workspace_bytes = B * K * C * C * float_bytes
            autograd_bytes = potentials_bytes + dp_state_bytes

        elif backend == "linear_scan_streaming":
            # TRUE streaming scan: O(K*C) DP state, independent of T
            # Uses ring buffer of size K for beta values
            dp_state_bytes = B * K * C * float_bytes + B * C * float_bytes
            workspace_bytes = B * K * C * C * float_bytes + B * K * C * float_bytes
            # Autograd: saves potentials + per-timestep DP state
            autograd_bytes = potentials_bytes + dp_state_bytes * N

    # ----------------------------------------
    # Streaming backends: compute edges on-the-fly
    # ----------------------------------------
    elif backend == "triton_streaming":
        # Streaming: cumulative scores (B, T+1, C) instead of edge tensor
        # Plus transition (C, C) and duration_bias (K, C)
        potentials_bytes = B * (T + 1) * C * float_bytes + C * C * float_bytes + K * C * float_bytes

        # Ring buffer: O(K*C) for DP state
        C_PAD = 1
        while C_PAD < C:
            C_PAD *= 2
        ring_buffer_bytes = B * K * C_PAD * float_bytes

        # Checkpoint interval for backward pass
        checkpoint_interval = max(1, int(math.sqrt(T * K)))
        num_checkpoints = T // checkpoint_interval + 1
        checkpoint_bytes = num_checkpoints * B * K * C * float_bytes

        dp_state_bytes = ring_buffer_bytes + checkpoint_bytes
        workspace_bytes = B * C * float_bytes  # small workspace
        autograd_bytes = potentials_bytes  # saves cumulative scores

    else:
        # Unknown backend - return zeros
        potentials_bytes = 0
        dp_state_bytes = 0
        workspace_bytes = 0
        autograd_bytes = 0

    return {
        "potentials_gb": bytes_to_gb(potentials_bytes),
        "dp_state_gb": bytes_to_gb(dp_state_bytes),
        "workspace_gb": bytes_to_gb(workspace_bytes),
        "autograd_gb": bytes_to_gb(autograd_bytes),
        "total_estimated_gb": bytes_to_gb(
            potentials_bytes + dp_state_bytes + workspace_bytes + autograd_bytes
        ),
    }


def should_skip_config(
    T: int,
    K: int,
    C: int,
    backend: str,
    oom_history: dict[str, list[tuple[int, int, int]]],
    max_memory_gb: float = 40.0,
) -> tuple[bool, str]:
    """
    Determine if we should skip this config based on:
    1. Predicted memory > max_memory_gb
    2. Adjacent config already OOM'd

    Returns (should_skip, reason).
    """
    # Check if adjacent (smaller) config OOM'd for this backend
    key = backend
    if key in oom_history:
        for oom_t, oom_k, oom_c in oom_history[key]:
            # If a smaller config OOM'd, skip larger ones
            if T >= oom_t and K >= oom_k and C >= oom_c:
                if T > oom_t or K > oom_k or C > oom_c:
                    return True, f"adjacent OOM at T={oom_t},K={oom_k},C={oom_c}"

    # Estimate memory and skip if too large
    breakdown = estimate_memory_breakdown(T, K, C, 4, backend)  # assume B=4
    if breakdown["total_estimated_gb"] > max_memory_gb:
        return True, f"predicted {breakdown['total_estimated_gb']:.1f}GB > {max_memory_gb}GB limit"

    return False, ""
