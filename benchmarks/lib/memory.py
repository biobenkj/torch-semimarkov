"""Memory estimation and skip logic for benchmarks."""

from __future__ import annotations


def bytes_to_gb(b: int) -> float:
    """Convert bytes to GB with 3 decimal places."""
    return round(b / (1024**3), 3)


def estimate_memory_breakdown(T: int, K: int, C: int, B: int, backend: str) -> dict[str, float]:
    """
    Estimate memory breakdown by category (in GB).

    Categories:
    - potentials: Edge potential tensor (B, T-1, K, C, C)
    - dp_state: Forward/backward DP tables
    - workspace: Intermediate computation tensors
    - autograd: Saved tensors for backward pass
    """
    float_bytes = 4  # float32
    N = T - 1  # number of positions

    # Potentials: (B, T-1, K, C, C) - always present
    potentials_bytes = B * N * K * C * C * float_bytes

    # DP state depends on backend
    if backend in ["linear_scan", "linear_scan_vectorized"]:
        # alpha: (ssize, B, N, K, C) -> O(T*K*C) resident
        # beta: list of N tensors of shape (ssize, B, C) -> O(T*C)
        alpha_bytes = 1 * B * N * K * C * float_bytes
        beta_bytes = 1 * B * N * C * float_bytes
        dp_state_bytes = alpha_bytes + beta_bytes

        if backend == "linear_scan_vectorized":
            workspace_bytes = B * K * C * float_bytes * 2
        else:
            workspace_bytes = B * C * float_bytes

        autograd_bytes = potentials_bytes

    elif backend == "linear_scan_streaming":
        # TRUE streaming scan: O(K*C) DP state, independent of T
        dp_state_bytes = 1 * B * K * C * float_bytes + 1 * B * C * float_bytes
        workspace_bytes = B * K * C * C * float_bytes + B * K * C * float_bytes
        autograd_bytes = potentials_bytes + dp_state_bytes * N

    elif backend == "binary_tree":
        # Tree stores chart matrices at each level
        # CRITICAL: log-semiring matmul materializes O((KC)^3) temporary
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * KC * KC * float_bytes  # (KC)^3 temporary
        autograd_bytes = B * N * KC * KC * float_bytes * 2

    elif backend == "binary_tree_sharded":
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        shard_size = 10000
        workspace_bytes = B * min(KC * KC * KC, shard_size * KC) * float_bytes
        autograd_bytes = B * N * KC * KC * float_bytes

    elif backend == "banded":
        KC = K * C
        bw = min(KC, K * 2)
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * bw * float_bytes
        autograd_bytes = B * N * KC * bw * float_bytes

    elif backend == "block_triangular":
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * KC * float_bytes
        autograd_bytes = B * N * KC * KC * float_bytes

    elif backend in ("triton", "triton_pytorch", "triton_checkpointing"):
        # Triton scan uses O(K*C) ring buffer
        C_PAD = 1
        while C_PAD < C:
            C_PAD *= 2
        ring_buffer_bytes = B * K * C_PAD * float_bytes
        output_bytes = B * float_bytes
        dp_state_bytes = ring_buffer_bytes + output_bytes
        workspace_bytes = B * C * float_bytes
        autograd_bytes = potentials_bytes

    else:
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
