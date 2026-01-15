#!/usr/bin/env python3
"""
Memory analysis benchmark for Semi-Markov CRF backends.

Generates data for:
1. OOM feasibility heatmaps (with consistent GB units)
2. Time vs state-space size plots (with median/IQR)
3. Memory breakdown by allocation category

Supports:
- Multiple backends including Triton-accelerated scan
- Forward-only, backward-only, or combined timing
- Different semirings (Log, Max, Entropy, etc.)

Example:
    python benchmarks/benchmark_memory_analysis.py \
        --device cuda:0 \
        --T 128,256,512,1024 \
        --K 4,8,12,16,20,24 \
        --C 3,6,9,12 \
        --B 4 \
        --repeats 5 \
        --phases both \
        --semirings Log,Max \
        --output-dir results/
"""

import argparse
import csv
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import EntropySemiring, LogSemiring, MaxSemiring
from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring
from torch_semimarkov.triton_scan import HAS_TRITON, semi_crf_triton_forward


# Reset torch.compile caches to avoid state issues between configurations
def reset_compile_caches():
    """Reset torch.compile and triton caches to avoid inter-configuration issues."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    # Also reset the module-level compiled function caches
    try:
        import torch_semimarkov.triton_scan as ts

        ts._compiled_forward_log = None
        ts._compiled_forward_max = None
    except Exception:
        pass


# Mapping of semiring names to classes
SEMIRING_MAP = {
    "Log": LogSemiring,
    "Max": MaxSemiring,
    "Entropy": EntropySemiring,
}

# Backends that only work with LogSemiring (due to hardcoded logsumexp)
LOG_SEMIRING_ONLY_BACKENDS: set[str] = set()

# Backends that only support a subset of semirings
# triton/triton_pytorch support Log and Max, but not Entropy
TRITON_SUPPORTED_SEMIRINGS = {"Log", "Max"}


@dataclass
class BenchmarkResult:
    """Single benchmark result with full metrics."""

    T: int
    K: int
    C: int
    B: int
    KC: int  # state-space size
    backend: str
    semiring: str  # "Log", "Max", "Entropy", etc.
    phase: str  # "forward", "backward", "both"

    # Timing (all runs)
    time_ms_median: float
    time_ms_iqr_low: float
    time_ms_iqr_high: float
    time_per_position_ms: float  # time_ms_median / T

    # Memory in GB (consistent units)
    peak_allocated_gb: float
    peak_reserved_gb: float

    # Status
    status: str  # "success", "oom", "not_tested", "error"
    error_msg: str = ""

    # Memory breakdown (if available)
    memory_potentials_gb: float = 0.0
    memory_dp_state_gb: float = 0.0
    memory_workspace_gb: float = 0.0
    memory_autograd_gb: float = 0.0


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [values[0]]
    n = len(values)
    idxs = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    sampled: list[int] = []
    for idx in idxs:
        if idx not in seen:
            sampled.append(values[idx])
            seen.add(idx)
    if len(sampled) < count:
        for idx in range(n):
            if idx not in seen:
                sampled.append(values[idx])
                seen.add(idx)
                if len(sampled) == count:
                    break
    return sampled


def _choose_grid_counts(t_count: int, kc_count: int, max_points: int) -> tuple[int, int]:
    best_t, best_kc = 1, 1
    best_prod = 1
    best_balance = float("inf")
    for t_idx in range(1, t_count + 1):
        kc_idx = min(kc_count, max_points // t_idx)
        if kc_idx < 1:
            continue
        prod = t_idx * kc_idx
        balance = abs((t_idx / t_count) - (kc_idx / kc_count))
        if prod > best_prod or (prod == best_prod and balance < best_balance):
            best_t, best_kc = t_idx, kc_idx
            best_prod = prod
            best_balance = balance
    return best_t, best_kc


def sample_configurations(
    T_list: list[int],
    K_list: list[int],
    C_list: list[int],
    B: int,
    max_points: int,
) -> list[tuple[int, int, int]]:
    """Sample (T, K, C) configs by T and K*C, anchored at min/max BTKC."""
    full_configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
    if max_points <= 0 or max_points >= len(full_configs):
        return full_configs

    t_values = sorted(set(T_list))
    kc_to_pairs: dict[int, list[tuple[int, int]]] = {}
    for K in K_list:
        for C in C_list:
            kc_to_pairs.setdefault(K * C, []).append((K, C))
    kc_values = sorted(kc_to_pairs.keys())

    t_count = len(t_values)
    kc_count = len(kc_values)
    t_sample_count, kc_sample_count = _choose_grid_counts(t_count, kc_count, max_points)

    t_samples = _evenly_spaced(t_values, t_sample_count)
    kc_samples = _evenly_spaced(kc_values, kc_sample_count)

    sampled_pairs: set[tuple[int, int]] = {(T, KC) for T in t_samples for KC in kc_samples}
    sampled_pairs.add((t_values[0], kc_values[0]))
    sampled_pairs.add((t_values[-1], kc_values[-1]))

    if len(sampled_pairs) < max_points:
        all_pairs = [(T, KC) for T in t_values for KC in kc_values]
        all_pairs.sort(key=lambda pair: B * pair[0] * pair[1])
        for pair in all_pairs:
            if len(sampled_pairs) >= max_points:
                break
            sampled_pairs.add(pair)

    t_index_map = {T: idx for idx, T in enumerate(t_values)}
    kc_index_map = {KC: idx for idx, KC in enumerate(kc_values)}

    configs: list[tuple[int, int, int]] = []
    for T, KC in sorted(sampled_pairs, key=lambda pair: (pair[0], pair[1])):
        pairs = kc_to_pairs[KC]
        pair_idx = (t_index_map[T] + kc_index_map[KC]) % len(pairs)
        K, C = pairs[pair_idx]
        configs.append((T, K, C))

    return configs


# =============================================================================
# Compile-Aware Sampling (reduces torch.compile overhead)
# =============================================================================

# Canonical shape buckets - these are the shapes we actually compile kernels for
T_BUCKETS = [64, 128, 256, 512, 1024, 2048]
KC_BUCKETS = [12, 24, 48, 72, 96, 144, 192, 288]


def bucket_to_canonical_shape(T: int, K: int, C: int) -> tuple[int, int, int]:
    """
    Round (T, K, C) to canonical shapes that maximize compiled kernel reuse.

    torch.compile generates specialized kernels per unique tensor shape.
    By bucketing to canonical shapes, we reduce compilation from O(configs)
    to O(buckets), typically 8-16 unique kernels instead of 50-100+.

    Returns:
        (T_canon, K_canon, C_canon) - the canonical shape to use for compilation
    """
    # Bucket T to nearest bucket >= T (or largest if T exceeds all)
    T_canon = T_BUCKETS[-1]
    for t_bucket in T_BUCKETS:
        if t_bucket >= T:
            T_canon = t_bucket
            break

    # Bucket K*C product
    KC = K * C
    KC_canon = KC_BUCKETS[-1]
    for kc_bucket in KC_BUCKETS:
        if kc_bucket >= KC:
            KC_canon = kc_bucket
            break

    # Find K, C factors of KC_canon that are closest to original ratio
    # Prefer keeping K close to original since it affects duration modeling
    best_K, best_C = K, C
    best_score = float("inf")
    for k in range(1, KC_canon + 1):
        if KC_canon % k == 0:
            c = KC_canon // k
            # Score: prefer K close to original, C close to original
            score = abs(k - K) + abs(c - C) * 0.5
            if score < best_score:
                best_K, best_C = k, c
                best_score = score

    return T_canon, best_K, best_C


def get_canonical_shapes(
    T_list: list[int], K_list: list[int], C_list: list[int]
) -> list[tuple[int, int, int]]:
    """Get the set of unique canonical shapes for a parameter grid."""
    seen: set[tuple[int, int, int]] = set()
    canonical: list[tuple[int, int, int]] = []

    for T in T_list:
        for K in K_list:
            for C in C_list:
                canon = bucket_to_canonical_shape(T, K, C)
                if canon not in seen:
                    seen.add(canon)
                    canonical.append(canon)

    # Sort by memory footprint (T * K * C)
    canonical.sort(key=lambda x: x[0] * x[1] * x[2])
    return canonical


def sample_compile_friendly(
    T_list: list[int],
    K_list: list[int],
    C_list: list[int],
    max_canonical_shapes: int = 8,
    samples_per_shape: int = 2,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], tuple[int, int, int]]]:
    """
    Sample configurations that minimize unique compiled shapes.

    This is a two-phase approach:
    1. Select a subset of canonical shapes (compile targets)
    2. Sample actual configs that map to those canonical shapes

    Args:
        T_list: Sequence lengths to consider
        K_list: Max durations to consider
        C_list: Label counts to consider
        max_canonical_shapes: Maximum unique shapes to compile
        samples_per_shape: Actual configs to benchmark per canonical shape

    Returns:
        (sampled_configs, config_to_canonical_map)
    """
    # Build all configs and group by canonical shape
    all_configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
    shape_groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for cfg in all_configs:
        canon = bucket_to_canonical_shape(*cfg)
        shape_groups.setdefault(canon, []).append(cfg)

    # Select canonical shapes with good coverage (evenly spaced by memory)
    all_canonical = sorted(shape_groups.keys(), key=lambda s: s[0] * s[1] * s[2])

    if len(all_canonical) <= max_canonical_shapes:
        selected_canonical = all_canonical
    else:
        # Evenly spaced selection
        selected_canonical = _evenly_spaced(all_canonical, max_canonical_shapes)

    # Sample actual configs from each selected canonical group
    sampled: list[tuple[int, int, int]] = []
    config_to_canon: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for canon in selected_canonical:
        group = shape_groups.get(canon, [])
        # Sort group by actual size and take evenly spaced samples
        group_sorted = sorted(group, key=lambda c: c[0] * c[1] * c[2])

        if len(group_sorted) <= samples_per_shape:
            selected = group_sorted
        else:
            selected = _evenly_spaced(group_sorted, samples_per_shape)

        for cfg in selected:
            sampled.append(cfg)
            config_to_canon[cfg] = canon

    return sampled, config_to_canon


def setup_compile_cache(
    output_dir: Path,
    cache_dir: Path | None = None,
    max_cache_size_gb: float = 10.0,
) -> Path:
    """
    Configure persistent compilation cache to avoid recompilation across runs.

    Args:
        output_dir: Default location for cache if cache_dir not specified
        cache_dir: Explicit cache directory (e.g., /tmp for HPC local scratch)
        max_cache_size_gb: Maximum cache size in GB (default 10GB)

    Returns the cache directory path.

    HPC Notes:
        - Use local scratch (e.g., /tmp, $TMPDIR, or node-local NVMe) for faster I/O
        - On shared filesystems, compile cache can cause lock contention
        - Set max_cache_size_gb based on available scratch space
    """
    if cache_dir is None:
        cache_dir = output_dir / ".torch_compile_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Enable FX graph caching
    inductor_config.fx_graph_cache = True
    inductor_config.fx_graph_remote_cache = False  # Local only

    # Set cache size limit (in bytes)
    # This controls automatic cache eviction when size exceeds limit
    try:
        inductor_config.fx_graph_cache_size_limit = int(max_cache_size_gb * 1024**3)
    except AttributeError:
        # Older PyTorch versions may not have this config
        pass

    # Set cache directory via environment variable
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    # Also set TRITON_CACHE_DIR for Triton kernel caching
    triton_cache = cache_dir / "triton"
    triton_cache.mkdir(exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    return cache_dir


def precompile_canonical_shapes(
    canonical_shapes: list[tuple[int, int, int]],
    device: torch.device,
    backends: list[str],
    semirings: list[str],
    use_compile: bool = True,
) -> None:
    """
    Pre-compile kernels for all canonical shapes before timing.

    This separates compilation time from benchmark timing, giving more
    accurate runtime measurements and avoiding compilation during timed runs.
    """
    from torch_semimarkov.triton_scan import semi_crf_triton_forward

    triton_backends = {"triton", "triton_pytorch", "triton_checkpointing"}
    active_triton_backends = [b for b in backends if b in triton_backends]

    if not active_triton_backends or not use_compile:
        return

    print(f"\nPre-compiling {len(canonical_shapes)} canonical shapes...")
    print(f"  Backends: {active_triton_backends}")
    print("  This may take a few minutes on first run (cached afterward)\n")

    for i, (T, K, C) in enumerate(canonical_shapes):
        print(
            f"  [{i+1}/{len(canonical_shapes)}] Shape T={T}, K={K}, C={C}...", end=" ", flush=True
        )

        for semiring_name in semirings:
            if semiring_name not in TRITON_SUPPORTED_SEMIRINGS:
                continue

            triton_semiring = semiring_name.lower()

            # Use B=1 for faster compilation (shape is what matters)
            edge = torch.randn(1, T - 1, K, C, C, device=device, requires_grad=True)
            lengths = torch.full((1,), T, dtype=torch.long, device=device)

            for backend in active_triton_backends:
                try:
                    use_triton_kernel = backend in ("triton", "triton_checkpointing")
                    backend_use_compile = use_compile and backend != "triton_checkpointing"

                    # Forward pass (triggers compilation)
                    v = semi_crf_triton_forward(
                        edge,
                        lengths,
                        use_triton=use_triton_kernel,
                        semiring=triton_semiring,
                        use_compile=backend_use_compile,
                    )
                    # Backward pass (triggers backward kernel compilation)
                    v.sum().backward()

                except Exception as e:
                    print(f"(warn: {backend}/{semiring_name}: {str(e)[:30]})", end=" ")

            del edge, lengths
            torch.cuda.empty_cache()

        print("done")

    print("\nPre-compilation complete. Starting benchmarks...\n")
    gc.collect()
    torch.cuda.empty_cache()


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

    NOTE: These are estimates based on actual implementation analysis.
    For precise measurements, use torch.cuda.memory_snapshot().

    Actual implementation details (as of code review):
    - linear_scan stores alpha: (ssize, B, N, K, C) -> O(T*K*C)
    - linear_scan stores beta: list of N tensors (ssize, B, C) -> O(T*C)
    - binary_tree log-semiring matmul materializes O((KC)^3) temporaries
    """
    float_bytes = 4  # float32
    N = T - 1  # number of positions

    # Potentials: (B, T-1, K, C, C) - always present
    potentials_bytes = B * N * K * C * C * float_bytes

    # DP state depends on backend
    if backend in ["linear_scan", "linear_scan_vectorized"]:
        # ACTUAL implementation (not theoretical O(TC)):
        # alpha: (ssize, B, N, K, C) -> O(T*K*C) resident
        # beta: list of N tensors of shape (ssize, B, C) -> O(T*C)
        # Total DP state is O(T*K*C + T*C) = O(T*K*C) when K > 1
        alpha_bytes = 1 * B * N * K * C * float_bytes  # ssize=1 for LogSemiring
        beta_bytes = 1 * B * N * C * float_bytes
        dp_state_bytes = alpha_bytes + beta_bytes

        # Workspace for vectorized: index tensors + gathered tensors
        if backend == "linear_scan_vectorized":
            # time_indices, dur_indices: (K,) each
            # gathered: (ssize, B, K, C) per step
            workspace_bytes = B * K * C * float_bytes * 2  # gathered + temp
        else:
            workspace_bytes = B * C * float_bytes  # minimal

        # Autograd: saves computation graph, roughly proportional to potentials
        autograd_bytes = potentials_bytes

    elif backend == "linear_scan_streaming":
        # TRUE streaming scan: O(K*C) DP state, independent of T
        # beta_hist: (ssize, B, K, C) ring buffer of last K betas
        # final_beta: (ssize, B, C)
        # NO alpha storage, NO full beta history
        dp_state_bytes = 1 * B * K * C * float_bytes + 1 * B * C * float_bytes
        # Workspace: edge_slice (B, k_eff, C, C), scores (B, k_eff, C) per step
        workspace_bytes = B * K * C * C * float_bytes + B * K * C * float_bytes
        # Autograd: saves edge (potentials) + ring buffer states
        autograd_bytes = potentials_bytes + dp_state_bytes * N  # graph grows with T

    elif backend == "binary_tree":
        # Tree stores chart matrices of shape (ssize, B, KC, KC) at each level
        # CRITICAL: log-semiring matmul materializes O((KC)^3) temporary
        # because it computes: result[i,k] = logsumexp_j(A[i,j] + B[j,k])
        # via broadcast: (KC, KC, 1) + (1, KC, KC) -> (KC, KC, KC)
        KC = K * C
        # Chart storage across ~log(T) levels, but dominant cost is at base
        dp_state_bytes = B * N * KC * float_bytes  # vector at each position
        # Workspace: the O((KC)^3) temporary per matmul is the killer
        # At each level, we do ~T/2^level matmuls; base level is worst
        workspace_bytes = B * KC * KC * KC * float_bytes  # (KC)^3 temporary!
        # Autograd saves inputs for backward through all matmuls
        autograd_bytes = B * N * KC * KC * float_bytes * 2

    elif backend == "binary_tree_sharded":
        # Same algorithm as binary_tree but using CheckpointShardSemiring
        # which splits the O((KC)^3) matmul into smaller shards
        # This reduces peak memory at the cost of more serial computation
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        # Workspace is reduced because we shard the matmul
        # Instead of (KC)^3 all at once, we do it in chunks
        shard_size = 10000  # default shard size from checkpoint.py
        workspace_bytes = B * min(KC * KC * KC, shard_size * KC) * float_bytes
        # Autograd still needs to save inputs but recomputes forward in backward
        autograd_bytes = B * N * KC * KC * float_bytes  # ~half of non-sharded

    elif backend == "banded":
        # O(T * KC * BW) where BW is bandwidth
        KC = K * C
        bw = min(KC, K * 2)  # rough bandwidth estimate
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * bw * float_bytes  # per-multiply workspace
        autograd_bytes = B * N * KC * bw * float_bytes

    elif backend == "block_triangular":
        # Similar to binary tree but with block sparsity
        # Current impl converts to/from dense at each level (suboptimal)
        KC = K * C
        dp_state_bytes = B * N * KC * float_bytes
        workspace_bytes = B * KC * KC * float_bytes  # dense conversion overhead
        autograd_bytes = B * N * KC * KC * float_bytes

    elif backend in ("triton", "triton_pytorch", "triton_checkpointing"):
        # Triton scan uses O(K*C) ring buffer, same as streaming scan
        # triton: Fused GPU kernel with ring buffer in L1/L2
        # triton_pytorch: Reference PyTorch implementation (same algorithm)
        # Ring buffer: (B, K, C_PAD) where C_PAD = next_power_of_2(C)
        C_PAD = 1
        while C_PAD < C:
            C_PAD *= 2
        ring_buffer_bytes = B * K * C_PAD * float_bytes
        # Output partition: (B,)
        output_bytes = B * float_bytes
        dp_state_bytes = ring_buffer_bytes + output_bytes
        # Workspace: minimal for triton (fused kernel)
        workspace_bytes = B * C * float_bytes  # final_beta temporary
        # Autograd: Triton uses gradient checkpointing (recomputes forward in backward)
        # So autograd memory is primarily the saved edge tensor
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


def run_single_benchmark(
    T: int,
    K: int,
    C: int,
    B: int,
    backend: str,
    device: torch.device,
    repeats: int = 5,
    semiring_name: str = "Log",
    phase: str = "both",
    use_compile: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        T: Sequence length
        K: Maximum duration
        C: Number of labels
        B: Batch size
        backend: Backend name (linear_scan, triton, etc.)
        device: Torch device
        repeats: Number of repetitions for timing
        semiring_name: Semiring to use ("Log", "Max", "Entropy")
        phase: "forward" (forward only), "backward" (backward only), or "both"
        use_compile: If True, use torch.compile for triton training backward pass.
            If False, use gradient checkpointing. Default: True
    """

    KC = K * C
    result_base = {
        "T": T,
        "K": K,
        "C": C,
        "B": B,
        "KC": KC,
        "backend": backend,
        "semiring": semiring_name,
        "phase": phase,
    }

    # Estimate memory breakdown
    breakdown = estimate_memory_breakdown(T, K, C, B, backend)

    # Get semiring class
    semiring_cls = SEMIRING_MAP.get(semiring_name, LogSemiring)

    try:
        # Create struct with appropriate semiring
        struct = SemiMarkov(semiring_cls)

        # Create potentials
        edge = torch.randn(B, T - 1, K, C, C, device=device, requires_grad=False)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        times_ms = []
        peak_allocated = 0
        peak_reserved = 0

        # Map semiring name to triton semiring string
        triton_semiring = semiring_name.lower()  # "Log" -> "log", "Max" -> "max"

        def run_backend_forward(edge_input, struct_to_use, use_compile=True):
            """Run forward pass for the specified backend, returning partition value.

            For triton backends, the execution path depends on edge_input.requires_grad:
            - requires_grad=False (inference): Custom Triton kernel (~45x speedup)
            - requires_grad=True + use_compile=True: torch.compile for efficient backward
            - requires_grad=True + use_compile=False: Gradient checkpointing (recomputes forward)
            """
            if backend == "triton":
                # Triton-accelerated scan (supports Log and Max semirings)
                # Hybrid routing: custom kernel for inference, torch.compile for training
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=True,
                    semiring=triton_semiring,
                    use_compile=use_compile,
                )
            elif backend == "triton_pytorch":
                # PyTorch reference for Triton (supports Log and Max semirings)
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=False,
                    semiring=triton_semiring,
                    use_compile=use_compile,
                )
            elif backend == "triton_checkpointing":
                # Triton with gradient checkpointing (old approach, for comparison)
                # Always uses use_compile=False to force checkpointing path
                return semi_crf_triton_forward(
                    edge_input,
                    lengths,
                    use_triton=True,
                    semiring=triton_semiring,
                    use_compile=False,
                )
            elif backend == "binary_tree":
                v, _ = struct_to_use.logpartition(
                    edge_input, lengths=lengths, use_linear_scan=False
                )
                return v
            elif backend == "linear_scan":
                v, _, _ = struct_to_use._dp_standard(edge_input, lengths, force_grad=True)
                return v
            elif backend == "linear_scan_vectorized":
                v, _, _ = struct_to_use._dp_standard_vectorized(
                    edge_input, lengths, force_grad=True
                )
                return v
            elif backend == "banded":
                v, _, _ = struct_to_use.logpartition(
                    edge_input,
                    lengths=lengths,
                    use_linear_scan=True,
                    use_vectorized=True,
                    use_banded=True,
                    banded_perm="auto",
                    banded_bw_ratio=0.6,
                )
                return v
            elif backend == "block_triangular":
                if hasattr(struct_to_use, "_dp_blocktriangular"):
                    v, _, _ = struct_to_use._dp_blocktriangular(
                        edge_input, lengths, force_grad=True
                    )
                    return v
                else:
                    raise NotImplementedError("block_triangular not available")
            elif backend == "binary_tree_sharded":
                # Use CheckpointShardSemiring to reduce peak memory
                ShardedSemiring = CheckpointShardSemiring(semiring_cls, max_size=10000)
                struct_sharded = SemiMarkov(ShardedSemiring)
                v, _ = struct_sharded.logpartition(
                    edge_input, lengths=lengths, use_linear_scan=False
                )
                return v
            elif backend == "linear_scan_streaming":
                v, _, _ = struct_to_use._dp_scan_streaming(edge_input, lengths, force_grad=True)
                return v
            else:
                raise ValueError(f"Unknown backend: {backend}")

        # Warmup run (not recorded) to stabilize CUDA state
        # IMPORTANT: Run the exact same computation as the timed run for accurate warmup
        # For triton with torch.compile, this triggers the JIT compilation
        if device.type == "cuda":
            warmup_needs_grad = phase in ("backward", "both")
            edge_warmup = edge.clone().detach().requires_grad_(warmup_needs_grad)
            try:
                v_warm = run_backend_forward(edge_warmup, struct, use_compile=use_compile)
                # Warmup backward pass too if we're timing backward
                if phase in ("backward", "both"):
                    v_warm.sum().backward()
                del edge_warmup, v_warm
            except Exception:
                pass  # If warmup fails, actual run will catch it
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()

        for _rep in range(repeats):
            # IMPORTANT: Reset memory stats BEFORE allocating edge_run
            # so that edge_run allocation is included in peak measurement
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            gc.collect()

            # Only need gradients if running backward pass
            needs_grad = phase in ("backward", "both")
            edge_run = edge.clone().detach().requires_grad_(needs_grad)

            # Phase-aware timing
            if phase == "forward":
                # Forward only - no backward pass
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                loss = None
            elif phase == "backward":
                # Run forward (untimed), then time backward only
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            else:  # phase == "both"
                # Time forward + backward together (original behavior)
                t0 = time.perf_counter()
                v = run_backend_forward(edge_run, struct, use_compile=use_compile)
                loss = v.sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times_ms.append(elapsed_ms)

            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))

            # Clean up (no empty_cache here - only between configs, not repeats)
            del edge_run, v
            if loss is not None:
                del loss
            gc.collect()

        # Compute statistics
        times_sorted = sorted(times_ms)
        n = len(times_sorted)
        median = statistics.median(times_sorted)
        q1 = times_sorted[n // 4] if n >= 4 else times_sorted[0]
        q3 = times_sorted[3 * n // 4] if n >= 4 else times_sorted[-1]

        return BenchmarkResult(
            **result_base,
            time_ms_median=round(median, 2),
            time_ms_iqr_low=round(q1, 2),
            time_ms_iqr_high=round(q3, 2),
            time_per_position_ms=round(median / T, 4),
            peak_allocated_gb=bytes_to_gb(peak_allocated),
            peak_reserved_gb=bytes_to_gb(peak_reserved),
            status="success",
            memory_potentials_gb=breakdown["potentials_gb"],
            memory_dp_state_gb=breakdown["dp_state_gb"],
            memory_workspace_gb=breakdown["workspace_gb"],
            memory_autograd_gb=breakdown["autograd_gb"],
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="oom",
                error_msg=str(e)[:100],
                memory_potentials_gb=breakdown["potentials_gb"],
                memory_dp_state_gb=breakdown["dp_state_gb"],
                memory_workspace_gb=breakdown["workspace_gb"],
                memory_autograd_gb=breakdown["autograd_gb"],
            )
        else:
            return BenchmarkResult(
                **result_base,
                time_ms_median=float("nan"),
                time_ms_iqr_low=float("nan"),
                time_ms_iqr_high=float("nan"),
                time_per_position_ms=float("nan"),
                peak_allocated_gb=float("nan"),
                peak_reserved_gb=float("nan"),
                status="error",
                error_msg=str(e)[:100],
            )
    except NotImplementedError as e:
        return BenchmarkResult(
            **result_base,
            time_ms_median=float("nan"),
            time_ms_iqr_low=float("nan"),
            time_ms_iqr_high=float("nan"),
            time_per_position_ms=float("nan"),
            peak_allocated_gb=float("nan"),
            peak_reserved_gb=float("nan"),
            status="not_implemented",
            error_msg=str(e)[:100],
        )
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--T", type=str, default="128,256,512,1024")
    parser.add_argument("--K", type=str, default="4,8,12,16,20,24")
    parser.add_argument("--C", type=str, default="3,6,9,12")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--sample-configs",
        type=int,
        default=0,
        help=(
            "Sample this many (T, K*C) configs from the full grid, anchored at "
            "min/max BTKC. Use 0 to run the full grid. Ignored if --compile-friendly is set."
        ),
    )
    parser.add_argument(
        "--compile-friendly",
        action="store_true",
        default=False,
        help=(
            "Use compile-aware sampling to minimize torch.compile overhead. "
            "Groups configs by canonical shapes and samples representative configs. "
            "Much faster than full grid when using triton backends with --use-compile."
        ),
    )
    parser.add_argument(
        "--max-canonical-shapes",
        type=int,
        default=8,
        help=(
            "Maximum number of unique canonical shapes to compile when using "
            "--compile-friendly. More shapes = better coverage but longer compile time. "
            "Default: 8"
        ),
    )
    parser.add_argument(
        "--samples-per-shape",
        type=int,
        default=2,
        help=(
            "Number of actual configs to benchmark per canonical shape when using "
            "--compile-friendly. Default: 2"
        ),
    )
    parser.add_argument(
        "--skip-precompile",
        action="store_true",
        default=False,
        help=(
            "Skip the pre-compilation warmup phase. Use this if you have a warm cache "
            "or want to include compilation time in benchmark results."
        ),
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for torch.compile and Triton kernel cache. "
            "On HPC, use local scratch (e.g., /tmp, $TMPDIR) for faster I/O. "
            "Default: --output-dir/.torch_compile_cache"
        ),
    )
    parser.add_argument(
        "--compile-cache-size-gb",
        type=float,
        default=10.0,
        help=(
            "Maximum compile cache size in GB. Older entries are evicted when exceeded. "
            "Set based on available scratch space. Default: 10.0"
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="linear_scan,linear_scan_vectorized,linear_scan_streaming,binary_tree,binary_tree_sharded,block_triangular",
        help=(
            "Comma-separated list of backends. Options: "
            "linear_scan, linear_scan_vectorized, linear_scan_streaming, "
            "binary_tree, binary_tree_sharded, block_triangular, "
            "triton (GPU Triton kernel with torch.compile for training), "
            "triton_pytorch (PyTorch reference), "
            "triton_checkpointing (Triton with gradient checkpointing, for comparison). "
            "Note: triton backends support Log and Max semirings."
        ),
    )
    parser.add_argument(
        "--semirings",
        type=str,
        default="Log",
        help=(
            "Comma-separated list of semirings. Options: Log, Max, Entropy. "
            "Note: triton/triton_pytorch backends support Log and Max semirings."
        ),
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="both",
        help=(
            "Comma-separated list of phases to time. Options: "
            "forward (forward pass only), backward (backward pass only), "
            "both (forward + backward together). Default: both"
        ),
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=True,
        help=(
            "Use torch.compile for triton training backward pass (default). "
            "This generates optimized kernels for both forward and backward. "
            "Disable with --no-use-compile to use gradient checkpointing instead."
        ),
    )
    parser.add_argument(
        "--no-use-compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile for triton (use gradient checkpointing instead).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Skip configs predicted to exceed this memory",
    )
    parser.add_argument(
        "--skip-adjacent-oom",
        action="store_true",
        default=True,
        help="Skip configs if smaller adjacent config OOM'd",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    T_list = parse_int_list(args.T)
    K_list = parse_int_list(args.K)
    C_list = parse_int_list(args.C)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    semirings = [s.strip() for s in args.semirings.split(",") if s.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Validate semirings
    for s in semirings:
        if s not in SEMIRING_MAP:
            print(f"WARNING: Unknown semiring '{s}', available: {list(SEMIRING_MAP.keys())}")

    # Validate phases
    valid_phases = {"forward", "backward", "both"}
    for p in phases:
        if p not in valid_phases:
            print(f"WARNING: Unknown phase '{p}', available: {valid_phases}")

    # Check Triton availability if needed
    if any(b in ("triton", "triton_pytorch", "triton_checkpointing") for b in backends):
        if not HAS_TRITON:
            print("WARNING: Triton not available, triton backends will use PyTorch fallback")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup persistent compile cache for torch.compile
    has_triton_backends = any(
        b in ("triton", "triton_pytorch", "triton_checkpointing") for b in backends
    )
    if args.use_compile and has_triton_backends:
        cache_dir = setup_compile_cache(
            args.output_dir,
            cache_dir=args.compile_cache_dir,
            max_cache_size_gb=args.compile_cache_size_gb,
        )
        print(f"Compile cache: {cache_dir} (max {args.compile_cache_size_gb}GB)")

    torch.manual_seed(42)

    results: list[BenchmarkResult] = []
    # OOM history is tracked per (backend, semiring, phase) combination
    oom_history: dict[str, list[tuple[int, int, int]]] = {}
    for b in backends:
        for s in semirings:
            for p in phases:
                oom_history[f"{b}_{s}_{p}"] = []

    full_config_count = len(T_list) * len(K_list) * len(C_list)

    # Choose sampling strategy
    config_to_canonical: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    canonical_shapes: list[tuple[int, int, int]] = []

    if args.compile_friendly:
        # Compile-aware sampling: minimize unique compiled shapes
        configs, config_to_canonical = sample_compile_friendly(
            T_list,
            K_list,
            C_list,
            max_canonical_shapes=args.max_canonical_shapes,
            samples_per_shape=args.samples_per_shape,
        )
        canonical_shapes = get_canonical_shapes(T_list, K_list, C_list)
        # Filter to only shapes we're actually using
        used_canonical = set(config_to_canonical.values())
        canonical_shapes = [s for s in canonical_shapes if s in used_canonical]
    elif args.sample_configs > 0:
        # Legacy sampling: sample by T and K*C range
        configs = sample_configurations(T_list, K_list, C_list, args.B, args.sample_configs)
        canonical_shapes = get_canonical_shapes(T_list, K_list, C_list)
    else:
        # Full grid
        configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
        canonical_shapes = get_canonical_shapes(T_list, K_list, C_list)

    total_configs = len(configs) * len(backends) * len(semirings) * len(phases)
    completed = 0

    print(f"Running {total_configs} configurations...")
    print(f"Device: {device}")
    print(f"T: {T_list}, K: {K_list}, C: {C_list}, B: {args.B}")
    print(f"Backends: {backends}")
    print(f"Semirings: {semirings}")
    print(f"Phases: {phases}")
    print(f"Repeats: {args.repeats}")
    print(f"Triton use_compile: {args.use_compile}")

    if args.compile_friendly:
        print(
            f"Compile-friendly sampling: {len(configs)} configs mapping to "
            f"{len(canonical_shapes)} canonical shapes"
        )
    elif args.sample_configs > 0 and len(configs) != full_config_count:
        print(f"Sampling {len(configs)} of {full_config_count} T/K/C configs by T and K*C range")

    if canonical_shapes:
        print(f"Canonical shapes for compilation: {len(canonical_shapes)}")

    print("-" * 80)

    # Pre-compile all canonical shapes before timing (separates compile time from runtime)
    if args.use_compile and has_triton_backends and not args.skip_precompile and canonical_shapes:
        precompile_canonical_shapes(canonical_shapes, device, backends, semirings, use_compile=True)

    for backend in backends:
        for semiring_name in semirings:
            # Only reset caches if NOT using compile-friendly mode
            # (compile-friendly pre-compiles everything upfront)
            if (
                args.use_compile
                and backend in ("triton", "triton_pytorch", "triton_checkpointing")
                and not args.compile_friendly
            ):
                reset_compile_caches()
            for phase in phases:
                for T, K, C in configs:
                    KC = K * C
                    completed += 1
                    oom_key = f"{backend}_{semiring_name}_{phase}"

                    # Check backend/semiring compatibility
                    if backend in LOG_SEMIRING_ONLY_BACKENDS and semiring_name != "Log":
                        print(
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: "
                            f"{backend} only supports Log semiring"
                        )
                        results.append(
                            BenchmarkResult(
                                T=T,
                                K=K,
                                C=C,
                                B=args.B,
                                KC=KC,
                                backend=backend,
                                semiring=semiring_name,
                                phase=phase,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="not_supported",
                                error_msg=f"{backend} only supports Log semiring",
                            )
                        )
                        continue

                    # Check triton backend semiring compatibility (Log, Max only)
                    if (
                        backend in ("triton", "triton_pytorch", "triton_checkpointing")
                        and semiring_name not in TRITON_SUPPORTED_SEMIRINGS
                    ):
                        print(
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: "
                            f"{backend} only supports Log/Max semirings"
                        )
                        results.append(
                            BenchmarkResult(
                                T=T,
                                K=K,
                                C=C,
                                B=args.B,
                                KC=KC,
                                backend=backend,
                                semiring=semiring_name,
                                phase=phase,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="not_supported",
                                error_msg=f"{backend} only supports Log/Max semirings",
                            )
                        )
                        continue

                    # Check if we should skip based on OOM history
                    if args.skip_adjacent_oom:
                        # Use backend-only key for memory estimation (semiring doesn't affect memory much)
                        skip, reason = should_skip_config(
                            T,
                            K,
                            C,
                            backend,
                            {backend: oom_history.get(oom_key, [])},
                            args.max_memory_gb,
                        )
                        if skip:
                            print(
                                f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, {backend}/{semiring_name}/{phase}: {reason}"
                            )
                            results.append(
                                BenchmarkResult(
                                    T=T,
                                    K=K,
                                    C=C,
                                    B=args.B,
                                    KC=KC,
                                    backend=backend,
                                    semiring=semiring_name,
                                    phase=phase,
                                    time_ms_median=float("nan"),
                                    time_ms_iqr_low=float("nan"),
                                    time_ms_iqr_high=float("nan"),
                                    time_per_position_ms=float("nan"),
                                    peak_allocated_gb=float("nan"),
                                    peak_reserved_gb=float("nan"),
                                    status="not_tested",
                                    error_msg=reason,
                                )
                            )
                            continue

                    print(
                        f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, {backend}/{semiring_name}/{phase}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_single_benchmark(
                        T,
                        K,
                        C,
                        args.B,
                        backend,
                        device,
                        args.repeats,
                        semiring_name=semiring_name,
                        phase=phase,
                        use_compile=args.use_compile,
                    )
                    results.append(result)

                    if result.status == "success":
                        print(
                            f"OK: {result.time_ms_median:.1f}ms, {result.peak_allocated_gb:.3f}GB allocated, {result.peak_reserved_gb:.3f}GB reserved"
                        )
                    elif result.status == "oom":
                        print("OOM")
                        oom_history[oom_key].append((T, K, C))
                    else:
                        print(f"{result.status}: {result.error_msg}")

    # Save results
    # 1. Full CSV with all metrics
    csv_path = args.output_dir / "benchmark_full.csv"
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nSaved full results to {csv_path}")

    # 2. Heatmap data (for OOM feasibility figures)
    heatmap_path = args.output_dir / "heatmap_data.json"
    heatmap_data = {}
    for r in results:
        key = f"{r.backend}_{r.semiring}_{r.phase}_T{r.T}"
        if key not in heatmap_data:
            heatmap_data[key] = {
                "backend": r.backend,
                "semiring": r.semiring,
                "phase": r.phase,
                "T": r.T,
                "cells": [],
            }
        heatmap_data[key]["cells"].append(
            {
                "K": r.K,
                "C": r.C,
                "KC": r.KC,
                "status": r.status,
                "peak_gb": r.peak_allocated_gb if r.status == "success" else None,
                "time_ms": r.time_ms_median if r.status == "success" else None,
            }
        )
    with open(heatmap_path, "w") as f:
        json.dump(heatmap_data, f, indent=2)
    print(f"Saved heatmap data to {heatmap_path}")

    # 3. Memory breakdown summary
    breakdown_path = args.output_dir / "memory_breakdown.csv"
    breakdown_fields = [
        "backend",
        "semiring",
        "phase",
        "T",
        "K",
        "C",
        "KC",
        "status",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "est_potentials_gb",
        "est_dp_state_gb",
        "est_workspace_gb",
        "est_autograd_gb",
    ]
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=breakdown_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "backend": r.backend,
                    "semiring": r.semiring,
                    "phase": r.phase,
                    "T": r.T,
                    "K": r.K,
                    "C": r.C,
                    "KC": r.KC,
                    "status": r.status,
                    "peak_allocated_gb": r.peak_allocated_gb,
                    "peak_reserved_gb": r.peak_reserved_gb,
                    "est_potentials_gb": r.memory_potentials_gb,
                    "est_dp_state_gb": r.memory_dp_state_gb,
                    "est_workspace_gb": r.memory_workspace_gb,
                    "est_autograd_gb": r.memory_autograd_gb,
                }
            )
    print(f"Saved memory breakdown to {breakdown_path}")

    # 4. Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for backend in backends:
        for semiring_name in semirings:
            for phase in phases:
                key_results = [
                    r
                    for r in results
                    if r.backend == backend and r.semiring == semiring_name and r.phase == phase
                ]
                if not key_results:
                    continue

                success = sum(1 for r in key_results if r.status == "success")
                oom = sum(1 for r in key_results if r.status == "oom")
                skipped = sum(1 for r in key_results if r.status in ("not_tested", "not_supported"))

                successful = [r for r in key_results if r.status == "success"]
                if successful:
                    max_kc = max(r.KC for r in successful)
                    max_mem = max(r.peak_allocated_gb for r in successful)
                else:
                    max_kc = 0
                    max_mem = 0

                label = f"{backend}/{semiring_name}/{phase}"
                print(
                    f"{label:40s}: {success:3d} success, {oom:3d} OOM, {skipped:3d} skipped | max KC={max_kc:4d}, max mem={max_mem:.2f}GB"
                )


if __name__ == "__main__":
    main()
