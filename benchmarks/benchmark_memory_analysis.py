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

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lib import (
    BenchmarkResult,
    get_canonical_shapes,
    precompile_canonical_shapes,
    print_summary,
    reset_compile_caches,
    run_single_benchmark,
    sample_compile_friendly,
    sample_configurations,
    save_results,
    setup_compile_cache,
    should_skip_config,
)
from lib.runner import LOG_SEMIRING_ONLY_BACKENDS, SEMIRING_MAP, TRITON_SUPPORTED_SEMIRINGS
from lib.sampling import parse_int_list

from torch_semimarkov.triton_scan import HAS_TRITON


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
        "--precompile-timeout",
        type=int,
        default=120,
        help=(
            "Timeout in seconds for compiling each shape during pre-compilation. "
            "Shapes that exceed this timeout will be skipped and compile on-demand. "
            "Set to 0 to disable timeout. Default: 120"
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
        "--compile-threads",
        type=int,
        default=None,
        help=(
            "Number of CPU threads for torch.compile. Default: all available. "
            "On HPC, set to match your CPU allocation (e.g., --compile-threads 8)."
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="triton,triton_pytorch,linear_scan_streaming",
        help=(
            "Comma-separated list of backends. Options: "
            "linear_scan_streaming (PyTorch reference), "
            "triton (GPU Triton kernel with torch.compile for training), "
            "triton_pytorch (PyTorch reference for Triton), "
            "triton_checkpointing (Triton with gradient checkpointing). "
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
            compile_threads=args.compile_threads,
        )
        threads_str = str(args.compile_threads) if args.compile_threads else "all"
        print(
            f"Compile cache: {cache_dir} (max {args.compile_cache_size_gb}GB, {threads_str} threads)"
        )

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
        precompile_canonical_shapes(
            canonical_shapes,
            device,
            backends,
            semirings,
            use_compile=True,
            timeout_seconds=args.precompile_timeout,
        )

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
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                            f"{backend}/{semiring_name}/{phase}: "
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
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                            f"{backend}/{semiring_name}/{phase}: "
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
                        # Use backend-only key for memory estimation
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
                                f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                                f"{backend}/{semiring_name}/{phase}: {reason}"
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
                        f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, "
                        f"{backend}/{semiring_name}/{phase}...",
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
                            f"OK: {result.time_ms_median:.1f}ms, "
                            f"{result.peak_allocated_gb:.3f}GB allocated, "
                            f"{result.peak_reserved_gb:.3f}GB reserved"
                        )
                    elif result.status == "oom":
                        print("OOM")
                        oom_history[oom_key].append((T, K, C))
                    else:
                        print(f"{result.status}: {result.error_msg}")

    # Save results and print summary
    save_results(results, args.output_dir, backends, semirings, phases)
    print_summary(results, backends, semirings, phases)


if __name__ == "__main__":
    main()
