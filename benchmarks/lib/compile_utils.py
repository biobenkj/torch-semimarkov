"""torch.compile cache and precompilation utilities."""

from __future__ import annotations

import gc
import os
import signal
from pathlib import Path

import torch
import torch._inductor.config as inductor_config


class CompileTimeoutError(Exception):
    """Raised when compilation exceeds timeout."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for compilation timeout."""
    raise CompileTimeoutError("Compilation timed out")


# Semirings supported by triton backends
TRITON_SUPPORTED_SEMIRINGS = {"Log", "Max"}


def reset_compile_caches() -> None:
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


def setup_compile_cache(
    output_dir: Path,
    cache_dir: Path | None = None,
    max_cache_size_gb: float = 10.0,
    compile_threads: int | None = None,
) -> Path:
    """
    Configure persistent compilation cache to avoid recompilation across runs.

    Args:
        output_dir: Default location for cache if cache_dir not specified
        cache_dir: Explicit cache directory (e.g., /tmp for HPC local scratch)
        max_cache_size_gb: Maximum cache size in GB (default 10GB)
        compile_threads: Number of CPU threads for compilation (None = all available)

    Returns the cache directory path.

    HPC Notes:
        - Use local scratch (e.g., /tmp, $TMPDIR, or node-local NVMe) for faster I/O
        - On shared filesystems, compile cache can cause lock contention
        - Set max_cache_size_gb based on available scratch space
        - Set compile_threads to match your CPU allocation
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

    # Limit compilation threads (important for HPC job schedulers)
    if compile_threads is not None:
        inductor_config.compile_threads = compile_threads

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
    timeout_seconds: int = 120,
) -> list[tuple[int, int, int]]:
    """
    Pre-compile kernels for all canonical shapes before timing.

    This separates compilation time from benchmark timing, giving more
    accurate runtime measurements and avoiding compilation during timed runs.

    Args:
        canonical_shapes: List of (T, K, C) shapes to precompile
        device: CUDA device to compile for
        backends: List of backend names
        semirings: List of semiring names
        use_compile: Whether to use torch.compile
        timeout_seconds: Max seconds per shape before skipping (0 = no timeout)

    Returns:
        List of shapes that were successfully compiled (excludes timed-out shapes)
    """
    from torch_semimarkov.triton_scan import semi_crf_triton_forward

    triton_backends = {"triton", "triton_pytorch", "triton_checkpointing"}
    active_triton_backends = [b for b in backends if b in triton_backends]

    if not active_triton_backends or not use_compile:
        return list(canonical_shapes)

    # Triton/torch.compile for GPU only - skip if on CPU
    if device.type != "cuda":
        print(f"\nSkipping pre-compilation (device={device}, not CUDA)")
        return list(canonical_shapes)

    # Ensure CUDA is ready
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    print(f"\nPre-compiling {len(canonical_shapes)} canonical shapes on {device}...")
    print(f"  Backends: {active_triton_backends}")
    if timeout_seconds > 0:
        print(f"  Timeout: {timeout_seconds}s per shape (shapes that hang will be skipped)")
    print("  This may take a few minutes on first run (cached afterward)\n")

    successful_shapes: list[tuple[int, int, int]] = []
    skipped_shapes: list[tuple[int, int, int]] = []

    for i, (T, K, C) in enumerate(canonical_shapes):
        print(
            f"  [{i+1}/{len(canonical_shapes)}] Shape T={T}, K={K}, C={C}...",
            end=" ",
            flush=True,
        )

        shape_success = True

        # Set up timeout if requested (Unix only)
        if timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)

        try:
            for semiring_name in semirings:
                if semiring_name not in TRITON_SUPPORTED_SEMIRINGS:
                    continue

                triton_semiring = semiring_name.lower()

                # Use B=1 for faster compilation (shape is what matters)
                # Explicitly create on CUDA device
                with torch.cuda.device(device):
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

                            # Ensure kernels are actually compiled (not just queued)
                            torch.cuda.synchronize(device)

                        except CompileTimeoutError:
                            raise  # Re-raise to be caught by outer handler
                        except Exception as e:
                            print(f"(warn: {backend}/{semiring_name}: {str(e)[:30]})", end=" ")

                    del edge, lengths
                    torch.cuda.empty_cache()

        except CompileTimeoutError:
            print(f"TIMEOUT (>{timeout_seconds}s) - skipping", end=" ")
            shape_success = False
            skipped_shapes.append((T, K, C))
            # Reset CUDA state after timeout
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()

        finally:
            # Cancel the alarm and restore old handler
            if timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        if shape_success:
            successful_shapes.append((T, K, C))
            print("done")
        else:
            print("")

    # Summary
    if skipped_shapes:
        print(f"\nWARNING: {len(skipped_shapes)} shape(s) timed out and were skipped:")
        for T, K, C in skipped_shapes:
            print(f"  - T={T}, K={K}, C={C}")
        print("These shapes will compile on-demand during benchmarking (may be slow).\n")

    print(
        f"\nPre-compilation complete. {len(successful_shapes)}/{len(canonical_shapes)} shapes ready.\n"
    )
    gc.collect()
    torch.cuda.empty_cache()

    return successful_shapes
