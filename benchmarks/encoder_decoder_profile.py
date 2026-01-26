#!/usr/bin/env python
"""Profile encoder-decoder pipeline to identify bottlenecks.

Measures relative time spent in:
- Encoder (Mamba/Transformer/Stub)
- CRF forward pass
- CRF backward pass

Usage:
    # Quick test on CPU (development)
    python benchmarks/encoder_decoder_profile.py --device cpu --config short

    # Full profiling on GPU
    python benchmarks/encoder_decoder_profile.py --device cuda --config all

    # With TensorBoard trace
    python benchmarks/encoder_decoder_profile.py --device cuda --tensorboard
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mamba_encoder_stub import get_encoder

from torch_semimarkov import SemiMarkovCRFHead


@dataclass
class ProfileConfig:
    """Configuration for a profiling run."""

    name: str
    T: int  # Sequence length
    K: int  # Max segment duration
    C: int  # Number of classes
    batch: int
    use_case: str


# Standard test configurations from the roadmap
CONFIGS = {
    "short": ProfileConfig(name="short", T=1000, K=16, C=24, batch=32, use_case="Typical NLP"),
    "medium": ProfileConfig(name="medium", T=10000, K=32, C=24, batch=16, use_case="Speech/Bio"),
    "long": ProfileConfig(name="long", T=100000, K=64, C=48, batch=4, use_case="Genomics"),
    "inference": ProfileConfig(
        name="inference", T=10000, K=16, C=24, batch=1, use_case="Single-sample"
    ),
}


class EncoderDecoderPipeline(nn.Module):
    """Encoder-decoder pipeline for profiling."""

    def __init__(
        self,
        encoder_type: str = "stub",
        d_model: int = 512,
        n_layers: int = 12,
        num_classes: int = 24,
        max_duration: int = 16,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_type=encoder_type,
            d_model=d_model,
            n_layer=n_layers,
            device=device,
            dtype=dtype,
        )

        self.crf = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=d_model,
        )
        if device is not None:
            self.crf = self.crf.to(device)
        if dtype is not None:
            self.crf = self.crf.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_triton: bool = True,
    ):
        """Forward pass through encoder and CRF.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            lengths: Sequence lengths (batch,)
            labels: Optional labels for loss computation (batch, seq_len)
            use_triton: Whether to use Triton kernel (if available)

        Returns:
            Loss if labels provided, else CRF output dict
        """
        hidden = self.encoder(x)
        if labels is not None:
            return self.crf.compute_loss(hidden, lengths, labels, use_triton=use_triton)
        return self.crf(hidden, lengths, use_triton=use_triton)


def profile_with_timer(
    pipeline: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    warmup: int = 5,
    iterations: int = 20,
    use_triton: bool = True,
) -> dict:
    """Profile using simple timing (works on CPU and GPU).

    Returns dict with timing breakdown.
    """
    device = x.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        loss = pipeline(x, lengths, labels, use_triton=use_triton)
        loss.backward()
        pipeline.zero_grad()
        sync()

    # Profile encoder
    encoder_times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        _ = pipeline.encoder(x)
        sync()
        encoder_times.append(time.perf_counter() - start)

    # Profile full forward
    forward_times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        loss = pipeline(x, lengths, labels, use_triton=use_triton)
        sync()
        forward_times.append(time.perf_counter() - start)
        # Clean up
        del loss

    # Profile full forward + backward
    total_times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        loss = pipeline(x, lengths, labels, use_triton=use_triton)
        loss.backward()
        sync()
        total_times.append(time.perf_counter() - start)
        pipeline.zero_grad()

    # Compute statistics
    encoder_mean = sum(encoder_times) / len(encoder_times)
    forward_mean = sum(forward_times) / len(forward_times)
    total_mean = sum(total_times) / len(total_times)

    crf_forward_mean = forward_mean - encoder_mean
    backward_mean = total_mean - forward_mean

    return {
        "encoder_ms": encoder_mean * 1000,
        "crf_forward_ms": crf_forward_mean * 1000,
        "backward_ms": backward_mean * 1000,
        "total_ms": total_mean * 1000,
        "encoder_pct": encoder_mean / total_mean * 100,
        "crf_forward_pct": crf_forward_mean / total_mean * 100,
        "backward_pct": backward_mean / total_mean * 100,
    }


def profile_with_profiler(
    pipeline: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Optional[Path] = None,
    warmup: int = 5,
    iterations: int = 20,
    use_triton: bool = True,
) -> dict:
    """Profile using torch.profiler (CUDA only, more detailed).

    Returns dict with timing breakdown and optionally saves TensorBoard trace.
    """
    from torch.profiler import ProfilerActivity, profile, schedule

    device = x.device
    if device.type != "cuda":
        print("Warning: torch.profiler works best with CUDA. Falling back to timer.")
        return profile_with_timer(pipeline, x, lengths, labels, warmup, iterations, use_triton)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # Setup trace handler if output dir provided
    trace_handler = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def trace_handler(prof):
            prof.export_chrome_trace(str(output_dir / "trace.json"))

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=warmup, active=iterations, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(1 + warmup + iterations):
            loss = pipeline(x, lengths, labels, use_triton=use_triton)
            loss.backward()
            pipeline.zero_grad()
            prof.step()

    # Parse profiler results
    # This is approximate - kernel names may vary
    table = prof.key_averages()

    encoder_time = 0
    crf_time = 0
    total_cuda_time = 0

    for event in table:
        cuda_time = event.cuda_time_total / 1000  # Convert to ms
        total_cuda_time += cuda_time

        # Heuristic: classify by kernel name
        name = event.key.lower()
        if "mamba" in name or "conv1d" in name or "silu" in name:
            encoder_time += cuda_time
        elif "semi_crf" in name or "streaming" in name or "logsumexp" in name:
            crf_time += cuda_time

    # Print detailed table
    print("\nProfiler Summary:")
    print(table.table(sort_by="cuda_time_total", row_limit=20))

    # Fallback to timer-based measurement for accurate component breakdown
    timer_results = profile_with_timer(pipeline, x, lengths, labels, warmup, iterations, use_triton)

    return timer_results


def run_profiling(
    config: ProfileConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    encoder_type: str = "stub",
    use_profiler: bool = False,
    output_dir: Optional[Path] = None,
    use_triton: bool = True,
) -> dict:
    """Run profiling for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Configuration: {config.name} ({config.use_case})")
    print(f"  T={config.T}, K={config.K}, C={config.C}, batch={config.batch}")
    print(f"  Device: {device}, Encoder: {encoder_type}")
    print(f"{'='*60}")

    # Check if configuration is feasible
    if device.type != "cuda" and config.T > 10000:
        print(f"  Skipping: T={config.T} too large for CPU profiling")
        return {"skipped": True, "reason": "T too large for CPU"}

    # Create pipeline
    pipeline = EncoderDecoderPipeline(
        encoder_type=encoder_type,
        d_model=512,
        n_layers=12,
        num_classes=config.C,
        max_duration=config.K,
        device=device,
        dtype=dtype,
    )

    # Create inputs
    x = torch.randn(config.batch, config.T, 512, device=device, dtype=dtype)
    lengths = torch.full((config.batch,), config.T, device=device, dtype=torch.long)
    labels = torch.randint(0, config.C, (config.batch, config.T), device=device)

    # Determine if Triton is available
    actual_use_triton = use_triton and device.type == "cuda"
    try:
        from torch_semimarkov.streaming import HAS_TRITON

        actual_use_triton = actual_use_triton and HAS_TRITON
    except ImportError:
        actual_use_triton = False

    print(f"  Using Triton: {actual_use_triton}")

    # Run profiling
    try:
        if use_profiler and device.type == "cuda":
            results = profile_with_profiler(
                pipeline,
                x,
                lengths,
                labels,
                output_dir=output_dir,
                use_triton=actual_use_triton,
            )
        else:
            results = profile_with_timer(pipeline, x, lengths, labels, use_triton=actual_use_triton)

        # Add config info
        results["config"] = config.name
        results["T"] = config.T
        results["K"] = config.K
        results["C"] = config.C
        results["batch"] = config.batch
        results["device"] = str(device)
        results["use_triton"] = actual_use_triton

        # Print results
        print("\n  Results:")
        print(f"    Encoder:     {results['encoder_ms']:8.2f} ms ({results['encoder_pct']:5.1f}%)")
        print(
            f"    CRF Forward: {results['crf_forward_ms']:8.2f} ms ({results['crf_forward_pct']:5.1f}%)"
        )
        print(
            f"    Backward:    {results['backward_ms']:8.2f} ms ({results['backward_pct']:5.1f}%)"
        )
        print(f"    Total:       {results['total_ms']:8.2f} ms")

        # Decision guidance
        crf_total_pct = results["crf_forward_pct"] + results["backward_pct"]
        if crf_total_pct < 20:
            print(f"\n  Decision: CRF is NOT the bottleneck ({crf_total_pct:.1f}% of total)")
        elif crf_total_pct > 30:
            print(f"\n  Decision: CRF IS the bottleneck ({crf_total_pct:.1f}% of total)")
            print("            Optimization would benefit this configuration.")
        else:
            print(f"\n  Decision: CRF is borderline ({crf_total_pct:.1f}% of total)")

        return results

    except Exception as e:
        print(f"  Error: {e}")
        return {"skipped": True, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Profile encoder-decoder pipeline")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to profile on (cuda or cpu)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="short",
        choices=list(CONFIGS.keys()) + ["all"],
        help="Configuration to profile",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="stub",
        choices=["stub", "mamba", "transformer"],
        help="Encoder type",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Generate TensorBoard trace (CUDA only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiling_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernel (use PyTorch reference)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Encoder-Decoder Pipeline Profiling")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Triton: {'disabled' if args.no_triton else 'enabled'}")

    # Determine configs to run
    if args.config == "all":
        configs_to_run = list(CONFIGS.values())
    else:
        configs_to_run = [CONFIGS[args.config]]

    # Run profiling
    all_results = []
    for config in configs_to_run:
        results = run_profiling(
            config=config,
            device=device,
            encoder_type=args.encoder,
            use_profiler=args.tensorboard,
            output_dir=output_dir / config.name if args.tensorboard else None,
            use_triton=not args.no_triton,
        )
        all_results.append(results)

    # Save results
    results_file = output_dir / "profiling_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    bottleneck_configs = []
    non_bottleneck_configs = []

    for result in all_results:
        if result.get("skipped"):
            continue

        crf_pct = result["crf_forward_pct"] + result["backward_pct"]
        if crf_pct > 30:
            bottleneck_configs.append((result["config"], crf_pct))
        else:
            non_bottleneck_configs.append((result["config"], crf_pct))

    if bottleneck_configs:
        print("\nConfigs where CRF IS the bottleneck (>30%):")
        for name, pct in bottleneck_configs:
            print(f"  {name}: {pct:.1f}%")

    if non_bottleneck_configs:
        print("\nConfigs where CRF is NOT the bottleneck (<30%):")
        for name, pct in non_bottleneck_configs:
            print(f"  {name}: {pct:.1f}%")

    print("\nRecommendation:")
    if bottleneck_configs:
        print("  CRF optimization would benefit at least some configurations.")
        print("  Proceed with Phase 2 (K-parallelization) of the roadmap.")
    else:
        print("  CRF is not a significant bottleneck in any configuration.")
        print("  Consider whether optimization is worth the effort.")


if __name__ == "__main__":
    main()
