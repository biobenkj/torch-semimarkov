"""Mamba encoder stubs for profiling without mamba-ssm dependency.

Provides CPU-compatible stubs that match Mamba's API and approximate
compute profile. Use for development/testing on machines without GPU.

For actual profiling, replace with real Mamba encoder on CUDA machine.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlockStub(nn.Module):
    """Single Mamba block stub matching Mamba's API.

    Approximates compute pattern without SSM-specific ops.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection (like Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False, **factory_kwargs)

        # Convolution (like Mamba's conv1d)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
            **factory_kwargs,
        )

        # SSM parameters (stub: use linear + activation to approximate compute)
        # Real Mamba has selective scan here
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner, bias=False, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

        # Layer norm
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching Mamba block pattern.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # In projection: split into x and z branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Convolution (transpose for conv1d)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, : x.shape[1]]  # Causal padding
        x_branch = x_branch.transpose(1, 2)

        # Activation
        x_branch = F.silu(x_branch)

        # SSM approximation (real Mamba has selective scan here)
        # This is a rough approximation of compute load
        x_branch = self.ssm_proj(x_branch)
        x_branch = F.silu(x_branch)

        # Gating
        x_branch = x_branch * F.silu(z)

        # Output projection
        out = self.out_proj(x_branch)

        return residual + out


class MambaEncoderStub(nn.Module):
    """Multi-layer Mamba encoder stub.

    Matches mamba-ssm Mamba class API for drop-in replacement.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layer: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.n_layer = n_layer

        # Stack of Mamba blocks
        self.layers = nn.ModuleList(
            [
                MambaBlockStub(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(n_layer)
            ]
        )

        # Final norm
        self.norm_f = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


def get_encoder(
    encoder_type: str = "stub",
    d_model: int = 512,
    n_layer: int = 12,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Factory function to get encoder by type.

    Args:
        encoder_type: One of "stub", "mamba", "transformer"
        d_model: Model dimension
        n_layer: Number of layers
        device: Target device
        dtype: Target dtype

    Returns:
        Encoder module
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    if encoder_type == "stub":
        return MambaEncoderStub(
            d_model=d_model,
            n_layer=n_layer,
            **factory_kwargs,
        )

    elif encoder_type == "mamba":
        try:
            from mamba_ssm import Mamba

            return Mamba(
                d_model=d_model,
                n_layer=n_layer,
                **factory_kwargs,
            )
        except ImportError:
            raise ImportError(
                "mamba-ssm not installed. Use encoder_type='stub' for development "
                "or install mamba-ssm: pip install mamba-ssm"
            )

    elif encoder_type == "transformer":
        # Simple transformer encoder for comparison
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
            **factory_kwargs,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def estimate_mamba_flops(batch: int, seq_len: int, d_model: int, n_layer: int) -> int:
    """Estimate FLOPs for Mamba encoder.

    Rough approximation based on Mamba paper.
    """
    expand = 2
    d_inner = d_model * expand
    d_state = 16
    d_conv = 4

    # Per layer:
    # - in_proj: 2 * seq_len * d_model * d_inner * 2
    # - conv1d: seq_len * d_inner * d_conv
    # - SSM: seq_len * d_inner * d_state * 2 (approx)
    # - out_proj: seq_len * d_inner * d_model

    flops_per_layer = seq_len * (
        2 * d_model * d_inner * 2  # in_proj
        + d_inner * d_conv  # conv
        + d_inner * d_state * 2  # ssm
        + d_inner * d_model  # out_proj
    )

    return batch * n_layer * flops_per_layer


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Testing on {device}")

    # Create encoder
    encoder = MambaEncoderStub(d_model=512, n_layer=12, device=device, dtype=dtype)

    # Test forward
    batch, seq_len = 4, 1000
    x = torch.randn(batch, seq_len, 512, device=device, dtype=dtype)

    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Estimate FLOPs
    flops = estimate_mamba_flops(batch, seq_len, 512, 12)
    print(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs")

    # Test backward
    loss = out.sum()
    loss.backward()
    print("Backward pass successful")
