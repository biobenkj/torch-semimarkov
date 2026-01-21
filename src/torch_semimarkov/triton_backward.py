r"""Semi-CRF backward pass implementations.

This module re-exports all backward pass functionality from the split modules:
- backward.py: PyTorch reference implementation
- checkpointed.py: Memory-efficient checkpointed implementations (RECOMMENDED)

For production use with long sequences (T up to 400K+), use the checkpointed
implementations which have O(√(T×K) × K × C) memory instead of O(T × C).

Recommended entry point:
    from torch_semimarkov.triton_backward import semi_crf_triton_checkpointed_backward

For testing/verification:
    from torch_semimarkov.backward import semi_crf_backward_pytorch
"""

# Re-export from backward.py (reference implementation)
from .backward import (
    HAS_TRITON,
    # Constants
    NEG_INF,
    SemiCRFBackward,
    # Helper functions
    _next_power_of_2,
    semi_crf_backward_beta,
    semi_crf_backward_pytorch,
    semi_crf_compute_marginals,
    semi_crf_forward_backward,
    # PyTorch reference implementations
    semi_crf_forward_with_alpha,
)

# Re-export from checkpointed.py (production implementation)
from .checkpointed import (
    SemiCRFOptimizedCheckpointedBackward,
    # Triton checkpointed (GPU-accelerated, RECOMMENDED)
    SemiCRFTritonCheckpointedBackward,
    # Helper functions
    _compute_checkpoint_interval,
    semi_crf_backward_from_ring_checkpoints,
    # Optimized checkpointed (O(T) compute)
    semi_crf_forward_with_ring_checkpoints,
    semi_crf_optimized_checkpointed_backward,
    semi_crf_triton_checkpointed_backward,
)

# Conditionally re-export Triton kernel launchers (only available when Triton is installed)
if HAS_TRITON:
    pass

__all__ = [
    # Constants
    "NEG_INF",
    "HAS_TRITON",
    # Helper functions
    "_next_power_of_2",
    "_compute_checkpoint_interval",
    # PyTorch reference
    "semi_crf_forward_with_alpha",
    "semi_crf_backward_beta",
    "semi_crf_compute_marginals",
    "semi_crf_backward_pytorch",
    "SemiCRFBackward",
    "semi_crf_forward_backward",
    # Optimized checkpointed
    "semi_crf_forward_with_ring_checkpoints",
    "semi_crf_backward_from_ring_checkpoints",
    "SemiCRFOptimizedCheckpointedBackward",
    "semi_crf_optimized_checkpointed_backward",
    # Triton checkpointed (RECOMMENDED)
    "SemiCRFTritonCheckpointedBackward",
    "semi_crf_triton_checkpointed_backward",
]

# Add Triton-specific exports when available
if HAS_TRITON:
    __all__.extend(
        [
            "_semi_crf_ckpt_segment_forward_kernel",
            "_semi_crf_ckpt_segment_backward_kernel",
            "launch_triton_checkpointed_backward_kernel",
        ]
    )
