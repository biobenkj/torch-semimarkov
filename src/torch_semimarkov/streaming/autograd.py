r"""Autograd functions and public API for streaming Semi-CRF."""

from typing import Optional

import torch

from ..validation import validate_cum_scores, validate_lengths
from .pytorch_reference import (
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)

# Triton imports are conditional
try:
    from .triton_backward import (
        launch_streaming_triton_backward,
        launch_streaming_triton_backward_fused,
    )
    from .triton_forward import HAS_TRITON, launch_streaming_triton_kernel
except ImportError:
    HAS_TRITON = False
    launch_streaming_triton_kernel = None
    launch_streaming_triton_backward = None
    launch_streaming_triton_backward_fused = None


class SemiCRFStreaming(torch.autograd.Function):
    r"""Pure PyTorch autograd function for streaming Semi-CRF. O(KC) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        proj_start: Optional[torch.Tensor] = None,
        proj_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Detach inputs for forward computation
        partition, ring_checkpoints, checkpoint_interval = semi_crf_streaming_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
            K,
            semiring,
            proj_start.detach() if proj_start is not None else None,
            proj_end.detach() if proj_end is not None else None,
        )

        # Save for backward
        ctx.save_for_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            partition,
            proj_start,
            proj_end,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = checkpoint_interval

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            partition,
            proj_start,
            proj_end,
        ) = ctx.saved_tensors

        # Validate partition from forward pass before running backward
        # If partition is already NaN/Inf, backward will produce garbage
        if not torch.isfinite(partition).all():
            nan_count = torch.isnan(partition).sum().item()
            inf_count = torch.isinf(partition).sum().item()
            raise RuntimeError(
                f"Non-finite partition from forward pass (PyTorch): "
                f"{nan_count} NaN, {inf_count} Inf. "
                f"Check forward pass numerical stability."
            )

        grads = semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ctx.K,
            partition,
            ring_checkpoints,
            ctx.checkpoint_interval,
            ctx.semiring,
            proj_start,
            proj_end,
        )

        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end = grads

        # Validate backward outputs - catch NaN/Inf before they corrupt parameters
        # This helps debug stochastic NaN issues in training
        if not torch.isfinite(grad_cum_scores).all():
            nan_count = torch.isnan(grad_cum_scores).sum().item()
            inf_count = torch.isinf(grad_cum_scores).sum().item()
            raise RuntimeError(
                f"Non-finite values in CRF backward (PyTorch): "
                f"grad_cum_scores has {nan_count} NaN, {inf_count} Inf"
            )

        # Scale by upstream gradient
        #
        # Per-batch parameters (cum_scores, proj_start, proj_end):
        #   Scale each batch element by its grad_output[b]
        #
        # Shared parameters (transition, duration_bias):
        #   These now come as per-batch tensors: (batch, C, C) or (batch, K, C, C)
        #   We apply weighted sum: grad = Σ_b[grad_output[b] × grad_per_batch[b]]
        #   Using einsum for memory efficiency (avoids large intermediate tensor)
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum (memory-efficient)
        # Notation: b=batch, k=duration, i=src_state, j=dst_state, c=state
        if grad_transition.ndim == 3:  # (batch, C, C) - static transitions
            grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
        else:  # (batch, K, C, C) - duration-dependent
            grad_transition = torch.einsum("bkij, b -> kij", grad_transition, grad_output)

        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start * scale
        if grad_proj_end is not None:
            grad_proj_end = grad_proj_end * scale

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
            grad_proj_start,
            grad_proj_end,
        )


class SemiCRFStreamingTriton(torch.autograd.Function):
    r"""Triton-accelerated autograd function for streaming Semi-CRF. O(KC) memory.

    Args:
        use_fused_backward: If True, use the fused backward kernel which eliminates
            atomic operations for transition and duration_bias gradients. This provides
            10-50x speedup on backward pass. Only works with static transitions (C, C).
            Default: True for static transitions, False for duration-dependent (K, C, C).
    """

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        proj_start: Optional[torch.Tensor] = None,
        proj_end: Optional[torch.Tensor] = None,
        accum_dtype: torch.dtype = torch.float64,
        num_warps: int = 4,
        use_fused_backward: Optional[bool] = None,
    ) -> torch.Tensor:
        # Use Triton kernel for forward
        partition, ring_checkpoints, checkpoint_interval = launch_streaming_triton_kernel(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
            K,
            semiring,
            proj_start=proj_start.detach() if proj_start is not None else None,
            proj_end=proj_end.detach() if proj_end is not None else None,
            num_warps=num_warps,
        )

        # Save for backward
        ctx.save_for_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            partition,
            proj_start,
            proj_end,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = checkpoint_interval
        ctx.accum_dtype = accum_dtype
        ctx.num_warps = num_warps

        # Determine whether to use fused backward
        # Default: use fused for static transitions (2D), original for duration-dependent (3D)
        if use_fused_backward is None:
            ctx.use_fused_backward = transition.ndim == 2
        else:
            ctx.use_fused_backward = use_fused_backward

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            partition,
            proj_start,
            proj_end,
        ) = ctx.saved_tensors

        # Validate partition from forward pass before running backward
        # If partition is already NaN/Inf, backward will produce garbage
        if not torch.isfinite(partition).all():
            nan_count = torch.isnan(partition).sum().item()
            inf_count = torch.isinf(partition).sum().item()
            raise RuntimeError(
                f"Non-finite partition from forward pass (Triton): "
                f"{nan_count} NaN, {inf_count} Inf. "
                f"Check forward pass numerical stability."
            )

        # Use Triton backward kernel for gradient computation
        # The kernel already scales by grad_output internally
        # Note: launch functions return 6 values; the 6th (boundary_marginals) is unused here
        if ctx.use_fused_backward:
            # Fused backward: eliminates atomics for transition/duration_bias gradients
            # Only works with static transitions (C, C)
            (
                grad_cum_scores,
                grad_transition,
                grad_duration_bias,
                grad_proj_start,
                grad_proj_end,
                _,
            ) = launch_streaming_triton_backward_fused(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                ctx.checkpoint_interval,
                grad_output,
                proj_start=proj_start,
                proj_end=proj_end,
                accum_dtype=ctx.accum_dtype,
                num_warps=ctx.num_warps,
            )
        else:
            # Original backward: uses atomics, supports duration-dependent transitions
            (
                grad_cum_scores,
                grad_transition,
                grad_duration_bias,
                grad_proj_start,
                grad_proj_end,
                _,
            ) = launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                ctx.checkpoint_interval,
                grad_output,
                proj_start=proj_start,
                proj_end=proj_end,
                accum_dtype=ctx.accum_dtype,
                num_warps=ctx.num_warps,
            )

        # Validate ALL backward outputs - catch NaN/Inf before they corrupt parameters
        # This helps debug stochastic NaN issues in Triton kernels
        for name, tensor in [
            ("grad_cum_scores", grad_cum_scores),
            ("grad_transition", grad_transition),
            ("grad_duration_bias", grad_duration_bias),
        ]:
            if not torch.isfinite(tensor).all():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                raise RuntimeError(
                    f"Non-finite values in CRF backward (Triton): "
                    f"{name} has {nan_count} NaN, {inf_count} Inf"
                )

        if grad_proj_start is not None:
            for name, tensor in [
                ("grad_proj_start", grad_proj_start),
                ("grad_proj_end", grad_proj_end),
            ]:
                if not torch.isfinite(tensor).all():
                    nan_count = torch.isnan(tensor).sum().item()
                    inf_count = torch.isinf(tensor).sum().item()
                    raise RuntimeError(
                        f"Non-finite values in CRF backward (Triton): "
                        f"{name} has {nan_count} NaN, {inf_count} Inf"
                    )

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
            grad_proj_start,
            grad_proj_end,
            None,  # accum_dtype
            None,  # num_warps
            None,  # use_fused_backward
        )


def semi_crf_streaming_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    use_compile: bool = False,  # Deprecated, kept for API compatibility
    accum_dtype: torch.dtype = torch.float64,
    num_warps: int = 4,
) -> torch.Tensor:
    r"""Compute Semi-CRF partition function with streaming edge computation.

    O(KC) memory (ring buffer), O(T×K×C²) compute. Uses Triton kernels on GPU.

    .. warning::
        ``cum_scores`` **MUST** be float32 for numerical stability at T > 100K.
        Zero-centering before cumsum is critical to prevent precision loss.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`. Must be float32 and zero-centered
            before cumsum for numerical stability.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent
            transitions. ``transition[c_src, c_dest]`` is the score for
            c_src → c_dest.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
            Required to compensate for sum-pooling length bias.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` (logsumexp for partition) or ``"max"``
            (Viterbi). Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        use_triton (bool, optional): If ``True``, use Triton kernels when available.
            Default: ``True``
        accum_dtype (torch.dtype, optional): Dtype for gradient accumulation in backward.
            Use ``torch.float64`` (default) for numerical stability at batch >= 128.
            Use ``torch.float32`` for lower memory at batch <= 64.
        num_warps (int, optional): Number of warps per block for Triton kernels.
            Higher values increase parallelism but also register pressure.
            Recommended range: 2-8. Default: ``4``

    Returns:
        Tensor: Log partition function (or max score) of shape :math:`(\text{batch},)`.

    Examples::

        >>> import torch
        >>> from torch_semimarkov.streaming import semi_crf_streaming_forward
        >>>
        >>> # Encoder output
        >>> batch, T, hidden_dim, C, K = 2, 100, 64, 4, 8
        >>> h = torch.randn(batch, T, hidden_dim)  # encoder output
        >>> W_content = torch.randn(hidden_dim, C)
        >>>
        >>> # Pre-project to label space (loop-invariant: outside kernel)
        >>> projected = h @ W_content  # (batch, T, C)
        >>>
        >>> # CRITICAL: Zero-center before cumsum
        >>> projected = projected - projected.mean(dim=1, keepdim=True)
        >>>
        >>> # Cumsum in float32
        >>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
        >>> cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
        >>>
        >>> # Model parameters
        >>> transition = torch.randn(C, C) * 0.1
        >>> duration_bias = torch.randn(K, C) * 0.1
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Streaming forward (uses Triton on GPU)
        >>> partition = semi_crf_streaming_forward(
        ...     cum_scores, transition, duration_bias, lengths, K
        ... )
        >>> partition.shape
        torch.Size([2])
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    # Input validation
    validate_cum_scores(cum_scores)
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    validate_lengths(lengths, T, batch_size=batch)

    can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

    # Determine if gradients are needed
    needs_grad = (
        cum_scores.requires_grad
        or transition.requires_grad
        or duration_bias.requires_grad
        or (proj_start is not None and proj_start.requires_grad)
        or (proj_end is not None and proj_end.requires_grad)
    )

    if needs_grad:
        # Training path
        if can_use_triton:
            # Use Triton forward + Triton backward kernels
            return SemiCRFStreamingTriton.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
                accum_dtype,
                num_warps,
            )
        else:
            # Pure PyTorch path
            return SemiCRFStreaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
            )
    else:
        # Inference path (no gradients)
        if can_use_triton:
            # Use fast custom Triton kernel
            partition, _, _ = launch_streaming_triton_kernel(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start=proj_start,
                proj_end=proj_end,
                num_warps=num_warps,
            )
            return partition
        else:
            # CPU fallback
            partition, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
            )
            return partition
