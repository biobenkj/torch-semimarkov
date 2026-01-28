r"""Autograd functions and public API for streaming Semi-CRF."""

from typing import Optional

import torch

from ..validation import validate_cum_scores, validate_lengths
from .pytorch_reference import (
    linear_crf_backward_pytorch,
    linear_crf_forward_pytorch,
    linear_crf_viterbi_pytorch,
    semi_crf_k2_backward_pytorch,
    semi_crf_k2_forward_pytorch,
    semi_crf_k2_viterbi_pytorch,
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)

# Triton imports are conditional
try:
    from .triton_backward import launch_streaming_triton_backward
    from .triton_forward import HAS_TRITON, launch_streaming_triton_kernel
except ImportError:
    HAS_TRITON = False
    launch_streaming_triton_kernel = None
    launch_streaming_triton_backward = None


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
    r"""Triton-accelerated autograd function for streaming Semi-CRF. O(KC) memory."""

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
        num_warps: int = 4,
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
        ctx.num_warps = num_warps

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
        # Note: launch_streaming_triton_backward returns 6 values; the 6th (boundary_marginals) is unused here
        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end, _ = (
            launch_streaming_triton_backward(
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
                num_warps=ctx.num_warps,
            )
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
            None,  # num_warps
        )


# =============================================================================
# K=1 Linear CRF Fast Path
# =============================================================================


class LinearCRFStreaming(torch.autograd.Function):
    r"""Optimized K=1 (linear CRF) autograd function. O(batch×C) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        if semiring == "max":
            # Viterbi: return scores only, no gradient support for max semiring
            scores, _ = linear_crf_viterbi_pytorch(
                cum_scores.detach(),
                transition.detach(),
                lengths,
                duration_bias.detach(),
            )
            return scores

        # Forward pass
        partition = linear_crf_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            lengths,
            duration_bias.detach(),
        )

        # Save for backward
        ctx.save_for_backward(cum_scores, transition, duration_bias, lengths, partition)
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cum_scores, transition, duration_bias, lengths, partition = ctx.saved_tensors

        # Validate partition before backward
        if not torch.isfinite(partition).all():
            nan_count = torch.isnan(partition).sum().item()
            inf_count = torch.isinf(partition).sum().item()
            raise RuntimeError(
                f"Non-finite partition from forward pass (K=1): "
                f"{nan_count} NaN, {inf_count} Inf."
            )

        grad_cum_scores, grad_transition, grad_duration_bias = linear_crf_backward_pytorch(
            cum_scores,
            transition,
            lengths,
            partition,
            duration_bias,
        )

        # Validate backward outputs
        if not torch.isfinite(grad_cum_scores).all():
            nan_count = torch.isnan(grad_cum_scores).sum().item()
            inf_count = torch.isinf(grad_cum_scores).sum().item()
            raise RuntimeError(
                f"Non-finite values in K=1 backward: "
                f"grad_cum_scores has {nan_count} NaN, {inf_count} Inf"
            )

        # Scale by upstream gradient
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum
        grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)

        if grad_duration_bias is not None:
            # grad_duration_bias is (batch, 1, C) -> reduce to (1, C)
            grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # semiring
        )


# =============================================================================
# K=2 Specialized Path
# =============================================================================


class SemiCRFK2Streaming(torch.autograd.Function):
    r"""Optimized K=2 semi-CRF autograd function. O(batch×C) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        if semiring == "max":
            # Viterbi: return scores only
            scores, _, _ = semi_crf_k2_viterbi_pytorch(
                cum_scores.detach(),
                transition.detach(),
                duration_bias.detach(),
                lengths,
            )
            return scores

        # Forward pass
        partition = semi_crf_k2_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
        )

        # Save for backward
        ctx.save_for_backward(cum_scores, transition, duration_bias, lengths, partition)
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cum_scores, transition, duration_bias, lengths, partition = ctx.saved_tensors

        # Validate partition before backward
        if not torch.isfinite(partition).all():
            nan_count = torch.isnan(partition).sum().item()
            inf_count = torch.isinf(partition).sum().item()
            raise RuntimeError(
                f"Non-finite partition from forward pass (K=2): "
                f"{nan_count} NaN, {inf_count} Inf."
            )

        grad_cum_scores, grad_transition, grad_duration_bias = semi_crf_k2_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
        )

        # Validate backward outputs
        if not torch.isfinite(grad_cum_scores).all():
            nan_count = torch.isnan(grad_cum_scores).sum().item()
            inf_count = torch.isinf(grad_cum_scores).sum().item()
            raise RuntimeError(
                f"Non-finite values in K=2 backward: "
                f"grad_cum_scores has {nan_count} NaN, {inf_count} Inf"
            )

        # Scale by upstream gradient
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum
        grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # semiring
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
    num_warps: int = 4,
) -> torch.Tensor:
    r"""Compute Semi-CRF partition function with streaming edge computation.

    O(KC) memory (ring buffer), O(T×K×C²) compute. Automatically dispatches to
    optimized implementations based on K:

    - **K=1**: Linear CRF fast path. O(batch×C) memory, no ring buffer.
    - **K=2**: Specialized 2-step path. O(batch×C) memory, explicit history.
    - **K≥3**: Triton streaming kernel on GPU, PyTorch fallback on CPU.

    .. warning::
        ``cum_scores`` **MUST** be float32 for numerical stability at T > 100K.
        Zero-centering before cumsum is critical to prevent precision loss.

    .. note::
        The Triton kernel requires K≥3 for correct operation due to ring buffer
        architecture constraints. K=1 and K=2 use specialized PyTorch paths.

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

    # Determine if gradients are needed
    needs_grad = (
        cum_scores.requires_grad
        or transition.requires_grad
        or duration_bias.requires_grad
        or (proj_start is not None and proj_start.requires_grad)
        or (proj_end is not None and proj_end.requires_grad)
    )

    # =========================================================================
    # K=1 Fast Path: Linear CRF (no ring buffer, no duration loop)
    # =========================================================================
    if K == 1:
        if proj_start is not None or proj_end is not None:
            # K=1 fast path doesn't support boundary projections yet
            # Fall through to generic implementation
            pass
        elif needs_grad:
            return LinearCRFStreaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                semiring,
            )
        else:
            # Inference path
            if semiring == "max":
                scores, _ = linear_crf_viterbi_pytorch(
                    cum_scores, transition, lengths, duration_bias
                )
                return scores
            else:
                return linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

    # =========================================================================
    # K=2 Fast Path: Explicit 2-step history (no ring buffer)
    # =========================================================================
    if K == 2:
        if proj_start is not None or proj_end is not None:
            # K=2 fast path doesn't support boundary projections yet
            # Fall through to generic implementation
            pass
        elif needs_grad:
            return SemiCRFK2Streaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                semiring,
            )
        else:
            # Inference path
            if semiring == "max":
                scores, _, _ = semi_crf_k2_viterbi_pytorch(
                    cum_scores, transition, duration_bias, lengths
                )
                return scores
            else:
                return semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths)

    # =========================================================================
    # K>=3: Triton streaming kernel (ring buffer architecture)
    # =========================================================================
    can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

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
