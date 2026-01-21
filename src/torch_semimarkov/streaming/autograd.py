r"""Autograd functions and public API for streaming Semi-CRF.

This module contains the :class:`torch.autograd.Function` classes and the main
entry point for the streaming Semi-CRF API.

Classes:
    SemiCRFStreaming: Pure PyTorch autograd function for streaming Semi-CRF.
    SemiCRFStreamingTriton: Triton-accelerated autograd function for streaming Semi-CRF.

Functions:
    semi_crf_streaming_forward: Main entry point for streaming Semi-CRF computation.
"""

from typing import Optional

import torch

from .pytorch_reference import (
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
    r"""Autograd function for streaming Semi-CRF with on-the-fly edge computation.

    This wraps the forward and backward passes to enable automatic differentiation.
    Memory usage is :math:`O(KC)` for the ring buffer, independent of sequence length T.

    The forward pass computes:

    .. math::
        \alpha[t, c] = \bigoplus_{k=1}^{K-1} \bigoplus_{c'} \alpha[t-k, c'] \otimes \text{edge}[t-k, k, c, c']

    where :math:`\oplus` is logsumexp (log semiring) or max (max semiring).

    .. note::
        This class is used internally by :func:`semi_crf_streaming_forward`.
        Users should call that function directly rather than using this class.

    See Also:
        :func:`semi_crf_streaming_forward`: Main entry point for streaming Semi-CRF
        :class:`SemiCRFStreamingTriton`: Triton-accelerated version
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
    r"""Autograd function using Triton forward and backward kernels.

    This class uses custom Triton kernels for both forward and backward passes,
    providing maximum performance for GPU training. The backward pass uses
    checkpointing to recompute alpha values, trading compute for memory.

    Memory complexity:

    - Forward: :math:`O(KC)` ring buffer + :math:`O(\frac{T}{S} \times KC)` checkpoints
    - Backward: :math:`O((S+K) \times C)` alpha buffer + :math:`O(KC)` beta ring

    where :math:`S = \sqrt{T \times K}` is the checkpoint interval.

    .. note::
        This class is used internally when ``use_triton=True`` and gradients
        are needed. Users should call :func:`semi_crf_streaming_forward` directly.

    See Also:
        :class:`SemiCRFStreaming`: Pure PyTorch autograd function
        :func:`semi_crf_streaming_forward`: Main entry point
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

        # Use Triton backward kernel for gradient computation
        # The kernel already scales by grad_output internally
        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end = (
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
            )
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
) -> torch.Tensor:
    r"""semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K, semiring="log", proj_start=None, proj_end=None, use_triton=True) -> Tensor

    Compute Semi-CRF partition function with streaming edge computation.

    This is the main entry point for the streaming API. Edge potentials are
    computed on-the-fly from cumulative scores, eliminating the need for the
    full :math:`(\text{batch}, T-1, K, C, C)` edge tensor.

    Memory: :math:`O(KC)` ring buffer, independent of sequence length T.
    Compute: :math:`O(T \times K \times C^2)` same as standard Semi-CRF.

    Uses custom Triton kernels for optimal performance on GPU:

    - **Inference** (no gradients): Uses custom Triton forward kernel
    - **Training** (with gradients): Uses custom Triton forward and backward kernels

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

    See Also:
        :class:`~torch_semimarkov.SemiMarkov`: Pre-computed edge tensor API
        :func:`compute_edge_block_streaming`: On-the-fly edge computation helper
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    # Check if Triton is available and applicable
    # Note: As of Phase 4B, Triton kernels now support boundary projections
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
            # Use Triton forward + Triton backward kernels (now supports boundaries)
            return SemiCRFStreamingTriton.apply(
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
            # Use fast custom Triton kernel (now supports boundaries)
            partition, _, _ = launch_streaming_triton_kernel(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start=proj_start,
                proj_end=proj_end,
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
