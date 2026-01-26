"""Generalized batch matrix multiplication with semiring-parameterized operations.

Provides custom autograd Functions for CUDA-accelerated matrix operations:

- ``LogMatMul``: Log-space matrix multiplication (logsumexp reduction)
- ``MaxMatMul``: Max-space matrix multiplication (max reduction)
- ``SampleMatMul``: Sampling-based matrix multiplication (experimental)
- ``ProdMaxMatMul``: Product-max matrix multiplication (experimental)

These are low-level building blocks for semiring DP algorithms.
Requires the optional ``_C`` CUDA extension; falls back gracefully if unavailable.

The CUDA kernel supports four semiring modes via integer flags:

- Mode 0: Log semiring (logsumexp)
- Mode 1: Max semiring (max with argmax tracking)
- Mode 2: Sample semiring (Gumbel-max sampling)
- Mode 3: Product-max semiring (product in forward, max in backward)

Example:
    >>> import torch
    >>> from torch_semimarkov._genbmm import logbmm
    >>> a = torch.randn(2, 3, 4, device="cuda")
    >>> b = torch.randn(2, 4, 5, device="cuda")
    >>> c = logbmm(a, b)  # (2, 3, 5) via logsumexp reduction
"""

import torch

try:
    from . import _C as _genbmm
except ImportError:
    pass


def trans(s: torch.Tensor) -> torch.Tensor:
    """Transpose last two dimensions and make contiguous.

    Args:
        s: Input tensor of shape ``(..., M, N)``.

    Returns:
        Transposed tensor of shape ``(..., N, M)``, contiguous in memory.
    """
    return s.transpose(-2, -1).contiguous()


class LogMatMulBack(torch.autograd.Function):
    """Backward pass for log-space matrix multiplication.

    Computes gradients for LogMatMul using the CUDA kernel's backward function.
    Supports second-order gradients via backbackward.
    """

    @staticmethod
    def forward(ctx, a, b, grad_out, part, maxes):
        ctx.save_for_backward(a, b, grad_out, part, maxes)
        grad_a, _ = _genbmm.backward(a, b, grad_out, part, maxes, 0)
        return grad_a

    @staticmethod
    def backward(ctx, grad_output):
        a, b, grad_out, part, maxes = ctx.saved_tensors
        grad_a, grad_b, grad_grad = _genbmm.backbackward(
            a, b, grad_out.contiguous(), part, maxes, grad_output.contiguous(), 0
        )

        return grad_a, grad_b, grad_grad, None, None


class LogMatMul(torch.autograd.Function):
    """Log-space batch matrix multiplication (logsumexp reduction).

    Computes ``C[b, i, j] = logsumexp_k(A[b, i, k] + B[b, k, j])`` using
    numerically stable CUDA kernels. This is the log-semiring analog of
    standard matrix multiplication.

    Used for partition function computation in structured prediction.
    """

    @staticmethod
    def forward(ctx, a, b):
        out, maxes = _genbmm.forward(a, b, 0)
        ctx.save_for_backward(a, b, out, maxes)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, out, maxes = ctx.saved_tensors
        grad_a = LogMatMulBack.apply(a, b, grad_output.contiguous(), out, maxes)
        grad_b = LogMatMulBack.apply(
            trans(b), trans(a), trans(grad_output), trans(out), trans(maxes)
        )

        return grad_a, trans(grad_b)


class MaxMatMul(torch.autograd.Function):
    """Max-space batch matrix multiplication (max reduction).

    Computes ``C[b, i, j] = max_k(A[b, i, k] + B[b, k, j])`` using CUDA kernels
    that track argmax indices (switches) for gradient routing.

    Used for Viterbi decoding in structured prediction.
    """

    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 1)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a.float(),
            b.float(),
            grad_output.contiguous().float(),
            switches.float(),
            switches.float(),
            1,
        )
        return grad_a.to(a.dtype), grad_b.to(b.dtype)


class SampleMatMul(torch.autograd.Function):
    """Sampling-based batch matrix multiplication (Gumbel-max sampling).

    Experimental semiring for stochastic structured prediction.
    Not intended for production use.
    """

    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 2)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a.float(),
            b.float(),
            grad_output.contiguous().float(),
            switches.float(),
            switches.float(),
            2,
        )
        return grad_a.to(a.dtype), grad_b.to(b.dtype)


class ProdMaxMatMul(torch.autograd.Function):
    """Product-max batch matrix multiplication.

    Experimental semiring using product in forward, max in backward.
    Not intended for production use.
    """

    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 3)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a.float(),
            b.float(),
            grad_output.contiguous().float(),
            switches.float(),
            switches.float(),
            3,
        )
        return grad_a.to(a.dtype), grad_b.to(b.dtype)


#: Log-space batch matrix multiplication (logsumexp reduction).
logbmm = LogMatMul.apply

#: Max-space batch matrix multiplication (max reduction).
maxbmm = MaxMatMul.apply

#: Sampling-based batch matrix multiplication. Experimental, not for production use.
samplebmm = SampleMatMul.apply

#: Product-max batch matrix multiplication. Experimental, not for production use.
prodmaxbmm = ProdMaxMatMul.apply
