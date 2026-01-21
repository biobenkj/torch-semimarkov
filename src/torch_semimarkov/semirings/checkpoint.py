r"""Gradient checkpointing wrappers for memory-efficient semiring operations.

This module provides semiring wrappers that trade compute for memory by
recomputing forward passes during backward (gradient checkpointing). Essential
for running binary tree algorithms on large state spaces without OOM.

Two strategies are provided:

- :func:`CheckpointSemiring`: Basic checkpointing for matrix multiplications
- :func:`CheckpointShardSemiring`: Sharded checkpointing for very large matrices

Examples::

    >>> from torch_semimarkov.semirings import LogSemiring
    >>> from torch_semimarkov.semirings.checkpoint import CheckpointSemiring
    >>> # Wrap LogSemiring with checkpointing for memory-efficient gradients
    >>> CheckpointedLog = CheckpointSemiring(LogSemiring, min_size=1000)
    >>> model = SemiMarkov(CheckpointedLog)
"""

import torch


def broadcast_size(a, b):
    r"""Compute the broadcasted tensor size."""
    return torch.tensor([max(i, j) for i, j in zip(a.shape, b.shape, strict=False)]).prod()


def matmul_size(a, b):
    r"""Compute the output shape of matrix multiplication."""
    size = [max(i, j) for i, j in zip(a.shape[:-2], b.shape[:-2], strict=False)]
    size.append(a.shape[-2])
    size.append(b.shape[-1])
    return size


def CheckpointSemiring(cls, min_size=0):
    r"""CheckpointSemiring(cls, min_size=0) -> class

    Wrap a semiring with gradient checkpointing for memory-efficient backward.

    During backward, the forward computation is recomputed instead of storing
    intermediate activations. This trades 2x compute for reduced memory.

    Args:
        cls: Base semiring class to wrap (e.g., :class:`LogSemiring`).
        min_size (int, optional): Minimum tensor size to trigger checkpointing.
            Smaller tensors use standard backward. Default: ``0``

    Returns:
        class: A wrapped semiring class with checkpointed matmul.

    Examples::

        >>> CheckpointedLog = CheckpointSemiring(LogSemiring, min_size=1000)
        >>> model = SemiMarkov(CheckpointedLog)
        >>> # Checkpointed backward uses less memory (but ~2x slower)
        >>> log_Z, _, _ = model.logpartition(edge)
    """

    class _Check(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return cls.matmul(a, b)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            with torch.enable_grad():
                q = cls.matmul(a, b)
                return torch.autograd.grad(q, (a, b), grad_output)

    class _CheckpointSemiring(cls):
        @staticmethod
        def matmul(a, b):
            if broadcast_size(a, b) > min_size:
                return _Check.apply(a, b)
            else:
                return cls.matmul(a, b)

    return _CheckpointSemiring


def CheckpointShardSemiring(cls, max_size, min_size=0):
    r"""CheckpointShardSemiring(cls, max_size, min_size=0) -> class

    Wrap a semiring with sharded gradient checkpointing for very large matrices.

    For matrices too large to fit in memory even with basic checkpointing, this
    wrapper splits the computation into shards that are processed sequentially.
    Each shard is checkpointed independently.

    Args:
        cls: Base semiring class to wrap (e.g., :class:`LogSemiring`).
        max_size (int): Maximum number of elements per shard. Smaller values
            reduce peak memory but increase overhead.
        min_size (int, optional): Minimum tensor size to trigger sharding.
            Smaller tensors use standard checkpointing. Default: ``0``

    Returns:
        class: A wrapped semiring class with sharded checkpointed matmul.

    Examples::

        >>> # For very large KC state spaces
        >>> ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=100000)
        >>> model = SemiMarkov(ShardedLog)

    .. warning::
        Sharded checkpointing has significant overhead. Only use when necessary
        to avoid OOM on the binary tree algorithm with large state spaces.
    """

    class _Check(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            size = matmul_size(a, b)
            return accumulate_(
                a,
                b,
                size,
                lambda a, b: cls.matmul(a, b),
                preserve=len(size),
                step=max_size // (b.shape[-2] * a.shape[-1]) + 2,
            )

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            grad_a, grad_b = unaccumulate_(
                a,
                b,
                grad_output,
                len(grad_output.shape),
                lambda a, b: cls.matmul(a, b),
                step=max_size // (b.shape[-2] * a.shape[-1]) + 2,
            )
            return grad_a, grad_b

    class _CheckpointSemiring(cls):
        @staticmethod
        def matmul(a, b):
            size = torch.tensor([max(i, j) for i, j in zip(a.shape, b.shape, strict=False)]).prod()
            if size < min_size:
                return cls.matmul(a, b)
            else:
                return _Check.apply(a, b)

    return _CheckpointSemiring


def ones(x):
    one = []
    for i, v in enumerate(x.shape[:-1]):
        if v == 1:
            one.append(i)
    return one


def mind(one, inds):
    inds = list(inds)
    for v in one:
        inds[v] = inds[v].clone().fill_(0)
    return inds


def accumulate_(a, b, size, fn, preserve, step=10000):
    slices = []
    total = 1
    for s in size[:preserve]:
        slices.append(slice(s))
        total *= s
    if step > total:
        return fn(a, b)

    ret = torch.zeros(*size, dtype=a.dtype, device=a.device)

    a = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
    b = b.expand(*size[:-2], b.shape[-2], b.shape[-1])

    a2 = a.contiguous().view(-1, a.shape[-2], a.shape[-1])
    b2 = b.contiguous().view(-1, b.shape[-2], b.shape[-1])
    ret = ret.view(-1, a.shape[-2], b.shape[-1])
    for p in range(0, ret.shape[0], step):
        ret[p : p + step, :] = fn(a2[p : p + step], b2[p : p + step])
    ret = ret.view(*size)
    return ret


def unaccumulate_(a, b, grad_output, preserve, fn, step=10000):
    slices = []
    total = 1
    size = grad_output.shape[:preserve]
    for s in grad_output.shape[:preserve]:
        slices.append(slice(s))
        total *= s

    if step > total:
        with torch.enable_grad():
            a_in = a.clone().requires_grad_(True)
            b_in = b.clone().requires_grad_(True)
            q = fn(a, b)
        ag, bg = torch.autograd.grad(q, (a, b), grad_output)
        return ag, bg

    a2 = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
    b2 = b.expand(*size[:-2], b.shape[-2], b.shape[-1])
    a2 = a2.contiguous().view(-1, a.shape[-2], a.shape[-1])
    b2 = b2.contiguous().view(-1, b.shape[-2], b.shape[-1])

    a_grad = a2.clone().fill_(0)
    b_grad = b2.clone().fill_(0)

    grad_output = grad_output.view(-1, a.shape[-2], b.shape[-1])
    for p in range(0, grad_output.shape[0], step):
        with torch.enable_grad():
            a_in = a2[p : p + step].clone().requires_grad_(True)
            b_in = b2[p : p + step].clone().requires_grad_(True)
            q = fn(a_in, b_in)
        ag, bg = torch.autograd.grad(q, (a_in, b_in), grad_output[p : p + step])
        a_grad[p : p + step] += ag
        b_grad[p : p + step] += bg

    a_grad = a_grad.view(*size[:-2], a.shape[-2], a.shape[-1])
    b_grad = b_grad.view(*size[:-2], b.shape[-2], b.shape[-1])
    a_ones = ones(a)
    b_ones = ones(b)
    f1, f2 = a_grad.sum(a_ones, keepdim=True), b_grad.sum(b_ones, keepdim=True)
    return f1, f2
