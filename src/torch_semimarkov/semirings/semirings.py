r"""Semiring implementations for structured prediction.

This module provides semiring abstractions that enable different inference
algorithms using the same underlying dynamic programming structure:

- :class:`LogSemiring`: Log-space operations (logsumexp, +). Gradients give marginals.
- :class:`MaxSemiring`: Max-plus operations for Viterbi. Gradients give argmax.
- :class:`StdSemiring`: Standard counting semiring (+, *).
- :func:`KMaxSemiring`: K-best structures.
- :class:`EntropySemiring`: Entropy computation.
- :class:`CrossEntropySemiring`: Cross-entropy between distributions.
- :class:`KLDivergenceSemiring`: KL divergence computation.

The semiring abstraction allows the same DP algorithm to compute different
quantities by changing the semiring operations.

Examples::

    >>> from torch_semimarkov.semirings import LogSemiring, MaxSemiring
    >>> # LogSemiring computes partition function, gradients give marginals
    >>> model = SemiMarkov(LogSemiring)
    >>> # MaxSemiring computes Viterbi score, gradients give argmax
    >>> model = SemiMarkov(MaxSemiring)
"""

import torch


def matmul(cls, a, b):
    dims = 1
    act_on = -(dims + 1)
    a = a.unsqueeze(-1)
    b = b.unsqueeze(act_on - 1)
    c = cls.times(a, b)
    for _d in range(act_on, -1, 1):
        c = cls.sum(c.transpose(-2, -1))
    return c


class Semiring:
    r"""Base semiring class for structured prediction algorithms.

    A semiring :math:`(K, \oplus, \otimes, \bar{0}, \bar{1})` provides:

    - A set K of values
    - An addition operation :math:`\oplus` (commutative, associative)
    - A multiplication operation :math:`\otimes` (associative, distributes over :math:`\oplus`)
    - Zero element :math:`\bar{0}` (identity for :math:`\oplus`, annihilator for :math:`\otimes`)
    - One element :math:`\bar{1}` (identity for :math:`\otimes`)

    Subclasses must define:

    - ``zero``: The zero element
    - ``one``: The one element
    - ``sum(xs, dim)``: The :math:`\oplus` operation (reduction)
    - ``mul(a, b)``: The :math:`\otimes` operation (elementwise)

    Based on semiring parsing framework from Goodman (1999).

    Attributes:
        zero (Tensor): The semiring zero element.
        one (Tensor): The semiring one element.
    """

    @classmethod
    def matmul(cls, a, b):
        r"""matmul(a, b) -> Tensor

        Generalized matrix multiplication using semiring operations.

        Computes :math:`C_{ij} = \bigoplus_k A_{ik} \otimes B_{kj}`.
        """
        return matmul(cls, a, b)

    @classmethod
    def size(cls):
        r"""size() -> int

        Return the semiring size (extra first dimension for multi-valued semirings).
        """
        return 1

    @classmethod
    def dot(cls, a, b):
        r"""dot(a, b) -> Tensor

        Semiring dot product along last dimension.
        """
        a = a.unsqueeze(-2)
        b = b.unsqueeze(-1)
        return cls.matmul(a, b).squeeze(-1).squeeze(-1)

    @staticmethod
    def fill(c, mask, v):
        r"""fill(c, mask, v) -> Tensor

        Fill tensor ``c`` with value ``v`` where ``mask`` is True.
        """
        mask = mask.to(c.device)
        return torch.where(mask, v.type_as(c).view((-1,) + (1,) * (len(c.shape) - 1)), c)

    @classmethod
    def times(cls, *ls):
        r"""times(*ls) -> Tensor

        Multiply a sequence of tensors together using :math:`\otimes`.
        """
        cur = ls[0]
        for item in ls[1:]:
            cur = cls.mul(cur, item)
        return cur

    @classmethod
    def convert(cls, potentials):
        r"""convert(potentials) -> Tensor

        Convert potentials to semiring representation (adds ssize dimension).
        """
        return potentials.unsqueeze(0)

    @classmethod
    def unconvert(cls, potentials):
        r"""unconvert(potentials) -> Tensor

        Convert from semiring representation (removes ssize dimension).
        """
        return potentials.squeeze(0)

    @staticmethod
    def sum(xs, dim=-1):
        r"""sum(xs, dim=-1) -> Tensor

        Semiring sum (:math:`\oplus`) reduction over dimension.
        """
        raise NotImplementedError()

    @classmethod
    def plus(cls, a, b):
        r"""plus(a, b) -> Tensor

        Binary semiring addition: :math:`a \oplus b`.
        """
        return cls.sum(torch.stack([a, b], dim=-1))


class _Base(Semiring):
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)

    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def prod(a, dim=-1):
        return torch.prod(a, dim=dim)


class _BaseLog(Semiring):
    zero = torch.tensor(-1e5)
    one = torch.tensor(-0.0)

    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def prod(a, dim=-1):
        return torch.sum(a, dim=dim)


class StdSemiring(_Base):
    r"""Standard counting semiring :math:`(\mathbb{R}, +, \times, 0, 1)`.

    The standard semiring uses addition for :math:`\oplus` and multiplication
    for :math:`\otimes`. Useful for counting paths or computing expectations.

    Operations:

    - :math:`\oplus`: ``torch.sum``
    - :math:`\otimes`: ``torch.mul``
    - :math:`\bar{0}`: ``0.0``
    - :math:`\bar{1}`: ``1.0``
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)

    @classmethod
    def matmul(cls, a, b):
        return torch.matmul(a, b)


class LogSemiring(_BaseLog):
    r"""Log-space semiring :math:`(\mathbb{R} \cup \{-\infty\}, \text{logsumexp}, +, -\infty, 0)`.

    The log semiring operates in log-space for numerical stability. Used for
    computing partition functions. **Gradients give posterior marginals.**

    Operations:

    - :math:`\oplus`: ``torch.logsumexp``
    - :math:`\otimes`: ``+`` (addition in log-space = multiplication in probability space)
    - :math:`\bar{0}`: ``-inf``
    - :math:`\bar{1}`: ``0.0``

    Examples::

        >>> model = SemiMarkov(LogSemiring)
        >>> log_Z, _ = model.logpartition(edge)  # Partition function
        >>> marginals = model.marginals(edge)    # Posterior marginals via autograd
    """


class MaxSemiring(_BaseLog):
    r"""Max-plus semiring :math:`(\mathbb{R} \cup \{-\infty\}, \max, +, -\infty, 0)`.

    The max semiring finds the highest-scoring structure (Viterbi algorithm).
    **Gradients give the argmax structure.**

    Operations:

    - :math:`\oplus`: ``torch.max``
    - :math:`\otimes`: ``+``
    - :math:`\bar{0}`: ``-inf``
    - :math:`\bar{1}`: ``0.0``

    Examples::

        >>> model = SemiMarkov(MaxSemiring)
        >>> viterbi_score, _ = model.logpartition(edge)  # Best path score
        >>> best_path = model.marginals(edge)            # Argmax via autograd
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @staticmethod
    def sparse_sum(xs, dim=-1):
        m, a = torch.max(xs, dim=dim)
        return m, (torch.zeros(a.shape).long(), a)


def KMaxSemiring(k):
    r"""KMaxSemiring(k) -> class

    Create a K-max semiring for finding the k-best structures.

    The K-max semiring tracks the top-k scores at each step, enabling
    extraction of the k-best segmentations.

    Args:
        k (int): Number of best structures to track.

    Returns:
        class: A semiring class configured for k-best computation.

    Examples::

        >>> KMax3 = KMaxSemiring(3)
        >>> model = SemiMarkov(KMax3)
        >>> top3_scores, _ = model.logpartition(edge)
    """

    class KMaxSemiring(_BaseLog):

        zero = torch.tensor([-1e5 for i in range(k)])
        one = torch.tensor([0 if i == 0 else -1e5 for i in range(k)])

        @staticmethod
        def size():
            return k

        @classmethod
        def convert(cls, orig_potentials):
            potentials = torch.zeros(
                (k,) + orig_potentials.shape,
                dtype=orig_potentials.dtype,
                device=orig_potentials.device,
            )
            potentials = cls.fill(potentials, torch.tensor(True), cls.zero)
            potentials[0] = orig_potentials
            return potentials

        @staticmethod
        def unconvert(potentials):
            return potentials[0]

        @staticmethod
        def sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs = torch.topk(xs, k, dim=-1)[0]
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs
            raise AssertionError("KMaxSemiring.sum only supports dim=-1")

        @staticmethod
        def sparse_sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs, xs2 = torch.topk(xs, k, dim=-1)
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                xs2 = xs2.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs, (xs2 % k, xs2 // k)
            raise AssertionError("KMaxSemiring.sparse_sum only supports dim=-1")

        @staticmethod
        def mul(a, b):
            a = a.view((k, 1) + a.shape[1:])
            b = b.view((1, k) + b.shape[1:])
            c = a + b
            c = c.contiguous().view((k * k,) + c.shape[2:])
            ret = torch.topk(c, k, 0)[0]
            assert ret.shape[0] == k
            return ret

    return KMaxSemiring


class KLDivergenceSemiring(Semiring):
    r"""KL-divergence expectation semiring.

    Computes the KL divergence :math:`D_{KL}(P \| Q)` between two distributions
    P and Q alongside their log partition functions.

    The semiring tracks three values:

    1. Log-partition of distribution P
    2. Log-partition of distribution Q
    3. Running KL divergence

    Based on expectation semiring framework from Eisner (2002) and Li & Eisner (2009).
    """

    zero = torch.tensor([-1e5, -1e5, 0.0])
    one = torch.tensor([0.0, 0.0, 0.0])

    @staticmethod
    def size():
        return 3

    @staticmethod
    def convert(xs):
        values = torch.zeros((3,) + xs[0].shape).type_as(xs[0])
        values[0] = xs[0]
        values[1] = xs[1]
        values[2] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[-1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part_p = torch.logsumexp(xs[0], dim=d)
        part_q = torch.logsumexp(xs[1], dim=d)
        log_sm_p = xs[0] - part_p.unsqueeze(d)
        log_sm_q = xs[1] - part_q.unsqueeze(d)
        sm_p = log_sm_p.exp()
        return torch.stack(
            (
                part_p,
                part_q,
                torch.sum(xs[2].mul(sm_p) - log_sm_q.mul(sm_p) + log_sm_p.mul(sm_p), dim=d),
            )
        )

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


class CrossEntropySemiring(Semiring):
    r"""Cross-entropy expectation semiring.

    Computes the cross-entropy :math:`H(P, Q) = -\sum_x P(x) \log Q(x)` between
    two distributions P and Q alongside their log partition functions.

    The semiring tracks three values:

    1. Log-partition of distribution P
    2. Log-partition of distribution Q
    3. Running cross-entropy

    Based on expectation semiring framework from Eisner (2002) and Li & Eisner (2009).
    """

    zero = torch.tensor([-1e5, -1e5, 0.0])
    one = torch.tensor([0.0, 0.0, 0.0])

    @staticmethod
    def size():
        return 3

    @staticmethod
    def convert(xs):
        values = torch.zeros((3,) + xs[0].shape).type_as(xs[0])
        values[0] = xs[0]
        values[1] = xs[1]
        values[2] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[-1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part_p = torch.logsumexp(xs[0], dim=d)
        part_q = torch.logsumexp(xs[1], dim=d)
        log_sm_p = xs[0] - part_p.unsqueeze(d)
        log_sm_q = xs[1] - part_q.unsqueeze(d)
        sm_p = log_sm_p.exp()
        return torch.stack((part_p, part_q, torch.sum(xs[2].mul(sm_p) - log_sm_q.mul(sm_p), dim=d)))

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


class EntropySemiring(Semiring):
    r"""Entropy expectation semiring.

    Computes the Shannon entropy :math:`H(P) = -\sum_x P(x) \log P(x)` of the
    distribution alongside its log partition function.

    The semiring tracks two values:

    1. Log-partition of the distribution
    2. Running entropy

    Useful for active learning, uncertainty quantification, and regularization.
    Based on expectation semiring framework from Eisner (2002) and Li & Eisner (2009).
    """

    zero = torch.tensor([-1e5, 0.0])
    one = torch.tensor([0.0, 0.0])

    @staticmethod
    def size():
        return 2

    @staticmethod
    def convert(xs):
        values = torch.zeros((2,) + xs.shape).type_as(xs)
        values[0] = xs
        values[1] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part = torch.logsumexp(xs[0], dim=d)
        log_sm = xs[0] - part.unsqueeze(d)
        sm = log_sm.exp()
        return torch.stack((part, torch.sum(xs[1].mul(sm) - log_sm.mul(sm), dim=d)))

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


def TempMax(alpha):
    class _TempMax(_BaseLog):
        """
        Implements a max forward, hot softmax backward.
        """

        @staticmethod
        def sum(xs, dim=-1):
            return torch.max(xs, dim=dim)[0]

        @staticmethod
        def sparse_sum(xs, dim=-1):
            m, _ = torch.max(xs, dim=dim)
            a = torch.softmax(alpha * xs, dim)
            return m, (torch.zeros(a.shape[:-1]).long(), a)

    return _TempMax
