import math

import torch

from .semirings import LogSemiring


class Chart:
    r"""Dynamic programming chart for structured prediction algorithms.

    Provides a tensor wrapper with automatic semiring initialization and
    gradient tracking for use in DP algorithms.

    Args:
        size (tuple): Shape of the chart (excluding semiring dimension).
        potentials (Tensor): Reference tensor for dtype and device.
        semiring: Semiring class defining the algebraic operations.

    Attributes:
        data (Tensor): The chart tensor of shape :math:`(\text{ssize},) + \text{size}`.
        grad (Tensor): Gradient accumulator tensor (same shape as data).
    """

    def __init__(self, size, potentials, semiring):
        c = torch.zeros(
            *((semiring.size(),) + size), dtype=potentials.dtype, device=potentials.device
        )
        c[:] = semiring.zero.view((semiring.size(),) + len(size) * (1,))

        self.data = c
        self.grad = self.data.detach().clone().fill_(0.0)

    def __getitem__(self, ind):
        slice_all = slice(None)
        return self.data[(slice_all, slice_all) + ind]

    def __setitem__(self, ind, new):
        slice_all = slice(None)
        self.data[(slice_all, slice_all) + ind] = new


class _Struct:
    r"""Base class for structured prediction models.

    Provides common infrastructure for dynamic programming algorithms over
    structured output spaces, including chart allocation, marginal computation
    via autograd, and semiring abstraction.

    Args:
        semiring: Semiring class defining the algebraic operations for inference.
            Default: :class:`~torch_semimarkov.semirings.LogSemiring`

    Attributes:
        semiring: The semiring used for inference operations.

    See Also:
        :class:`~torch_semimarkov.SemiMarkov`: Semi-Markov CRF implementation
    """

    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def score(self, potentials, parts, batch_dims=None):
        if batch_dims is None:
            batch_dims = [0]
        score = torch.mul(potentials, parts)
        batch = tuple(score.shape[b] for b in batch_dims)
        return self.semiring.prod(score.view(batch + (-1,)))

    def _bin_length(self, length):
        log_N = int(math.ceil(math.log(length, 2)))
        bin_N = int(math.pow(2, log_N))
        return log_N, bin_N

    def _get_dimension(self, edge):
        if isinstance(edge, list):
            for t in edge:
                t.requires_grad_(True)
            return edge[0].shape
        else:
            edge.requires_grad_(True)
            return edge.shape

    def _chart(self, size, potentials, force_grad):
        return self._make_chart(1, size, potentials, force_grad)[0]

    def _make_chart(self, N, size, potentials, force_grad=False):
        chart = []
        for _ in range(N):
            c = torch.zeros(
                *((self.semiring.size(),) + size), dtype=potentials.dtype, device=potentials.device
            )
            c[:] = self.semiring.zero.view((self.semiring.size(),) + len(size) * (1,))
            c.requires_grad_(force_grad and not potentials.requires_grad)
            chart.append(c)
        return chart

    def sum(self, logpotentials, lengths=None, _raw=False, **kwargs):
        r"""sum(logpotentials, lengths=None, _raw=False, **kwargs) -> Tensor

        Compute the semiring sum over all valid structures.

        For :class:`LogSemiring`, this returns the log partition function.
        For :class:`MaxSemiring`, this returns the Viterbi score.

        Args:
            logpotentials (Tensor): Model potentials (structure-specific shape).
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None``
            _raw (bool, optional): If ``True``, return unconverted semiring values.
                Default: ``False``
            **kwargs: Additional arguments passed to :meth:`logpartition`.

        Returns:
            Tensor: Semiring sum of shape :math:`(\text{batch},)`.
        """
        v = self.logpartition(logpotentials, lengths, **kwargs)[0]
        if _raw:
            return v
        return self.semiring.unconvert(v)

    def marginals(self, logpotentials, lengths=None, _raw=False, **kwargs):
        r"""marginals(logpotentials, lengths=None, _raw=False, **kwargs) -> Tensor

        Compute posterior marginals via automatic differentiation.

        The marginal of each potential is computed as the gradient of the log
        partition function with respect to that potential, which equals the
        posterior probability under the model.

        Args:
            logpotentials (Tensor): Model potentials (structure-specific shape).
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None``
            _raw (bool, optional): If ``True``, return raw semiring marginals.
                Default: ``False``
            **kwargs: Additional arguments passed to :meth:`logpartition`.

        Returns:
            Tensor: Marginal probabilities with same shape as ``logpotentials``.

        Examples::

            >>> model = SemiMarkov(LogSemiring)
            >>> edge = torch.randn(2, 99, 8, 4, 4)
            >>> marginals = model.marginals(edge)
            >>> marginals.shape
            torch.Size([2, 99, 8, 4, 4])
        """
        v, edges, _ = self.logpartition(logpotentials, lengths=lengths, force_grad=True, **kwargs)
        if _raw:
            all_m = []
            for k in range(v.shape[0]):
                obj = v[k].sum(dim=0)

                marg = torch.autograd.grad(
                    obj,
                    edges,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=False,
                )
                all_m.append(self.semiring.unconvert(self._arrange_marginals(marg)))
            return torch.stack(all_m, dim=0)
        else:
            obj = self.semiring.unconvert(v).sum(dim=0)
            marg = torch.autograd.grad(
                obj, edges, create_graph=True, only_inputs=True, allow_unused=False
            )
            a_m = self._arrange_marginals(marg)
            return self.semiring.unconvert(a_m)

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        return spans

    @staticmethod
    def from_parts(spans):
        return spans, None

    def _arrange_marginals(self, marg):
        return marg[0]
