r"""Duration distribution classes for Semi-Markov CRF.

This module provides flexible duration parameterization through a factory pattern.
Different distributions can be used to model segment durations, allowing for
domain-specific priors or learned parameters.

Classes:
    :class:`DurationDistribution`: Abstract base class for duration distributions.
    :class:`LearnedDuration`: Fully learned duration bias (default behavior).
    :class:`GeometricDuration`: Geometric distribution :math:`P(k) \propto p(1-p)^{k-1}`.
    :class:`NegativeBinomialDuration`: Negative binomial distribution.
    :class:`PoissonDuration`: Poisson-like distribution.
    :class:`UniformDuration`: Uniform (no duration preference).
    :class:`CallableDuration`: User-provided callable for custom distributions.

Functions:
    :func:`create_duration_distribution`: Factory function to create distributions.

Numerical Stability:
    Most distributions include safeguards against numerical instability:

    - **GeometricDuration**: Numerically stable for all parameter values.
    - **PoissonDuration**: Stable; epsilon added to prevent ``log(0)``.
    - **NegativeBinomialDuration**: Can produce non-finite values with very small
      shape parameter :math:`r`. A runtime warning is emitted when this occurs.
      See :class:`NegativeBinomialDuration` for mitigation strategies.

Examples::

    >>> from torch_semimarkov.duration import LearnedDuration, GeometricDuration
    >>>
    >>> # Default learned duration bias (current behavior)
    >>> dur = LearnedDuration(max_duration=8, num_classes=4)
    >>> bias = dur()  # Returns (K, C) tensor
    >>>
    >>> # Geometric distribution with learned rate
    >>> dur = GeometricDuration(max_duration=8, num_classes=4)
    >>> bias = dur()  # Returns (K, C) tensor with geometric shape

See Also:
    :class:`~torch_semimarkov.nn.SemiMarkovCRFHead`: Uses duration distributions
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class DurationDistribution(nn.Module, ABC):
    r"""Base class for duration distributions.

    Duration distributions produce a bias tensor of shape :math:`(K, C)` where:

    - :math:`K` is the maximum segment duration
    - :math:`C` is the number of classes/labels

    The bias is added to segment scores in the Semi-Markov CRF, effectively
    implementing a prior over segment durations for each class.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.

    Attributes:
        max_duration (int): Maximum segment duration.
        num_classes (int): Number of label classes.
    """

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__()
        self.max_duration = max_duration
        self.num_classes = num_classes

    @abstractmethod
    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Compute duration bias tensor.

        Returns:
            Tensor: Duration biases of shape :math:`(K, C)` in log-space.
        """
        raise NotImplementedError


class LearnedDuration(DurationDistribution):
    r"""Fully learned duration bias (default behavior).

    Each (duration, class) combination has an independent learned parameter.
    This is the most flexible but requires the most data to learn.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        init_std (float, optional): Standard deviation for initialization.
            Default: ``0.1``

    Attributes:
        duration_bias (Parameter): Learned bias of shape :math:`(K, C)`.

    Examples::

        >>> dur = LearnedDuration(max_duration=8, num_classes=4)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])
    """

    def __init__(self, max_duration: int, num_classes: int, init_std: float = 0.1):
        super().__init__(max_duration, num_classes)
        self.duration_bias = nn.Parameter(torch.randn(max_duration, num_classes) * init_std)

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Return the learned duration bias.

        Returns:
            Tensor: Duration biases of shape :math:`(K, C)`.
        """
        return self.duration_bias


class GeometricDuration(DurationDistribution):
    r"""Geometric duration distribution.

    Models durations with exponential decay, where short segments are more
    likely than long ones. The rate parameter :math:`p` can be learned per-class.

    .. math::
        P(k) \propto p(1-p)^{k-1}

    In log space:

    .. math::
        \log P(k) = \log(p) + (k-1) \cdot \log(1-p)

    .. note::
        This distribution is numerically stable for all parameter values. The
        probability :math:`p` is constrained to :math:`(0, 1)` via sigmoid, and
        a small epsilon (``1e-8``) is added before taking logs to prevent
        ``log(0)`` at extreme values.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        init_logit (float, optional): Initial logit for success probability.
            Default: ``0.0`` (corresponds to :math:`p = 0.5`)
        learn_rate (bool, optional): If ``True``, the rate parameter is learned.
            Default: ``True``

    Attributes:
        logit_p (Parameter or Tensor): Logit of success probability, shape :math:`(C,)`.

    Examples::

        >>> dur = GeometricDuration(max_duration=8, num_classes=4)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])
    """

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_logit: float = 0.0,
        learn_rate: bool = True,
    ):
        super().__init__(max_duration, num_classes)

        # Logit of success probability p, one per class
        if learn_rate:
            self.logit_p = nn.Parameter(torch.full((num_classes,), init_logit))
        else:
            self.register_buffer("logit_p", torch.full((num_classes,), init_logit))

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Compute geometric duration bias.

        Returns:
            Tensor: Log-probabilities of shape :math:`(K, C)`.
        """
        # p = sigmoid(logit_p), constrained to (0, 1)
        p = torch.sigmoid(self.logit_p)  # (C,)

        # k values from 1 to K
        k = torch.arange(1, self.max_duration + 1, device=p.device, dtype=p.dtype)

        # log P(k) = log(p) + (k-1) * log(1-p)
        # Shape: (K,) + (K, 1) * (C,) -> (K, C) via broadcasting
        log_p = torch.log(p + 1e-8)  # (C,)
        log_1_minus_p = torch.log(1 - p + 1e-8)  # (C,)

        # (K, C) = (C,) + (K, 1) * (C,)
        log_prob = log_p.unsqueeze(0) + (k - 1).unsqueeze(1) * log_1_minus_p.unsqueeze(0)

        return log_prob


class NegativeBinomialDuration(DurationDistribution):
    r"""Negative binomial duration distribution.

    Generalizes geometric distribution with an additional shape parameter :math:`r`.

    .. math::
        P(k) \propto \binom{k+r-2}{k-1} p^r (1-p)^{k-1}

    When :math:`r=1`, reduces to geometric distribution.
    Larger :math:`r` values create more peaked distributions around the mode.

    .. warning::
        Very small values of :math:`r` (e.g., ``init_log_r < -10``) can cause
        numerical instability due to :func:`torch.lgamma` overflow in the binomial
        coefficient computation. When this occurs, a :class:`UserWarning` is emitted
        at runtime.

        If you encounter non-finite values, consider:

        - Using a larger ``init_log_r`` (e.g., ``-5.0`` or higher)
        - Switching to :class:`GeometricDuration` which is numerically stable
        - Clamping the learned ``log_r`` parameter during training

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        init_logit (float, optional): Initial logit for success probability.
            Default: ``0.0``
        init_log_r (float, optional): Initial log of shape parameter. Values below
            ``-10`` may cause numerical instability. Default: ``0.0`` (corresponds
            to :math:`r = 1`)
        learn_rate (bool, optional): If ``True``, the rate parameter is learned.
            Default: ``True``
        learn_shape (bool, optional): If ``True``, the shape parameter is learned.
            Default: ``True``

    Attributes:
        logit_p (Parameter or Tensor): Logit of success probability, shape :math:`(C,)`.
        log_r (Parameter or Tensor): Log of shape parameter, shape :math:`(C,)`.

    Examples::

        >>> dur = NegativeBinomialDuration(max_duration=8, num_classes=4)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])

    See Also:
        :class:`GeometricDuration`: Equivalent to negative binomial with :math:`r=1`,
            but numerically stable for all parameter values.
    """

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_logit: float = 0.0,
        init_log_r: float = 0.0,
        learn_rate: bool = True,
        learn_shape: bool = True,
    ):
        super().__init__(max_duration, num_classes)

        # Logit of success probability p
        if learn_rate:
            self.logit_p = nn.Parameter(torch.full((num_classes,), init_logit))
        else:
            self.register_buffer("logit_p", torch.full((num_classes,), init_logit))

        # Log of shape parameter r (positive)
        if learn_shape:
            self.log_r = nn.Parameter(torch.full((num_classes,), init_log_r))
        else:
            self.register_buffer("log_r", torch.full((num_classes,), init_log_r))

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Compute negative binomial duration bias.

        Returns:
            Tensor: Log-probabilities of shape :math:`(K, C)`.

        Warns:
            UserWarning: If the output contains non-finite values (NaN or Inf),
                typically caused by very small :math:`r` values. This check is
                skipped during TorchScript compilation and ``torch.compile``.
        """
        p = torch.sigmoid(self.logit_p)  # (C,)
        r = torch.exp(self.log_r) + 1e-8  # (C,), ensure positive

        k = torch.arange(1, self.max_duration + 1, device=p.device, dtype=p.dtype)

        # Use log-gamma for numerical stability
        # log C(k+r-2, k-1) = lgamma(k+r-1) - lgamma(r) - lgamma(k)
        # log P(k) = log_binom + r*log(p) + (k-1)*log(1-p)

        k_expanded = k.unsqueeze(1)  # (K, 1)
        r_expanded = r.unsqueeze(0)  # (1, C)

        log_binom = (
            torch.lgamma(k_expanded + r_expanded - 1)
            - torch.lgamma(r_expanded)
            - torch.lgamma(k_expanded)
        )

        log_p = torch.log(p + 1e-8)
        log_1_minus_p = torch.log(1 - p + 1e-8)

        log_prob = (
            log_binom
            + r_expanded * log_p.unsqueeze(0)
            + (k_expanded - 1) * log_1_minus_p.unsqueeze(0)
        )

        # Warn if numerical instability detected (common with very small r)
        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            non_finite_count = (~torch.isfinite(log_prob)).sum().item()
            if non_finite_count > 0:
                r_min = r.min().item()
                warnings.warn(
                    f"NegativeBinomialDuration produced {non_finite_count} non-finite values. "
                    f"This typically occurs when r is very small (current min r={r_min:.2e}). "
                    f"Consider using a larger init_log_r or switching to GeometricDuration.",
                    stacklevel=2,
                )

        return log_prob


class PoissonDuration(DurationDistribution):
    r"""Poisson-like duration distribution.

    Models durations centered around the mean :math:`\lambda`. Useful when segments
    have a characteristic length.

    .. math::
        P(k) \propto \frac{\lambda^k}{k!}

    .. note::
        This is a shifted Poisson (k starts at 1, not 0). A small epsilon (``1e-8``)
        is added to :math:`\lambda` before taking the log to prevent ``log(0)``
        when :math:`\lambda \to 0`.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        init_log_lambda (float, optional): Initial log of rate parameter.
            Default: ``1.0`` (corresponds to :math:`\lambda \approx 2.7`)
        learn_rate (bool, optional): If ``True``, the rate parameter is learned.
            Default: ``True``

    Attributes:
        log_lambda (Parameter or Tensor): Log of rate parameter, shape :math:`(C,)`.

    Examples::

        >>> dur = PoissonDuration(max_duration=8, num_classes=4)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])
    """

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_log_lambda: float = 1.0,
        learn_rate: bool = True,
    ):
        super().__init__(max_duration, num_classes)

        if learn_rate:
            self.log_lambda = nn.Parameter(torch.full((num_classes,), init_log_lambda))
        else:
            self.register_buffer("log_lambda", torch.full((num_classes,), init_log_lambda))

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Compute Poisson duration bias.

        Returns:
            Tensor: Log-probabilities of shape :math:`(K, C)`.
        """
        lam = torch.exp(self.log_lambda)  # (C,)

        k = torch.arange(1, self.max_duration + 1, device=lam.device, dtype=lam.dtype)

        # log P(k) = k * log(λ) - lgamma(k+1) - λ
        # (for shifted Poisson, we use k instead of k-1)
        k_expanded = k.unsqueeze(1)  # (K, 1)
        lam_expanded = lam.unsqueeze(0)  # (1, C)

        log_prob = (
            k_expanded * torch.log(lam_expanded + 1e-8)
            - torch.lgamma(k_expanded + 1)
            - lam_expanded
        )

        return log_prob


class CallableDuration(DurationDistribution):
    r"""User-provided callable for custom duration distributions.

    Allows full flexibility by accepting any function that returns
    a :math:`(K, C)` tensor of log-probabilities.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        func (Callable): Function with signature ``func(K, C, device) -> Tensor``
            that returns log-probabilities of shape :math:`(K, C)`.

    Examples::

        >>> def my_duration(K, C, device):
        ...     # Custom duration logic
        ...     return torch.zeros(K, C, device=device)
        >>> dur = CallableDuration(8, 4, my_duration)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])
    """

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        func: Callable[[int, int, torch.device], Tensor],
    ):
        super().__init__(max_duration, num_classes)
        self._func = func
        # Create a dummy parameter to track device
        self._device_tracker = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Call the user-provided function to compute duration bias.

        Returns:
            Tensor: Duration biases of shape :math:`(K, C)`.
        """
        device = self._device_tracker.device
        return self._func(self.max_duration, self.num_classes, device)


class UniformDuration(DurationDistribution):
    r"""Uniform duration distribution (no duration preference).

    All durations have equal probability: :math:`\log P(k) = 0` for all k.
    Useful as a baseline or when duration information is uninformative.

    Args:
        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.

    Examples::

        >>> dur = UniformDuration(max_duration=8, num_classes=4)
        >>> bias = dur()
        >>> bias.shape
        torch.Size([8, 4])
        >>> (bias == 0).all()
        tensor(True)
    """

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__(max_duration, num_classes)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("_zeros", torch.zeros(max_duration, num_classes))

    def forward(self) -> Tensor:
        r"""forward() -> Tensor

        Return zero bias (uniform distribution).

        Returns:
            Tensor: Zeros of shape :math:`(K, C)`.
        """
        return self._zeros


def create_duration_distribution(
    distribution: Union[str, DurationDistribution, None],
    max_duration: int,
    num_classes: int,
    **kwargs,
) -> DurationDistribution:
    r"""create_duration_distribution(distribution, max_duration, num_classes, **kwargs) -> DurationDistribution

    Factory function to create duration distributions.

    Args:
        distribution (str, DurationDistribution, optional): Distribution type. Can be:

            - ``None`` or ``"learned"``: :class:`LearnedDuration` (default)
            - ``"geometric"``: :class:`GeometricDuration`
            - ``"negative_binomial"`` or ``"negbin"``: :class:`NegativeBinomialDuration`
            - ``"poisson"``: :class:`PoissonDuration`
            - ``"uniform"``: :class:`UniformDuration`
            - A :class:`DurationDistribution` instance (returned as-is)

        max_duration (int): Maximum segment duration :math:`K`.
        num_classes (int): Number of label classes :math:`C`.
        **kwargs: Additional arguments passed to the distribution constructor.

    Returns:
        DurationDistribution: A duration distribution instance.

    Raises:
        ValueError: If ``distribution`` is an unknown string.

    Examples::

        >>> # Create geometric distribution with custom initial logit
        >>> dur = create_duration_distribution("geometric", 8, 4, init_logit=-1.0)
        >>> dur
        GeometricDuration(...)

        >>> # Pass through existing instance
        >>> existing = LearnedDuration(8, 4)
        >>> dur = create_duration_distribution(existing, 8, 4)
        >>> dur is existing
        True
    """
    if distribution is None or distribution == "learned":
        return LearnedDuration(max_duration, num_classes, **kwargs)
    elif isinstance(distribution, DurationDistribution):
        return distribution
    elif distribution == "geometric":
        return GeometricDuration(max_duration, num_classes, **kwargs)
    elif distribution in ("negative_binomial", "negbin"):
        return NegativeBinomialDuration(max_duration, num_classes, **kwargs)
    elif distribution == "poisson":
        return PoissonDuration(max_duration, num_classes, **kwargs)
    elif distribution == "uniform":
        return UniformDuration(max_duration, num_classes)
    else:
        raise ValueError(
            f"Unknown duration distribution: {distribution}. "
            f"Options: learned, geometric, negative_binomial, poisson, uniform"
        )
