"""
Duration distribution classes for Semi-Markov CRF.

This module provides flexible duration parameterization through a factory pattern.
Different distributions can be used to model segment durations, allowing for
domain-specific priors or learned parameters.

Example usage:
    >>> from torch_semimarkov.duration import LearnedDuration, GeometricDuration
    >>>
    >>> # Default learned duration bias (current behavior)
    >>> dur = LearnedDuration(max_duration=8, num_classes=4)
    >>> bias = dur()  # Returns (K, C) tensor
    >>>
    >>> # Geometric distribution with learned rate
    >>> dur = GeometricDuration(max_duration=8, num_classes=4)
    >>> bias = dur()  # Returns (K, C) tensor with geometric shape
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class DurationDistribution(nn.Module, ABC):
    """Base class for duration distributions.

    Duration distributions produce a bias tensor of shape (K, C) where:
    - K is the maximum segment duration
    - C is the number of classes/labels

    The bias is added to segment scores in the Semi-Markov CRF, effectively
    implementing a prior over segment durations for each class.
    """

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__()
        self.max_duration = max_duration
        self.num_classes = num_classes

    @abstractmethod
    def forward(self) -> Tensor:
        """Compute duration bias tensor.

        Returns:
            Tensor of shape (K, C) containing log-space duration biases.
        """
        raise NotImplementedError


class LearnedDuration(DurationDistribution):
    """Fully learned duration bias (default behavior).

    Each (duration, class) combination has an independent learned parameter.
    This is the most flexible but requires the most data to learn.
    """

    def __init__(self, max_duration: int, num_classes: int, init_std: float = 0.1):
        super().__init__(max_duration, num_classes)
        self.duration_bias = nn.Parameter(torch.randn(max_duration, num_classes) * init_std)

    def forward(self) -> Tensor:
        return self.duration_bias


class GeometricDuration(DurationDistribution):
    """Geometric duration distribution: P(k) ∝ p(1-p)^(k-1).

    Models durations with exponential decay, where short segments are more
    likely than long ones. The rate parameter p can be learned per-class.

    In log space: log P(k) = log(p) + (k-1) * log(1-p)
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
    """Negative binomial duration distribution.

    Generalizes geometric distribution with an additional shape parameter r.
    P(k) ∝ C(k+r-2, k-1) * p^r * (1-p)^(k-1)

    When r=1, reduces to geometric distribution.
    Larger r values create more peaked distributions around the mode.
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

        return log_prob


class PoissonDuration(DurationDistribution):
    """Poisson-like duration distribution: P(k) ∝ λ^k / k!.

    Models durations centered around the mean λ. Useful when segments
    have a characteristic length.

    Note: This is a shifted Poisson (k starts at 1, not 0).
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
    """User-provided callable for custom duration distributions.

    Allows full flexibility by accepting any function that returns
    a (K, C) tensor of log-probabilities.

    Example:
        >>> def my_duration(K, C, device):
        ...     # Custom duration logic
        ...     return torch.zeros(K, C, device=device)
        >>> dur = CallableDuration(8, 4, my_duration)
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
        device = self._device_tracker.device
        return self._func(self.max_duration, self.num_classes, device)


class UniformDuration(DurationDistribution):
    """Uniform duration distribution (no duration preference).

    All durations have equal probability: log P(k) = 0 for all k.
    Useful as a baseline or when duration information is uninformative.
    """

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__(max_duration, num_classes)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("_zeros", torch.zeros(max_duration, num_classes))

    def forward(self) -> Tensor:
        return self._zeros


def create_duration_distribution(
    distribution: Union[str, DurationDistribution, None],
    max_duration: int,
    num_classes: int,
    **kwargs,
) -> DurationDistribution:
    """Factory function to create duration distributions.

    Args:
        distribution: Distribution type. Can be:
            - None or "learned": LearnedDuration (default)
            - "geometric": GeometricDuration
            - "negative_binomial" or "negbin": NegativeBinomialDuration
            - "poisson": PoissonDuration
            - "uniform": UniformDuration
            - A DurationDistribution instance (returned as-is)
        max_duration: Maximum segment duration K
        num_classes: Number of classes C
        **kwargs: Additional arguments passed to the distribution constructor

    Returns:
        A DurationDistribution instance

    Example:
        >>> dur = create_duration_distribution("geometric", 8, 4, init_logit=-1.0)
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
