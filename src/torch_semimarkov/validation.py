"""Input validation utilities for Semi-Markov CRF.

This module provides validation functions for common inputs to the Semi-Markov CRF
API. These functions raise informative errors early, preventing cryptic downstream
failures.

Functions:
    validate_hidden_states: Validate hidden_states tensor shape and properties.
    validate_lengths: Validate lengths tensor bounds and properties.
    validate_labels: Validate labels tensor shape and value range.
    validate_cum_scores: Validate cumulative scores tensor for streaming API.
    validate_device_consistency: Validate all tensors are on the same device.
"""

import warnings
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "validate_hidden_states",
    "validate_lengths",
    "validate_labels",
    "validate_cum_scores",
    "validate_device_consistency",
]


def validate_hidden_states(
    hidden_states: Tensor,
    name: str = "hidden_states",
    check_nan: bool = True,
    check_inf: bool = True,
) -> None:
    r"""validate_hidden_states(hidden_states, name='hidden_states', check_nan=True, check_inf=True) -> None

    Validates hidden_states tensor shape and properties.

    Args:
        hidden_states (Tensor): tensor to validate, expected shape
          :math:`(\text{batch}, T, \text{features})`
        name (str, optional): name to use in error messages. Default: ``"hidden_states"``
        check_nan (bool, optional): whether to check for NaN values. Default: ``True``
        check_inf (bool, optional): whether to check for Inf values. Default: ``True``

    Raises:
        ValueError: If tensor has wrong number of dimensions.
        ValueError: If tensor contains NaN values (when ``check_nan=True``).
        ValueError: If tensor contains Inf values (when ``check_inf=True``).

    Examples::

        >>> hidden = torch.randn(2, 100, 64)
        >>> validate_hidden_states(hidden)  # OK

        >>> bad_hidden = torch.randn(100, 64)  # Missing batch dim
        >>> validate_hidden_states(bad_hidden)
        ValueError: hidden_states must be 3D (batch, T, features), got 2D
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"{name} must be 3D (batch, T, features), got {hidden_states.ndim}D")

    if check_nan and torch.isnan(hidden_states).any():
        raise ValueError(f"{name} contains NaN values")

    if check_inf and torch.isinf(hidden_states).any():
        raise ValueError(f"{name} contains Inf values")


def validate_lengths(
    lengths: Tensor,
    max_length: int,
    batch_size: Optional[int] = None,
    name: str = "lengths",
) -> None:
    r"""validate_lengths(lengths, max_length, batch_size=None, name='lengths') -> None

    Validates lengths tensor bounds and properties.

    Args:
        lengths (Tensor): tensor to validate, expected shape :math:`(\text{batch},)`
        max_length (int): maximum allowed length (typically :math:`T` from hidden_states)
        batch_size (int, optional): expected batch size. If provided, validates
          ``lengths.shape[0]``. Default: ``None``
        name (str, optional): name to use in error messages. Default: ``"lengths"``

    Raises:
        ValueError: If tensor is not 1D.
        ValueError: If batch size doesn't match (when ``batch_size`` provided).
        ValueError: If any length is :math:`\leq 0`.
        ValueError: If any length exceeds ``max_length``.

    Examples::

        >>> lengths = torch.tensor([100, 100])
        >>> validate_lengths(lengths, max_length=100)  # OK

        >>> bad_lengths = torch.tensor([100, 200])
        >>> validate_lengths(bad_lengths, max_length=100)
        ValueError: lengths cannot exceed T=100, got max=200
    """
    if lengths.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {lengths.ndim}D")

    if batch_size is not None and lengths.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size {lengths.shape[0]} doesn't match expected {batch_size}"
        )

    # Check for non-positive lengths
    if (lengths <= 0).any():
        raise ValueError(f"{name} must be positive, got min={lengths.min().item()}")

    # Check for lengths exceeding max
    if (lengths > max_length).any():
        raise ValueError(f"{name} cannot exceed T={max_length}, got max={lengths.max().item()}")


def validate_labels(
    labels: Tensor,
    num_classes: int,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    name: str = "labels",
) -> None:
    r"""validate_labels(labels, num_classes, batch_size=None, seq_length=None, name='labels') -> None

    Validates labels tensor shape and value range.

    Args:
        labels (Tensor): tensor to validate, expected shape :math:`(\text{batch}, T)`
        num_classes (int): number of valid classes (labels must be in
          :math:`[0, \text{num\_classes})`)
        batch_size (int, optional): expected batch size. If provided, validates
          ``labels.shape[0]``. Default: ``None``
        seq_length (int, optional): expected sequence length. If provided, validates
          ``labels.shape[1]``. Default: ``None``
        name (str, optional): name to use in error messages. Default: ``"labels"``

    Raises:
        ValueError: If tensor is not 2D.
        ValueError: If batch size doesn't match (when ``batch_size`` provided).
        ValueError: If sequence length doesn't match (when ``seq_length`` provided).
        ValueError: If any label is outside :math:`[0, \text{num\_classes})`.

    Examples::

        >>> labels = torch.randint(0, 4, (2, 100))
        >>> validate_labels(labels, num_classes=4)  # OK

        >>> bad_labels = torch.randint(0, 10, (2, 100))
        >>> validate_labels(bad_labels, num_classes=4)
        ValueError: labels must be in [0, 4), got range [0, 9]
    """
    if labels.ndim != 2:
        raise ValueError(f"{name} must be 2D (batch, T), got {labels.ndim}D")

    if batch_size is not None and labels.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {labels.shape[0]} doesn't match expected {batch_size}")

    if seq_length is not None and labels.shape[1] != seq_length:
        raise ValueError(
            f"{name} sequence length {labels.shape[1]} doesn't match expected {seq_length}"
        )

    # Check value range
    min_val = labels.min().item()
    max_val = labels.max().item()
    if min_val < 0 or max_val >= num_classes:
        raise ValueError(f"{name} must be in [0, {num_classes}), got range [{min_val}, {max_val}]")


def validate_cum_scores(
    cum_scores: Tensor,
    name: str = "cum_scores",
    warn_dtype: bool = True,
    check_leading_zeros: bool = False,
) -> None:
    r"""validate_cum_scores(cum_scores, name='cum_scores', warn_dtype=True, check_leading_zeros=False) -> None

    Validates cumulative scores tensor for streaming API.

    The cumulative scores tensor stores prefix sums of emission scores, enabling
    efficient :math:`O(1)` range queries for segment scoring in the Semi-Markov CRF.

    Args:
        cum_scores (Tensor): tensor to validate, expected shape
          :math:`(\text{batch}, T+1, C)` where :math:`C` is the number of classes
        name (str, optional): name to use in error messages. Default: ``"cum_scores"``
        warn_dtype (bool, optional): whether to warn if dtype is not ``float32``.
          Default: ``True``
        check_leading_zeros (bool, optional): whether to verify
          ``cum_scores[:, 0, :] == 0``. Default: ``False``

    Raises:
        ValueError: If tensor is not 3D.
        ValueError: If :math:`T+1` dimension is :math:`< 2` (need at least :math:`T=1`).

    Warns:
        UserWarning: If dtype is not ``float32`` (when ``warn_dtype=True``).

    Examples::

        >>> cum_scores = torch.zeros(2, 101, 4)
        >>> validate_cum_scores(cum_scores)  # OK

        >>> bad_cum = torch.zeros(2, 1, 4)  # T=0, invalid
        >>> validate_cum_scores(bad_cum)
        ValueError: cum_scores T+1 dimension must be >= 2, got 1
    """
    if cum_scores.ndim != 3:
        raise ValueError(f"{name} must be 3D (batch, T+1, C), got {cum_scores.ndim}D")

    batch, T_plus_1, C = cum_scores.shape
    if T_plus_1 < 2:
        raise ValueError(f"{name} T+1 dimension must be >= 2 (need at least T=1), got {T_plus_1}")

    if warn_dtype and cum_scores.dtype != torch.float32:
        warnings.warn(
            f"{name} should be float32 for numerical stability at long sequences, "
            f"got {cum_scores.dtype}",
            UserWarning,
            stacklevel=3,
        )

    if check_leading_zeros:
        leading = cum_scores[:, 0, :]
        if not torch.allclose(leading, torch.zeros_like(leading)):
            warnings.warn(
                f"{name}[:, 0, :] should be zeros (cumsum convention), "
                f"got max abs value {leading.abs().max().item():.2e}",
                UserWarning,
                stacklevel=3,
            )


def validate_device_consistency(
    *tensors: Tensor,
    names: Optional[list[str]] = None,
) -> None:
    r"""validate_device_consistency(*tensors, names=None) -> None

    Validates that all tensors are on the same device.

    Args:
        *tensors (Tensor): tensors to check (``None`` values are skipped)
        names (list[str], optional): list of names for error messages.
          Default: ``None``

    Raises:
        ValueError: If tensors are on different devices.

    Examples::

        >>> t1 = torch.randn(2, 3)
        >>> t2 = torch.randn(2, 3)
        >>> validate_device_consistency(t1, t2, names=["a", "b"])  # OK (both CPU)

        >>> # With CUDA available:
        >>> t1_cuda = t1.cuda()
        >>> validate_device_consistency(t1, t1_cuda, names=["cpu_t", "cuda_t"])
        ValueError: Device mismatch: {'cpu_t': device(type='cpu'), 'cuda_t': device(type='cuda', index=0)}
    """
    # Filter out None values
    valid_tensors = [t for t in tensors if t is not None]
    if len(valid_tensors) <= 1:
        return  # Nothing to compare

    # Get devices
    devices = [t.device for t in valid_tensors]

    # Check if all devices are the same
    if len({str(d) for d in devices}) > 1:
        if names is not None:
            valid_names = [n for n, t in zip(names, tensors, strict=False) if t is not None]
            device_map = dict(zip(valid_names, devices, strict=True))
        else:
            device_map = {f"tensor_{i}": d for i, d in enumerate(devices)}
        raise ValueError(f"Device mismatch: {device_map}")
