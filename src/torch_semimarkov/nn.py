r"""Neural network modules for Semi-Markov CRF.

This module provides :class:`torch.nn.Module` wrappers around the streaming Semi-CRF
kernels, making them easy to integrate with PyTorch Lightning and other training
frameworks.

Classes:
    :class:`SemiMarkovCRFHead`: Basic CRF head for sequence labeling.
    :class:`UncertaintySemiMarkovCRFHead`: Extended CRF head with uncertainty
        quantification methods for clinical applications.

For clinical applications requiring boundary uncertainty or focused learning,
use :class:`UncertaintySemiMarkovCRFHead` which provides:

- :meth:`~UncertaintyMixin.compute_boundary_marginals`: :math:`P(\text{boundary at position } t)`
- :meth:`~UncertaintyMixin.compute_position_marginals`: :math:`P(\text{label}=c \text{ at position } t)`
- :meth:`~UncertaintyMixin.compute_entropy_streaming`: Approximate entropy for uncertainty
- :meth:`~UncertaintyMixin.compute_loss_uncertainty_weighted`: Uncertainty-weighted loss for active learning

Examples::

    >>> from torch_semimarkov import UncertaintySemiMarkovCRFHead
    >>> model = UncertaintySemiMarkovCRFHead(num_classes=5, max_duration=100, hidden_dim=64)
    >>> boundary_probs = model.compute_boundary_marginals(hidden, lengths)

See Also:
    :mod:`torch_semimarkov.uncertainty`: Uncertainty quantification module
    :func:`~torch_semimarkov.streaming.semi_crf_streaming_forward`: Streaming API
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .streaming import semi_crf_streaming_forward

# Re-export uncertainty module for convenience
from .uncertainty import UncertaintyMixin, UncertaintySemiMarkovCRFHead

__all__ = ["SemiMarkovCRFHead", "UncertaintyMixin", "UncertaintySemiMarkovCRFHead"]


class SemiMarkovCRFHead(nn.Module):
    r"""CRF head for Semi-Markov sequence labeling.

    Wraps Triton streaming kernels in a simple :class:`torch.nn.Module` with learnable
    transition and duration parameters. Compatible with DDP - gradients sync
    automatically via standard PyTorch mechanisms.

    The module computes:

    - Partition function :math:`Z` via streaming forward algorithm
    - Gold sequence score for NLL loss computation
    - Memory: :math:`O(KC)` independent of sequence length :math:`T`

    Args:
        num_classes (int): Number of label classes (C).
        max_duration (int): Maximum segment duration (K).
        hidden_dim (int, optional): If provided, adds a projection layer from
            ``hidden_dim`` to ``num_classes``. Default: ``None``
        init_scale (float, optional): Scale for parameter initialization.
            Default: ``0.1``

    Attributes:
        transition (Parameter): Label transition scores of shape :math:`(C, C)`.
        duration_bias (Parameter): Duration-specific bias of shape :math:`(K, C)`.
        projection (Linear or None): Optional projection from encoder hidden dim.

    Examples::

        >>> import torch
        >>> from torch_semimarkov import SemiMarkovCRFHead
        >>>
        >>> # Create CRF head
        >>> crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)
        >>>
        >>> # Encoder output
        >>> batch, T = 4, 1000
        >>> hidden_states = torch.randn(batch, T, 512)
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Forward pass
        >>> result = crf(hidden_states, lengths)
        >>> print(result['partition'].shape)  # (batch,)
        >>>
        >>> # Training with labels
        >>> labels = torch.randint(0, 24, (batch, T))
        >>> loss = crf.compute_loss(hidden_states, lengths, labels)
        >>> loss.backward()

    .. note::
        For numerical stability at T > 100K, all computations are done in float32.
        When using with PyTorch Lightning, set ``precision=32`` in the trainer.

    See Also:
        :class:`UncertaintySemiMarkovCRFHead`: Extended version with uncertainty methods
        :func:`~torch_semimarkov.streaming.semi_crf_streaming_forward`: Underlying API
    """

    def __init__(
        self,
        num_classes: int,
        max_duration: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_duration = max_duration

        # CRF parameters
        self.transition = nn.Parameter(torch.randn(num_classes, num_classes) * init_scale)
        self.duration_bias = nn.Parameter(torch.randn(max_duration, num_classes) * init_scale)

        # Optional projection from encoder hidden dim
        if hidden_dim is not None:
            self.projection = nn.Linear(hidden_dim, num_classes)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
    ) -> dict:
        r"""forward(hidden_states, lengths, use_triton=True) -> dict

        Compute partition function from encoder hidden states.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                if projection is enabled, or :math:`(\text{batch}, T, C)` if projection is ``None``.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``

        Returns:
            dict: Dictionary containing:

            - **partition** (Tensor): Log partition function of shape :math:`(\text{batch},)`.
            - **cum_scores** (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`
              for loss computation.
        """
        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)  # (batch, T, C)
        else:
            scores = hidden_states

        # Build cumulative scores for prefix-sum edge retrieval
        # CRITICAL: Use float32 for numerical stability at T > 100K
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float32, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Compute partition function via streaming algorithm
        partition = semi_crf_streaming_forward(
            cum_scores,
            self.transition,
            self.duration_bias,
            lengths,
            self.max_duration,
            semiring="log",
            use_triton=use_triton,
        )

        return {"partition": partition, "cum_scores": cum_scores}

    def compute_loss(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        labels: Tensor,
        use_triton: bool = True,
        reduction: str = "mean",
    ) -> Tensor:
        r"""compute_loss(hidden_states, lengths, labels, use_triton=True, reduction="mean") -> Tensor

        Compute negative log-likelihood loss.

        The NLL loss is computed as:

        .. math::
            \text{NLL} = \log Z - \text{score}(y^*)

        where :math:`Z` is the partition function and :math:`y^*` is the gold segmentation.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`. Each position
                has a label ID. Segments are extracted by finding where labels change.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``
            reduction (str, optional): Reduction mode: ``"mean"``, ``"sum"``, or ``"none"``.
                Default: ``"mean"``

        Returns:
            Tensor: NLL loss. Scalar if reduction is ``"mean"`` or ``"sum"``,
            shape :math:`(\text{batch},)` if ``"none"``.
        """
        result = self.forward(hidden_states, lengths, use_triton)
        partition = result["partition"]
        cum_scores = result["cum_scores"]

        # Score the gold segmentation
        gold_score = self._score_gold(cum_scores, labels, lengths)

        # NLL = partition - gold_score
        nll = partition - gold_score

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        return nll

    def _score_gold(
        self,
        cum_scores: Tensor,
        labels: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""_score_gold(cum_scores, labels, lengths) -> Tensor

        Score the gold segmentation.

        Extracts segments from per-position labels (where label changes indicate
        segment boundaries) and computes:

        .. math::
            \text{score} = \sum_i \text{content}_i + \sum_i \text{duration\_bias}_i + \sum_i \text{transition}_i

        Args:
            cum_scores (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`.
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Gold sequence scores of shape :math:`(\text{batch},)`.
        """
        batch = cum_scores.shape[0]
        device = cum_scores.device
        scores = torch.zeros(batch, device=device, dtype=cum_scores.dtype)

        for b in range(batch):
            seq_len = lengths[b].item()
            if seq_len == 0:
                continue

            seq_labels = labels[b, :seq_len]

            # Find segment boundaries (where label changes)
            # A segment ends at position t if label[t] != label[t+1] or t is last position
            changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

            # Segment end positions (inclusive)
            seg_ends = torch.cat([changes, torch.tensor([seq_len - 1], device=device)])
            # Segment start positions
            seg_starts = torch.cat([torch.tensor([0], device=device), changes + 1])

            prev_label = None
            for i in range(len(seg_starts)):
                start = seg_starts[i].item()
                end = seg_ends[i].item()
                duration = end - start + 1
                label = seq_labels[start].item()

                # Content score: cum_scores[end+1] - cum_scores[start]
                # Note: cum_scores is 1-indexed (cum_scores[0] = 0)
                content_score = cum_scores[b, end + 1, label] - cum_scores[b, start, label]
                scores[b] += content_score

                # Duration bias (clamped to max_duration)
                dur_idx = min(duration, self.max_duration) - 1  # 0-indexed
                if dur_idx >= 0 and dur_idx < self.max_duration:
                    scores[b] += self.duration_bias[dur_idx, label]

                # Transition score (skip for first segment)
                if prev_label is not None:
                    scores[b] += self.transition[prev_label, label]

                prev_label = label

        return scores

    def decode(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
    ) -> Tensor:
        r"""decode(hidden_states, lengths, use_triton=True) -> Tensor

        Decode best segmentation using Viterbi algorithm.

        Computes the maximum score over all valid segmentations using the
        max semiring (Viterbi decoding).

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``

        Returns:
            Tensor: Best score (max over all segmentations) of shape :math:`(\text{batch},)`.

        .. note::
            This returns the score, not the actual segmentation. For the full
            segmentation path, use the :class:`~torch_semimarkov.SemiMarkov` class
            with :class:`~torch_semimarkov.semirings.MaxSemiring` and extract via marginals.
        """
        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build cumulative scores
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float32, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Use max semiring for Viterbi
        max_score = semi_crf_streaming_forward(
            cum_scores,
            self.transition,
            self.duration_bias,
            lengths,
            self.max_duration,
            semiring="max",
            use_triton=use_triton,
        )

        return max_score

    def extra_repr(self) -> str:
        parts = [
            f"num_classes={self.num_classes}",
            f"max_duration={self.max_duration}",
        ]
        if self.projection is not None:
            parts.append(f"hidden_dim={self.projection.in_features}")
        return ", ".join(parts)
