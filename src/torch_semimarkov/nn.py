"""Neural network modules for Semi-Markov CRF.

This module provides nn.Module wrappers around the streaming Semi-CRF kernels,
making them easy to integrate with PyTorch Lightning and other training frameworks.

Classes:
    SemiMarkovCRFHead: Basic CRF head for sequence labeling.
    UncertaintySemiMarkovCRFHead: Extended CRF head with uncertainty quantification
        methods for clinical applications. See :mod:`torch_semimarkov.uncertainty`.

For clinical applications requiring boundary uncertainty or focused learning,
use :class:`UncertaintySemiMarkovCRFHead` which provides:
    - ``compute_boundary_marginals()``: P(boundary at position t)
    - ``compute_position_marginals()``: P(label=c at position t)
    - ``compute_entropy_streaming()``: Approximate entropy for uncertainty
    - ``compute_loss_uncertainty_weighted()``: Uncertainty-weighted loss for active learning

Example::

    >>> from torch_semimarkov import UncertaintySemiMarkovCRFHead
    >>> model = UncertaintySemiMarkovCRFHead(num_classes=5, max_duration=100, hidden_dim=64)
    >>> boundary_probs = model.compute_boundary_marginals(hidden, lengths)
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
    """CRF head for Semi-Markov sequence labeling.

    Wraps Triton streaming kernels in a simple nn.Module with learnable
    transition and duration parameters. Compatible with DDP - gradients
    sync automatically via standard PyTorch mechanisms.

    The module computes:
        - Partition function Z via streaming forward algorithm
        - Gold sequence score for NLL loss computation
        - Memory: O(KC) independent of sequence length T

    Args:
        num_classes: Number of label classes (C).
        max_duration: Maximum segment duration (K).
        hidden_dim: If provided, adds a projection layer from hidden_dim to num_classes.
        init_scale: Scale for parameter initialization. Default: 0.1.

    Example::

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

    Note:
        For numerical stability at T > 100K, all computations are done in float32.
        When using with PyTorch Lightning, set ``precision=32`` in the trainer.
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
        """Compute partition function from encoder hidden states.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) if projection
                is enabled, or (batch, T, num_classes) if projection is None.
            lengths: Sequence lengths of shape (batch,).
            use_triton: Whether to use Triton kernels (default: True).

        Returns:
            Dictionary with:
                - 'partition': Log partition function of shape (batch,)
                - 'cum_scores': Cumulative scores of shape (batch, T+1, C) for loss computation
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
        """Compute negative log-likelihood loss.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).
            labels: Per-position labels of shape (batch, T). Each position has a label ID.
                Segments are extracted by finding where labels change.
            use_triton: Whether to use Triton kernels (default: True).
            reduction: 'mean', 'sum', or 'none'. Default: 'mean'.

        Returns:
            NLL loss. Scalar if reduction is 'mean' or 'sum', (batch,) if 'none'.
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
        """Score the gold segmentation.

        Extracts segments from per-position labels (where label changes indicate
        segment boundaries) and computes:
            score = sum(content_scores) + sum(duration_biases) + sum(transitions)

        Args:
            cum_scores: Cumulative scores of shape (batch, T+1, C).
            labels: Per-position labels of shape (batch, T).
            lengths: Sequence lengths of shape (batch,).

        Returns:
            Gold sequence scores of shape (batch,).
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
        """Decode best segmentation using Viterbi algorithm.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).
            use_triton: Whether to use Triton kernels (default: True).

        Returns:
            Best score (max over all segmentations) of shape (batch,).

        Note:
            This returns the score, not the actual segmentation. For the full
            segmentation path, use the SemiMarkov class with argmax.
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
