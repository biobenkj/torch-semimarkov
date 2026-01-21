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

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .duration import DurationDistribution, LearnedDuration, create_duration_distribution
from .helpers import Segment, ViterbiResult, score_gold_vectorized
from .streaming import semi_crf_streaming_forward
from .streaming.constants import NEG_INF
from .streaming.pytorch_reference import compute_edge_block_streaming

# Re-export uncertainty module for convenience
from .uncertainty import UncertaintyMixin, UncertaintySemiMarkovCRFHead

__all__ = [
    "SemiMarkovCRFHead",
    "UncertaintyMixin",
    "UncertaintySemiMarkovCRFHead",
    "Segment",
    "ViterbiResult",
]


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
        duration_distribution (str, DurationDistribution, optional): Duration
            distribution to use. Can be:

            - ``None`` or ``"learned"``: Fully learned bias (default, current behavior)
            - ``"geometric"``: Geometric distribution with learnable rate
            - ``"negative_binomial"`` or ``"negbin"``: Negative binomial distribution
            - ``"poisson"``: Poisson-like distribution
            - ``"uniform"``: Uniform (no duration preference)
            - A :class:`~torch_semimarkov.duration.DurationDistribution` instance

            Default: ``None`` (uses learned duration bias)

    Attributes:
        transition (Parameter): Label transition scores of shape :math:`(C, C)`.
        duration_dist (DurationDistribution): Duration distribution module.
        projection (Linear or None): Optional projection from encoder hidden dim.

    Properties:
        duration_bias: Returns the current duration bias tensor of shape :math:`(K, C)`.
            This is a property for backward compatibility - internally uses ``duration_dist()``.

    Examples::

        >>> import torch
        >>> from torch_semimarkov import SemiMarkovCRFHead
        >>>
        >>> # Create CRF head with default learned duration
        >>> crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)
        >>>
        >>> # Or with geometric duration distribution
        >>> crf = SemiMarkovCRFHead(
        ...     num_classes=24, max_duration=100, hidden_dim=512,
        ...     duration_distribution="geometric"
        ... )
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
        :mod:`torch_semimarkov.duration`: Available duration distributions
    """

    def __init__(
        self,
        num_classes: int,
        max_duration: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.1,
        duration_distribution: Optional[Union[str, DurationDistribution]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_duration = max_duration

        # CRF parameters
        self.transition = nn.Parameter(torch.randn(num_classes, num_classes) * init_scale)

        # Duration distribution (supports multiple parameterizations)
        if duration_distribution is None or duration_distribution == "learned":
            # Default: fully learned, use init_scale for consistency
            self.duration_dist = LearnedDuration(max_duration, num_classes, init_std=init_scale)
        else:
            self.duration_dist = create_duration_distribution(
                duration_distribution, max_duration, num_classes
            )

        # Optional projection from encoder hidden dim
        if hidden_dim is not None:
            self.projection = nn.Linear(hidden_dim, num_classes)
        else:
            self.projection = None

    @property
    def duration_bias(self) -> Tensor:
        """Duration bias tensor of shape (K, C).

        This property provides backward compatibility. Internally calls
        ``self.duration_dist()`` to compute the duration bias.
        """
        return self.duration_dist()

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
        return score_gold_vectorized(
            cum_scores=cum_scores,
            labels=labels,
            lengths=lengths,
            transition=self.transition,
            duration_bias=self.duration_bias,
            max_duration=self.max_duration,
        )

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

    def decode_with_traceback(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        max_traceback_length: int = 10000,
    ) -> ViterbiResult:
        r"""Decode best segmentation with full path reconstruction.

        Computes the maximum-scoring segmentation using Viterbi algorithm
        and returns both the score and the actual segment boundaries.

        Args:
            hidden_states (Tensor): Encoder output of shape
                :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            max_traceback_length (int, optional): Maximum sequence length for
                traceback. Sequences longer than this will have empty segment
                lists (score is still computed). Default: ``10000``

        Returns:
            ViterbiResult: Named tuple containing:
                - **scores** (Tensor): Best scores of shape :math:`(\text{batch},)`.
                - **segments** (List[List[Segment]]): Per-batch segment lists.

        Note:
            For very long sequences (T > max_traceback_length), traceback requires
            O(T × C) memory which may not be feasible. In such cases, the returned
            segments list will be empty but scores are still computed.

        Example::

            >>> result = crf.decode_with_traceback(hidden_states, lengths)
            >>> print(f"Score: {result.scores[0].item():.2f}")
            >>> for seg in result.segments[0]:
            ...     print(f"  [{seg.start}, {seg.end}] label={seg.label}")
        """
        batch, T, _ = hidden_states.shape
        device = hidden_states.device

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build cumulative scores
        cum_scores = torch.zeros(batch, T + 1, self.num_classes, dtype=torch.float32, device=device)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Get max scores using streaming API
        max_scores = semi_crf_streaming_forward(
            cum_scores,
            self.transition,
            self.duration_bias,
            lengths,
            self.max_duration,
            semiring="max",
            use_triton=False,  # Use PyTorch for traceback compatibility
        )

        # Perform traceback for each sequence
        all_segments: list[list[Segment]] = []

        for b in range(batch):
            seq_len = lengths[b].item()

            # Skip traceback for very long sequences
            if seq_len > max_traceback_length:
                all_segments.append([])
                continue

            if seq_len == 0:
                all_segments.append([])
                continue

            # Run forward pass with backpointer storage for this sequence
            segments = self._traceback_single(
                cum_scores[b : b + 1],
                seq_len,
            )
            all_segments.append(segments)

        return ViterbiResult(scores=max_scores, segments=all_segments)

    def _traceback_single(
        self,
        cum_scores: Tensor,
        seq_len: int,
    ) -> list[Segment]:
        """Traceback for a single sequence to recover optimal segmentation.

        Args:
            cum_scores: Cumulative scores of shape (1, T+1, C).
            seq_len: Actual sequence length.

        Returns:
            List of Segment objects forming the optimal segmentation.
        """
        device = cum_scores.device
        C = self.num_classes
        K = self.max_duration

        # Allocate alpha table and backpointers
        # alpha[t, c] = best score to reach position t ending in state c
        alpha = torch.full((seq_len + 1, C), NEG_INF, device=device, dtype=torch.float32)
        alpha[0, :] = 0.0  # Start: all states equally valid

        # Backpointers: for each (t, c), store (best_k, best_c_src)
        bp_k = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)
        bp_c = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)

        # Forward pass with backpointer storage
        for t in range(1, seq_len + 1):
            k_eff = min(K - 1, t)

            for k in range(1, max(k_eff + 1, 2)):  # max ensures K=1 processes duration 1
                start = t - k

                # Compute edge block for this (start, k)
                edge_block = compute_edge_block_streaming(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    start,
                    k,
                )  # (1, C_dest, C_src)

                # scores[c_dest, c_src] = alpha[start, c_src] + edge[c_dest, c_src]
                candidate_scores = alpha[start, :].unsqueeze(0) + edge_block[0]  # (C_dest, C_src)

                # Find best c_src for each c_dest
                best_scores_from_src, best_c_src = candidate_scores.max(dim=-1)  # (C_dest,)

                # Update alpha and backpointers where this k gives better score
                better_mask = best_scores_from_src > alpha[t, :]
                alpha[t, :] = torch.where(better_mask, best_scores_from_src, alpha[t, :])
                bp_k[t, :] = torch.where(better_mask, k, bp_k[t, :])
                bp_c[t, :] = torch.where(better_mask, best_c_src, bp_c[t, :])

        # Traceback from final position
        final_scores = alpha[seq_len, :]
        best_final_c = final_scores.argmax().item()

        segments: list[Segment] = []
        t = seq_len
        c = best_final_c

        while t > 0:
            k = bp_k[t, c].item()
            c_prev = bp_c[t, c].item()

            if k == 0:
                # Safety check - shouldn't happen with proper initialization
                break

            start = t - k
            end = t - 1  # Segment is [start, end] inclusive

            # Compute segment score contribution
            # Always include transition, even for first segment - the forward pass
            # computes: max_{c_src} [alpha[0, c_src] + segment_score + transition[c_src, c_dest]]
            # where alpha[0, c_src] = 0, so transition from c_prev is part of the score
            seg_score = self._compute_segment_score(cum_scores[0], start, end, c, c_prev)

            segments.append(Segment(start=start, end=end, label=c, score=seg_score))

            t = start
            c = c_prev

        # Reverse to get segments in order
        segments.reverse()
        return segments

    def _compute_segment_score(
        self,
        cum_scores: Tensor,
        start: int,
        end: int,
        label: int,
        prev_label: Optional[int],
    ) -> float:
        """Compute the score contribution of a single segment."""
        # Content score
        content = (cum_scores[end + 1, label] - cum_scores[start, label]).item()

        # Duration bias (duration_bias[k] stores bias for segments of duration k)
        duration = end - start + 1
        dur_idx = min(duration, self.max_duration - 1)  # Clamp to valid range
        dur_bias = self.duration_bias[dur_idx, label].item()

        # Transition (if not first segment)
        trans = 0.0
        if prev_label is not None:
            trans = self.transition[prev_label, label].item()

        return content + dur_bias + trans

    def extra_repr(self) -> str:
        parts = [
            f"num_classes={self.num_classes}",
            f"max_duration={self.max_duration}",
        ]
        if self.projection is not None:
            parts.append(f"hidden_dim={self.projection.in_features}")
        parts.append(f"duration_dist={self.duration_dist.__class__.__name__}")
        return ", ".join(parts)
