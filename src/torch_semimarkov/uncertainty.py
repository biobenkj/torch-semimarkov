r"""Uncertainty quantification for Semi-Markov CRF.

This module provides uncertainty quantification methods for the
:class:`~torch_semimarkov.nn.SemiMarkovCRFHead`, enabling clinical applications
where boundary confidence is critical.

Two approaches are supported:

1. **Streaming-compatible** (:math:`T \geq 10K`): Uses gradient-based marginals
   from backward pass
2. **Exact** (:math:`T < 10K`): Uses :meth:`SemiMarkov.marginals` and
   :class:`~torch_semimarkov.semirings.EntropySemiring` with pre-computed edges

Key insight: The gradient of :math:`\log Z` w.r.t. cumulative scores gives marginal
information via the forward-backward algorithm. This works with the streaming API
for any sequence length.

See Also:
    :class:`UncertaintySemiMarkovCRFHead`: CRF head with uncertainty methods
    :class:`UncertaintyMixin`: Mixin class providing uncertainty methods
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


class UncertaintyMixin:
    r"""Mixin providing uncertainty quantification for SemiMarkovCRFHead.

    Uses gradient-based marginals for streaming API compatibility (:math:`T \geq 10K`).
    Falls back to exact methods for short sequences when edges fit in memory.

    This mixin provides the following methods:

    - :meth:`compute_boundary_marginals`: :math:`P(\text{boundary at position } t)` for each position
    - :meth:`compute_position_marginals`: Per-position label distributions
    - :meth:`compute_entropy_streaming`: Approximate entropy from marginal distribution
    - :meth:`compute_entropy_exact`: Exact entropy via :class:`~torch_semimarkov.semirings.EntropySemiring`
      (:math:`T < 10K` only)
    - :meth:`compute_loss_uncertainty_weighted`: Uncertainty-weighted NLL loss

    .. note::
        Classes using this mixin must have ``projection``, ``transition``,
        ``duration_bias``, ``num_classes``, and ``max_duration`` attributes.
    """

    def compute_boundary_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_streaming: bool = True,
        normalize: bool = True,
    ) -> Tensor:
        r"""compute_boundary_marginals(hidden_states, lengths, use_streaming=True, normalize=True) -> Tensor

        Compute :math:`P(\text{boundary at position } t)` for each position.

        For streaming mode (:math:`T \geq 10K`): Uses gradient of :math:`\log Z` w.r.t.
        cumulative scores. For exact mode (:math:`T < 10K`): Uses
        :meth:`SemiMarkov.marginals` if edges fit in memory.

        The gradient of the log partition function w.r.t. the cumulative scores
        encodes information about which positions contribute to the partition,
        which correlates with boundary probability.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_streaming (bool, optional): If ``True``, use streaming gradient-based method.
                Default: ``True``
            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]` range.
                Default: ``True``

        Returns:
            Tensor: Boundary probabilities of shape :math:`(\text{batch}, T)`.
        """
        if use_streaming:
            return self._compute_boundary_marginals_streaming(hidden_states, lengths, normalize)
        else:
            return self._compute_boundary_marginals_exact(hidden_states, lengths, normalize)

    def _compute_boundary_marginals_streaming(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        r"""Compute boundary marginals via streaming gradient method.

        Uses the fact that :math:`\nabla_{\text{cum\_scores}} \log Z` encodes marginal
        information from the forward-backward algorithm.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]`. Default: ``True``

        Returns:
            Tensor: Boundary signal of shape :math:`(\text{batch}, T)`.
        """
        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            with torch.no_grad():
                scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Use enable_grad() to ensure gradient computation works even if called
        # from within a no_grad() context (e.g., from compute_loss_uncertainty_weighted)
        with torch.enable_grad():
            # Create a fresh tensor with gradients
            cum_scores = torch.zeros(
                batch,
                T + 1,
                self.num_classes,
                dtype=torch.float32,
                device=scores.device,
            )
            cum_scores[:, 1:] = torch.cumsum(scores.float().detach(), dim=1)
            cum_scores = cum_scores.requires_grad_(True)

            # Forward pass
            partition = semi_crf_streaming_forward(
                cum_scores,
                self.transition.detach(),
                self.duration_bias.detach(),
                lengths,
                self.max_duration,
                semiring="log",
                use_triton=False,  # Use PyTorch for reliable gradients
            )

            # Backward to get marginals
            partition.sum().backward()

        # Extract gradient (encodes marginal information)
        grad = cum_scores.grad  # (batch, T+1, C)

        # Boundary probability approximation:
        # Large gradient magnitude at position t suggests t is informative for
        # the partition function, which correlates with segment boundaries.
        # We use the absolute gradient summed over classes.
        boundary_signal = grad[:, 1:].abs().sum(dim=-1)  # (batch, T)

        if normalize:
            # Normalize per sequence to [0, 1]
            max_val = boundary_signal.max(dim=-1, keepdim=True)[0] + 1e-8
            boundary_signal = boundary_signal / max_val

        return boundary_signal

    def _compute_boundary_marginals_exact(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        r"""Compute boundary marginals via exact edge marginals (:math:`T < 10K`).

        Uses :meth:`SemiMarkov.marginals` which requires materializing edge tensor.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]`. Default: ``True``

        Returns:
            Tensor: Boundary probabilities of shape :math:`(\text{batch}, T)`.

        .. warning::
            Will OOM for large T - use streaming method instead.
        """
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring

        batch, T, _ = hidden_states.shape

        # Project to label space
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build edge tensor (WARNING: O(T*K*C^2) memory!)
        edge = self._build_edge_tensor(scores, lengths)

        # Compute marginals
        model = SemiMarkov(LogSemiring)
        edge_marginals = model.marginals(edge, lengths=lengths)

        # Aggregate to boundary probability
        # edge_marginals[b, n, k, c_dest, c_src] = P(segment ending at n+k with label c_dest)
        # Sum over k, c_dest, c_src to get P(any boundary at position n)
        boundary_probs = edge_marginals.sum(dim=(2, 3, 4))  # (batch, T-1)

        # Pad to length T (position 0 is always a boundary start)
        boundary_probs = torch.cat(
            [torch.ones(batch, 1, device=boundary_probs.device), boundary_probs], dim=1
        )

        if normalize:
            max_val = boundary_probs.max(dim=-1, keepdim=True)[0] + 1e-8
            boundary_probs = boundary_probs / max_val

        return boundary_probs

    def _build_edge_tensor(self, scores: Tensor, lengths: Tensor) -> Tensor:
        r"""Build edge tensor from scores (for exact marginals).

        Args:
            scores (Tensor): Projected scores of shape :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Edge potentials of shape :math:`(\text{batch}, T-1, K, C, C)`.

        .. warning::
            This is :math:`O(T \times K \times C^2)` memory and will OOM for large T!
        """
        batch, T, C = scores.shape
        K = self.max_duration

        # Cumulative scores for content computation
        cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float32, device=scores.device)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        # Build edge tensor
        edge = torch.full(
            (batch, T - 1, K, C, C), float("-inf"), dtype=torch.float32, device=scores.device
        )

        for n in range(T - 1):
            for k in range(1, max(min(K, T - n + 1), 2)):  # max ensures K=1 processes duration 1
                # Content score for segment [n, n+k)
                content = cum_scores[:, n + k, :] - cum_scores[:, n, :]  # (batch, C)

                # Use same indexing convention as streaming implementation
                dur_idx = min(k, K - 1)

                # Add duration bias
                segment_score = content + self.duration_bias[dur_idx]  # (batch, C)

                # Add transition (C_dest x C_src)
                # edge[n, k, c_dest, c_src] = segment_score[c_dest] + transition[c_src, c_dest]
                edge[:, n, dur_idx] = segment_score.unsqueeze(-1) + self.transition.T.unsqueeze(0)

        return edge

    def compute_position_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_position_marginals(hidden_states, lengths) -> Tensor

        Compute per-position label marginals :math:`P(\text{label}=c \text{ at position } t)`.

        Uses gradient of :math:`\log Z` w.r.t. projected scores to get per-position
        label distribution.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Label probabilities of shape :math:`(\text{batch}, T, C)`.
        """
        batch, T, _ = hidden_states.shape

        # Project to label space
        if self.projection is not None:
            with torch.no_grad():
                scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Use enable_grad() to ensure gradient computation works even if called
        # from within a no_grad() context
        with torch.enable_grad():
            # Make scores require grad
            scores_for_grad = scores.detach().requires_grad_(True)

            # Build cumulative scores
            cum_scores = torch.zeros(
                batch,
                T + 1,
                self.num_classes,
                dtype=torch.float32,
                device=scores_for_grad.device,
            )
            cum_scores[:, 1:] = torch.cumsum(scores_for_grad.float(), dim=1)

            # Forward pass
            partition = semi_crf_streaming_forward(
                cum_scores,
                self.transition.detach(),
                self.duration_bias.detach(),
                lengths,
                self.max_duration,
                semiring="log",
                use_triton=False,
            )

            # Backward to get per-position, per-label gradients
            partition.sum().backward()

        # Gradient magnitude per position per class
        grad = scores_for_grad.grad  # (batch, T, C)

        # Convert to probabilities via softmax over classes
        position_marginals = torch.softmax(grad.abs(), dim=-1)

        return position_marginals

    def compute_entropy_streaming(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_entropy_streaming(hidden_states, lengths) -> Tensor

        Approximate entropy from marginal distribution.

        For very long sequences where :class:`~torch_semimarkov.semirings.EntropySemiring`
        isn't available. Computes:

        .. math::
            H \approx -\sum_t P(\text{boundary}_t) \log P(\text{boundary}_t)

        as a proxy for segmentation uncertainty.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Entropy estimates of shape :math:`(\text{batch},)`.
        """
        # Get boundary marginals
        boundary_probs = self.compute_boundary_marginals(
            hidden_states, lengths, use_streaming=True, normalize=False
        )

        # Normalize to valid probability distribution
        boundary_probs = boundary_probs / (boundary_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute entropy: H = -sum(p * log(p))
        # Clamp to avoid log(0)
        log_probs = torch.log(boundary_probs.clamp(min=1e-10))
        entropy = -(boundary_probs * log_probs).sum(dim=-1)

        return entropy

    def compute_entropy_exact(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_entropy_exact(hidden_states, lengths) -> Tensor

        Compute exact entropy via :class:`~torch_semimarkov.semirings.EntropySemiring` (:math:`T < 10K` only).

        Computes the exact Shannon entropy of the segmentation distribution:

        .. math::
            H(P) = -\sum_y P(y) \log P(y)

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Exact entropy values of shape :math:`(\text{batch},)`.

        Raises:
            MemoryError: If T is too large and edge tensor doesn't fit in memory.

        .. warning::
            Requires materializing edge tensor - will OOM for :math:`T > 10K`!
        """
        from .semimarkov import SemiMarkov
        from .semirings import EntropySemiring

        batch, T, _ = hidden_states.shape

        # Project to label space
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build edge tensor (WARNING: O(T*K*C^2) memory!)
        edge = self._build_edge_tensor(scores, lengths)

        # Compute entropy using EntropySemiring
        model = SemiMarkov(EntropySemiring)
        entropy = model.sum(edge, lengths=lengths)

        return entropy

    def compute_loss_uncertainty_weighted(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        labels: Tensor,
        uncertainty_weight: float = 1.0,
        focus_mode: str = "high_uncertainty",
        use_triton: bool = False,
        reduction: str = "mean",
    ) -> Tensor:
        r"""compute_loss_uncertainty_weighted(hidden_states, lengths, labels, uncertainty_weight=1.0, focus_mode="high_uncertainty", use_triton=False, reduction="mean") -> Tensor

        Compute loss weighted by uncertainty for focused learning.

        Enables active learning by weighting the NLL loss by uncertainty:

        - ``"high_uncertainty"``: Higher weight on uncertain regions
        - ``"boundary_regions"``: Higher weight on boundary positions

        The weighted loss is:

        .. math::
            \mathcal{L}_{\text{weighted}} = (1 + \lambda \cdot u) \cdot \text{NLL}

        where :math:`\lambda` is the uncertainty weight and :math:`u` is the
        uncertainty measure.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
            uncertainty_weight (float, optional): Scaling factor :math:`\lambda` for
                uncertainty weighting. Default: ``1.0``
            focus_mode (str, optional): Uncertainty mode: ``"high_uncertainty"`` or
                ``"boundary_regions"``. Default: ``"high_uncertainty"``
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``False``
            reduction (str, optional): Reduction mode: ``"mean"``, ``"sum"``, or ``"none"``.
                Default: ``"mean"``

        Returns:
            Tensor: Weighted NLL loss.
        """
        # First pass: compute uncertainty (no gradients needed)
        with torch.no_grad():
            if focus_mode == "high_uncertainty":
                # Use entropy as uncertainty measure
                uncertainty = self.compute_entropy_streaming(hidden_states.detach(), lengths)
                # Normalize to [0, 1]
                uncertainty = uncertainty / (uncertainty.max() + 1e-8)
            elif focus_mode == "boundary_regions":
                # Weight by average boundary probability
                boundary_probs = self.compute_boundary_marginals(
                    hidden_states.detach(), lengths, use_streaming=True
                )
                uncertainty = boundary_probs.mean(dim=-1)  # (batch,)
            else:
                raise ValueError(f"Unknown focus_mode: {focus_mode}")

        # Second pass: compute weighted loss
        # Standard forward + loss
        result = self.forward(hidden_states, lengths, use_triton)
        partition = result["partition"]
        cum_scores = result["cum_scores"]

        # Score the gold segmentation
        gold_score = self._score_gold(cum_scores, labels, lengths)

        # NLL = partition - gold_score
        nll = partition - gold_score

        # Apply uncertainty weighting
        weight = 1.0 + uncertainty_weight * uncertainty
        weighted_nll = nll * weight

        if reduction == "mean":
            return weighted_nll.mean()
        elif reduction == "sum":
            return weighted_nll.sum()
        return weighted_nll


class UncertaintySemiMarkovCRFHead(UncertaintyMixin, nn.Module):
    r"""SemiMarkovCRFHead with uncertainty quantification methods.

    Combines the base :class:`~torch_semimarkov.nn.SemiMarkovCRFHead` functionality
    with :class:`UncertaintyMixin` methods for clinical applications requiring
    boundary confidence estimates.

    Args:
        num_classes (int): Number of label classes (C).
        max_duration (int): Maximum segment duration (K).
        hidden_dim (int, optional): If provided, adds a projection layer from
            ``hidden_dim`` to ``num_classes``. Default: ``None``
        init_scale (float, optional): Scale for parameter initialization.
            Default: ``0.1``
        duration_distribution (str, DurationDistribution, optional): Duration
            distribution to use. Can be:

            - ``None`` or ``"learned"``: Fully learned bias (default)
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

        >>> model = UncertaintySemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)
        >>> hidden = encoder(x)  # (batch, T, 512)
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Standard loss
        >>> loss = model.compute_loss(hidden, lengths, labels)
        >>>
        >>> # Uncertainty-weighted loss (for active learning)
        >>> loss = model.compute_loss_uncertainty_weighted(hidden, lengths, labels)
        >>>
        >>> # Boundary marginals (for clinical decision support)
        >>> boundary_probs = model.compute_boundary_marginals(hidden, lengths)
        >>>
        >>> # With geometric duration distribution
        >>> model = UncertaintySemiMarkovCRFHead(
        ...     num_classes=24, max_duration=100, hidden_dim=512,
        ...     duration_distribution="geometric"
        ... )

    See Also:
        :class:`~torch_semimarkov.nn.SemiMarkovCRFHead`: Base CRF head without uncertainty
        :class:`UncertaintyMixin`: Mixin providing uncertainty methods
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
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.max_duration = max_duration

        # CRF parameters
        self.transition = nn.Parameter(torch.randn(num_classes, num_classes) * init_scale)

        # Duration distribution (supports multiple parameterizations)
        if duration_distribution is None or duration_distribution == "learned":
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
            - **cum_scores** (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`.
        """
        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build cumulative scores for prefix-sum edge retrieval
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
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
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
        r"""Score the gold segmentation.

        Extracts segments from per-position labels and computes:

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
        """
        batch, T, _ = hidden_states.shape

        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float32, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

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
        """
        batch, T, _ = hidden_states.shape
        device = hidden_states.device

        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        cum_scores = torch.zeros(batch, T + 1, self.num_classes, dtype=torch.float32, device=device)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        max_scores = semi_crf_streaming_forward(
            cum_scores,
            self.transition,
            self.duration_bias,
            lengths,
            self.max_duration,
            semiring="max",
            use_triton=False,
        )

        all_segments: list[list[Segment]] = []

        for b in range(batch):
            seq_len = lengths[b].item()

            if seq_len > max_traceback_length or seq_len == 0:
                all_segments.append([])
                continue

            segments = self._traceback_single(cum_scores[b : b + 1], seq_len)
            all_segments.append(segments)

        return ViterbiResult(scores=max_scores, segments=all_segments)

    def _traceback_single(
        self,
        cum_scores: Tensor,
        seq_len: int,
    ) -> list[Segment]:
        """Traceback for a single sequence to recover optimal segmentation."""
        device = cum_scores.device
        C = self.num_classes
        K = self.max_duration

        alpha = torch.full((seq_len + 1, C), NEG_INF, device=device, dtype=torch.float32)
        alpha[0, :] = 0.0

        bp_k = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)
        bp_c = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)

        for t in range(1, seq_len + 1):
            k_eff = min(K - 1, t)

            for k in range(1, max(k_eff + 1, 2)):  # max ensures K=1 processes duration 1
                start = t - k

                edge_block = compute_edge_block_streaming(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    start,
                    k,
                )

                candidate_scores = alpha[start, :].unsqueeze(0) + edge_block[0]
                best_scores_from_src, best_c_src = candidate_scores.max(dim=-1)

                better_mask = best_scores_from_src > alpha[t, :]
                alpha[t, :] = torch.where(better_mask, best_scores_from_src, alpha[t, :])
                bp_k[t, :] = torch.where(better_mask, k, bp_k[t, :])
                bp_c[t, :] = torch.where(better_mask, best_c_src, bp_c[t, :])

        final_scores = alpha[seq_len, :]
        best_final_c = final_scores.argmax().item()

        segments: list[Segment] = []
        t = seq_len
        c = best_final_c

        while t > 0:
            k = bp_k[t, c].item()
            c_prev = bp_c[t, c].item()

            if k == 0:
                break

            start = t - k
            end = t - 1

            seg_score = self._compute_segment_score(
                cum_scores[0], start, end, c, c_prev if start > 0 else None
            )
            segments.append(Segment(start=start, end=end, label=c, score=seg_score))

            t = start
            c = c_prev

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
        content = (cum_scores[end + 1, label] - cum_scores[start, label]).item()

        duration = end - start + 1
        dur_idx = min(duration, self.max_duration) - 1
        dur_bias = self.duration_bias[dur_idx, label].item()

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
