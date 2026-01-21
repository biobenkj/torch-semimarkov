"""Uncertainty quantification for Semi-Markov CRF.

This module provides uncertainty quantification methods for the SemiMarkovCRFHead,
enabling clinical applications where boundary confidence is critical.

Two approaches are supported:
1. **Streaming-compatible (T >= 10K)**: Uses gradient-based marginals from backward pass
2. **Exact (T < 10K)**: Uses SemiMarkov.marginals() and EntropySemiring with pre-computed edges

Key insight: The gradient of log Z w.r.t. cumulative scores gives marginal information
via the forward-backward algorithm. This works with the streaming API for any sequence length.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .streaming import semi_crf_streaming_forward


class UncertaintyMixin:
    """Mixin providing uncertainty quantification for SemiMarkovCRFHead.

    Uses gradient-based marginals for streaming API compatibility (T >= 10K).
    Falls back to exact methods for short sequences when edges fit in memory.

    Methods:
        compute_boundary_marginals: P(boundary at position t) for each position
        compute_position_marginals: Per-position label distributions
        compute_entropy_streaming: Approximate entropy from marginal distribution
        compute_entropy_exact: Exact entropy via EntropySemiring (T < 10K only)
        compute_loss_uncertainty_weighted: Uncertainty-weighted NLL loss
    """

    def compute_boundary_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_streaming: bool = True,
        normalize: bool = True,
    ) -> Tensor:
        """Compute P(boundary at position t) for each position.

        For streaming mode (T >= 10K): Uses gradient of log Z w.r.t. cum_scores
        For exact mode (T < 10K): Uses SemiMarkov.marginals() if edges fit in memory

        The gradient of the log partition function w.r.t. the cumulative scores
        encodes information about which positions contribute to the partition,
        which correlates with boundary probability.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).
            use_streaming: If True, use streaming gradient-based method.
            normalize: If True, normalize to [0, 1] range.

        Returns:
            Tensor of shape (batch, T) with boundary probabilities.
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
        """Compute boundary marginals via streaming gradient method.

        Uses the fact that grad(log Z) / grad(cum_scores) encodes marginal
        information from the forward-backward algorithm.
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
        """Compute boundary marginals via exact edge marginals (T < 10K).

        Uses SemiMarkov.marginals() which requires materializing edge tensor.
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
        """Build edge tensor from scores (for exact marginals).

        WARNING: This is O(T*K*C^2) memory and will OOM for large T!
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
            for k in range(1, min(K, T - n)):
                # Content score for segment [n, n+k)
                content = cum_scores[:, n + k, :] - cum_scores[:, n, :]  # (batch, C)

                # Add duration bias
                segment_score = content + self.duration_bias[k - 1]  # (batch, C)

                # Add transition (C_dest x C_src)
                # edge[n, k, c_dest, c_src] = segment_score[c_dest] + transition[c_src, c_dest]
                edge[:, n, k - 1] = segment_score.unsqueeze(-1) + self.transition.T.unsqueeze(0)

        return edge

    def compute_position_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Compute per-position label marginals P(label=c at position t).

        Uses gradient of log Z w.r.t. projected scores to get per-position
        label distribution.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).

        Returns:
            Tensor of shape (batch, T, C) with label probabilities per position.
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
        """Approximate entropy from marginal distribution.

        For very long sequences where EntropySemiring isn't available.
        Computes H ≈ -sum(P(boundary_t) * log(P(boundary_t))) as a proxy
        for segmentation uncertainty.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).

        Returns:
            Tensor of shape (batch,) with entropy estimates.
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
        """Compute exact entropy via EntropySemiring (T < 10K only).

        Requires materializing edge tensor - will OOM for T > 10K!

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).

        Returns:
            Tensor of shape (batch,) with exact entropy values.

        Raises:
            MemoryError: If T is too large and edge tensor doesn't fit in memory.
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
        """Compute loss weighted by uncertainty for focused learning.

        Enables active learning by weighting the NLL loss by uncertainty:
        - "high_uncertainty": Higher weight on uncertain regions
        - "boundary_regions": Higher weight on boundary positions

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).
            labels: Per-position labels of shape (batch, T).
            uncertainty_weight: Scaling factor for uncertainty weighting.
            focus_mode: "high_uncertainty" or "boundary_regions"
            use_triton: Whether to use Triton kernels.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            Weighted NLL loss.
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
    """SemiMarkovCRFHead with uncertainty quantification methods.

    Combines the base SemiMarkovCRFHead functionality with UncertaintyMixin
    methods for clinical applications requiring boundary confidence estimates.

    Example::

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
    """

    def __init__(
        self,
        num_classes: int,
        max_duration: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.1,
    ):
        """Initialize UncertaintySemiMarkovCRFHead.

        Args:
            num_classes: Number of label classes (C).
            max_duration: Maximum segment duration (K).
            hidden_dim: If provided, adds a projection layer from hidden_dim to num_classes.
            init_scale: Scale for parameter initialization.
        """
        nn.Module.__init__(self)

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
            use_triton: Whether to use Triton kernels.

        Returns:
            Dictionary with 'partition' and 'cum_scores'.
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
        """Compute negative log-likelihood loss.

        Args:
            hidden_states: Encoder output of shape (batch, T, hidden_dim) or (batch, T, C).
            lengths: Sequence lengths of shape (batch,).
            labels: Per-position labels of shape (batch, T).
            use_triton: Whether to use Triton kernels.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            NLL loss.
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

        Extracts segments from per-position labels and computes:
            score = sum(content_scores) + sum(duration_biases) + sum(transitions)
        """
        batch = cum_scores.shape[0]
        device = cum_scores.device
        scores = torch.zeros(batch, device=device, dtype=cum_scores.dtype)

        for b in range(batch):
            seq_len = lengths[b].item()
            if seq_len == 0:
                continue

            seq_labels = labels[b, :seq_len]

            # Find segment boundaries
            changes = torch.where(seq_labels[:-1] != seq_labels[1:])[0]

            # Segment positions
            seg_ends = torch.cat([changes, torch.tensor([seq_len - 1], device=device)])
            seg_starts = torch.cat([torch.tensor([0], device=device), changes + 1])

            prev_label = None
            for i in range(len(seg_starts)):
                start = seg_starts[i].item()
                end = seg_ends[i].item()
                duration = end - start + 1
                label = seq_labels[start].item()

                # Content score
                content_score = cum_scores[b, end + 1, label] - cum_scores[b, start, label]
                scores[b] += content_score

                # Duration bias
                dur_idx = min(duration, self.max_duration) - 1
                if 0 <= dur_idx < self.max_duration:
                    scores[b] += self.duration_bias[dur_idx, label]

                # Transition score
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

        Returns the best score (max over all segmentations).
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

    def extra_repr(self) -> str:
        parts = [
            f"num_classes={self.num_classes}",
            f"max_duration={self.max_duration}",
        ]
        if self.projection is not None:
            parts.append(f"hidden_dim={self.projection.in_features}")
        return ", ".join(parts)
