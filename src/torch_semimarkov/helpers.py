import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .semirings import LogSemiring


@dataclass
class Segment:
    """A segment in a semi-Markov segmentation.

    Attributes:
        start: Start position (inclusive).
        end: End position (inclusive).
        label: Segment label (class index).
        score: Segment score contribution.
    """

    start: int
    end: int
    label: int
    score: float = 0.0

    @property
    def duration(self) -> int:
        """Segment duration (number of positions)."""
        return self.end - self.start + 1


@dataclass
class ViterbiResult:
    """Result from Viterbi decoding with traceback.

    Attributes:
        scores: Best segmentation scores of shape (batch,).
        segments: Per-batch list of segments forming the optimal segmentation.
            Each inner list contains Segment objects in order from start to end.
    """

    scores: Tensor
    segments: list[list[Segment]]


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

    Examples::

        >>> from torch_semimarkov.semirings import LogSemiring
        >>> potentials = torch.randn(2, 10, 4, 4)
        >>> chart = Chart((2, 10, 4), potentials, LogSemiring)
        >>> chart.data.shape
        torch.Size([1, 2, 10, 4])
    """

    def __init__(self, size, potentials, semiring):
        c = torch.zeros(
            *((semiring.size(),) + size), dtype=potentials.dtype, device=potentials.device
        )
        c[:] = semiring.zero.view((semiring.size(),) + len(size) * (1,))

        self.data = c
        self.grad = self.data.detach().clone().fill_(0.0)

    def __getitem__(self, ind):
        r"""Index into chart data, preserving semiring and batch dimensions.

        The first two dimensions (semiring and batch) are automatically included.

        Args:
            ind (tuple): Index into remaining dimensions.

        Returns:
            Tensor: Sliced chart data.
        """
        slice_all = slice(None)
        return self.data[(slice_all, slice_all) + ind]

    def __setitem__(self, ind, new):
        r"""Set chart data at given index, preserving semiring and batch dimensions.

        Args:
            ind (tuple): Index into remaining dimensions.
            new (Tensor): Values to assign.
        """
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
        r"""score(potentials, parts, batch_dims=None) -> Tensor

        Compute the score of a specific structure under the model.

        Args:
            potentials (Tensor): Model potentials (structure-specific shape).
            parts (Tensor): Binary indicator of structure parts (same shape as potentials).
            batch_dims (list, optional): Dimensions to treat as batch. Default: ``[0]``

        Returns:
            Tensor: Score for each batch element.
        """
        if batch_dims is None:
            batch_dims = [0]
        score = torch.mul(potentials, parts)
        batch = tuple(score.shape[b] for b in batch_dims)
        return self.semiring.prod(score.view(batch + (-1,)))

    def _bin_length(self, length):
        r"""Compute binary tree parameters for a given sequence length.

        Args:
            length (int): Sequence length.

        Returns:
            Tuple[int, int]: ``(log_N, bin_N)`` where ``log_N`` is tree depth and
            ``bin_N`` is padded length (power of 2).
        """
        log_N = int(math.ceil(math.log(length, 2)))
        bin_N = int(math.pow(2, log_N))
        return log_N, bin_N

    def _get_dimension(self, edge):
        r"""Extract dimensions from edge potentials and enable gradients.

        Args:
            edge (Tensor or list): Edge potentials or list of tensors.

        Returns:
            tuple: Shape of the edge potentials.
        """
        if isinstance(edge, list):
            for t in edge:
                t.requires_grad_(True)
            return edge[0].shape
        else:
            edge.requires_grad_(True)
            return edge.shape

    def _chart(self, size, potentials, force_grad):
        r"""Allocate a single DP chart tensor.

        Args:
            size (tuple): Shape of the chart (excluding semiring dimension).
            potentials (Tensor): Reference tensor for dtype and device.
            force_grad (bool): Force gradient computation.

        Returns:
            Tensor: Initialized chart tensor.
        """
        return self._make_chart(1, size, potentials, force_grad)[0]

    def _make_chart(self, N, size, potentials, force_grad=False):
        r"""Allocate multiple DP chart tensors.

        Args:
            N (int): Number of charts to allocate.
            size (tuple): Shape of each chart (excluding semiring dimension).
            potentials (Tensor): Reference tensor for dtype and device.
            force_grad (bool, optional): Force gradient computation. Default: ``False``

        Returns:
            List[Tensor]: List of N initialized chart tensors.
        """
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
        r"""Convert structure representation to parts tensor.

        Base implementation returns input unchanged. Subclasses override
        to implement structure-specific conversion.

        Args:
            spans (Tensor): Structure representation.
            extra: Additional conversion parameters.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``

        Returns:
            Tensor: Parts tensor for scoring.
        """
        return spans

    @staticmethod
    def from_parts(spans):
        r"""Convert parts tensor to structure representation.

        Base implementation returns input unchanged. Subclasses override
        to implement structure-specific conversion.

        Args:
            spans (Tensor): Parts tensor.

        Returns:
            Tuple[Tensor, Any]: ``(structure, extra)`` where extra contains
            additional information needed for reconstruction.
        """
        return spans, None

    def _arrange_marginals(self, marg):
        r"""Arrange marginal gradients into output format.

        Args:
            marg (tuple): Tuple of gradient tensors from autograd.

        Returns:
            Tensor: Arranged marginal tensor.
        """
        return marg[0]


def score_gold_vectorized(
    cum_scores: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    max_duration: int,
) -> torch.Tensor:
    r"""Vectorized gold sequence scoring without Python loops.

    Computes the score of gold segmentations by extracting segments from
    per-position labels and summing content, duration, and transition scores.

    This is a vectorized replacement for the loop-based ``_score_gold`` method,
    providing significant speedup for batches with many segments.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)`.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        max_duration (int): Maximum segment duration (K).

    Returns:
        Tensor: Gold sequence scores of shape :math:`(\text{batch},)`.

    Note:
        Segments are identified by label changes. A segment at positions
        ``[start, end]`` (inclusive) with label ``c`` contributes:

        - Content: ``cum_scores[end+1, c] - cum_scores[start, c]``
        - Duration: ``duration_bias[min(duration, K-1), c]`` (duration_bias[k] stores bias for duration k)
        - Transition: ``transition[prev_label, c]`` (except first segment)
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    if T == 0:
        return torch.zeros(batch, device=device, dtype=dtype)

    # Handle single-position sequences
    if T == 1:
        # Single segment per batch: content + duration_bias[1, label] (duration=1 uses index 1)
        label_0 = labels[:, 0]  # (batch,)
        content = cum_scores[:, 1, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        content -= cum_scores[:, 0, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        dur_bias = duration_bias[1, label_0]  # duration=1 uses duration_bias[1]
        scores = content + dur_bias
        # Zero out scores for zero-length sequences
        scores = scores * (lengths > 0).to(dtype)
        return scores

    # Step 1: Detect segment boundaries (where label changes)
    # changes[b, t] = True if labels[b, t] != labels[b, t+1]
    changes = labels[:, :-1] != labels[:, 1:]  # (batch, T-1)

    # Create position indices
    positions = torch.arange(T - 1, device=device)  # (T-1,)

    # Step 2: Build segment representation
    # For each batch, segments end at change positions and at seq_len-1
    # We'll create padded tensors for seg_starts, seg_ends, seg_labels

    # Count segments per batch: num_changes + 1
    num_changes = changes.sum(dim=1)  # (batch,)

    # Create mask for valid positions within each sequence
    pos_mask = positions.unsqueeze(0) < (lengths.unsqueeze(1) - 1)  # (batch, T-1)
    changes_masked = changes & pos_mask  # Only count changes within valid positions

    # Recount with mask
    num_changes = changes_masked.sum(dim=1)  # (batch,)
    num_segments = num_changes + 1  # (batch,)
    max_segments = num_segments.max().item()

    if max_segments == 0:
        return torch.zeros(batch, device=device, dtype=dtype)

    # Create padded segment tensors
    seg_starts = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_ends = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_labels = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_mask = torch.zeros(batch, max_segments, device=device, dtype=torch.bool)

    # Fill segment tensors - this loop is over batch, not segments
    # For very large batches, this could be further optimized, but batch is typically small
    for b in range(batch):
        seq_len = lengths[b].item()
        if seq_len == 0:
            continue

        # Find change positions for this batch element
        if seq_len == 1:
            # Single position = single segment
            seg_starts[b, 0] = 0
            seg_ends[b, 0] = 0
            seg_labels[b, 0] = labels[b, 0]
            seg_mask[b, 0] = True
        else:
            change_positions = torch.where(changes_masked[b])[0]
            n_segs = change_positions.shape[0] + 1

            # Segment ends: change positions and seq_len-1
            ends = torch.cat([change_positions, torch.tensor([seq_len - 1], device=device)])
            # Segment starts: 0 and change positions + 1
            starts = torch.cat([torch.tensor([0], device=device), change_positions + 1])

            seg_starts[b, :n_segs] = starts
            seg_ends[b, :n_segs] = ends
            seg_labels[b, :n_segs] = labels[b, starts]
            seg_mask[b, :n_segs] = True

    # Step 3: Vectorized score computation

    # Content scores: cum_scores[b, end+1, label] - cum_scores[b, start, label]

    # Gather end positions (end + 1 for cumulative indexing)
    end_positions = seg_ends + 1  # (batch, max_segments)
    start_positions = seg_starts  # (batch, max_segments)

    # Create indices for gathering: need (batch, max_segments) indices into (batch, T+1, C)
    # We'll gather all C values then select the correct label

    # cum_scores shape: (batch, T+1, C)
    # We want cum_scores[b, end+1, :] and cum_scores[b, start, :]

    # Expand indices for gathering
    end_idx = end_positions.unsqueeze(-1).expand(-1, -1, C)  # (batch, max_segments, C)
    start_idx = start_positions.unsqueeze(-1).expand(-1, -1, C)  # (batch, max_segments, C)

    # Gather cumulative scores at segment boundaries
    cum_at_end = cum_scores.gather(1, end_idx)  # (batch, max_segments, C)
    cum_at_start = cum_scores.gather(1, start_idx)  # (batch, max_segments, C)

    # Compute content scores for all labels, then select correct label
    content_all = cum_at_end - cum_at_start  # (batch, max_segments, C)

    # Select content score for the segment's label
    seg_labels_expanded = seg_labels.unsqueeze(-1)  # (batch, max_segments, 1)
    content_scores = content_all.gather(2, seg_labels_expanded).squeeze(-1)  # (batch, max_segments)

    # Duration scores (duration_bias[k] stores bias for segments of duration k)
    durations = seg_ends - seg_starts + 1  # (batch, max_segments)
    dur_indices = durations.clamp(
        1, max_duration - 1
    )  # duration k uses index k, clamped to valid range

    # Gather duration bias: duration_bias[dur_idx, label]
    # duration_bias shape: (K, C), we need (batch, max_segments) values
    dur_bias_flat = duration_bias.view(-1)  # (K * C,)
    flat_indices = dur_indices * C + seg_labels  # (batch, max_segments)
    dur_scores = dur_bias_flat[flat_indices]  # (batch, max_segments)

    # Transition scores
    # Need transition[prev_label, curr_label] for segments 1, 2, ...
    # First segment has no transition
    prev_labels = torch.zeros_like(seg_labels)
    prev_labels[:, 1:] = seg_labels[:, :-1]  # Shift labels right

    # Gather transitions: transition[prev, curr]
    trans_flat = transition.view(-1)  # (C * C,)
    trans_indices = prev_labels * C + seg_labels  # (batch, max_segments)
    trans_scores = trans_flat[trans_indices]  # (batch, max_segments)

    # Zero out first segment's transition
    first_seg_mask = torch.zeros_like(seg_mask)
    first_seg_mask[:, 0] = True
    trans_scores = trans_scores * (~first_seg_mask).to(dtype)

    # Step 4: Sum with masking
    total_per_segment = content_scores + dur_scores + trans_scores  # (batch, max_segments)
    total_per_segment = total_per_segment * seg_mask.to(dtype)  # Zero out padded segments

    scores = total_per_segment.sum(dim=1)  # (batch,)

    return scores
