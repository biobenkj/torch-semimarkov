r"""Neural network modules for Semi-Markov CRF.

Provides :class:`torch.nn.Module` wrappers around streaming Semi-CRF kernels.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .duration import DurationDistribution, LearnedDuration, create_duration_distribution
from .helpers import Segment, ViterbiResult, score_gold_vectorized
from .streaming import (
    HAS_TRITON,
    semi_crf_streaming_forward,
    semi_crf_streaming_viterbi_with_backpointers,
)

# Conditionally import Triton viterbi for GPU acceleration
if HAS_TRITON:
    from .streaming import semi_crf_streaming_viterbi_triton
from .streaming.constants import NEG_INF
from .streaming.pytorch_reference import compute_edge_block_streaming
from .validation import (
    validate_device_consistency,
    validate_hidden_states,
    validate_labels,
    validate_lengths,
)

__all__ = [
    "SemiMarkovCRFHead",
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
        edge_memory_threshold (float, optional): Memory threshold in bytes for
            switching to streaming backend. Default: ``8e9`` (8GB)
        num_warps (int, optional): Number of warps per block for Triton kernels.
            Higher values increase parallelism but also register pressure.
            Recommended range: 2-8. Default: ``4``

    Attributes:
        transition (Parameter): Label transition scores of shape :math:`(C, C)`.
            Uses ``(source, destination)`` indexing: ``transition[i, j]`` is the
            score for transitioning FROM label ``i`` TO label ``j``.
        duration_dist (DurationDistribution): Duration distribution module.
        projection (Linear or None): Optional projection from encoder hidden dim.

    Properties:
        duration_bias: Returns the current duration bias tensor of shape :math:`(K, C)`.

    Transition Matrix Convention:
        ``transition[i, j]`` = score for transitioning FROM label ``i`` TO label ``j``.

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
        For T > 100K, use float32 precision for numerical stability.
    """

    def __init__(
        self,
        num_classes: int,
        max_duration: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.1,
        duration_distribution: Optional[Union[str, DurationDistribution]] = None,
        edge_memory_threshold: float = 8e9,
        num_warps: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_duration = max_duration
        self.edge_memory_threshold = edge_memory_threshold
        self.num_warps = num_warps

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
        """Duration bias tensor of shape :math:`(K, C)`."""
        return self.duration_dist()

    def _should_use_streaming(self, T: int) -> bool:
        """Return True if edge tensor would exceed memory threshold."""
        K = self.max_duration
        C = self.num_classes

        # Edge tensor size in bytes: T * K * C * C * 4 (float32)
        edge_tensor_bytes = T * K * C * C * 4

        return edge_tensor_bytes > self.edge_memory_threshold

    def _select_backend(self, T: int, semiring: str, use_triton: bool) -> tuple[str, bool]:
        """Select backend based on memory and semiring requirements."""
        # Semirings beyond log/max require exact backend
        if semiring not in ("log", "max"):
            if self._should_use_streaming(T):
                K, C = self.max_duration, self.num_classes
                raise ValueError(
                    f"Semiring '{semiring}' requires exact backend, but T={T}, K={K}, C={C} "
                    f"would require ~{T * K * C * C * 4 / 1e9:.1f}GB edge tensor. "
                    f"Use 'log' or 'max' semiring for streaming, or reduce T/K/C."
                )
            return "exact", False

        # Heuristic-based automatic selection
        if self._should_use_streaming(T):
            return "streaming", use_triton
        else:
            return "exact", False

    def _build_edge_tensor(self, scores: Tensor, lengths: Tensor) -> Tensor:
        """Build edge potentials of shape (batch, T, K, C, C). O(TKC²) memory."""
        batch, T, C = scores.shape
        K = self.max_duration

        # Cumulative scores for content computation
        # Zero-center before cumsum to match streaming preprocessing
        # Skip for T=1 since mean of single value zeros out content scores
        scores_float = scores.float()
        if T > 1:
            scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float32, device=scores.device)
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Build edge tensor with T positions (streaming can access positions 0 to T-1)
        edge = torch.full(
            (batch, T, K, C, C), float("-inf"), dtype=torch.float32, device=scores.device
        )

        for n in range(T):
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

    def _forward_exact(self, scores: Tensor, lengths: Tensor, semiring: str) -> Tensor:
        """Compute partition via exact edge tensor. O(TKC²) memory."""
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring, MaxSemiring

        SEMIRING_MAP = {"log": LogSemiring, "max": MaxSemiring}
        semiring_cls = SEMIRING_MAP[semiring]

        # Build edge tensor (O(T*K*C^2) memory)
        # Edge tensor has shape (batch, T, K, C, C) to match streaming API
        edge = self._build_edge_tensor(scores, lengths)

        # SemiMarkov interprets edge shape (batch, N-1, K, C, C) as sequence of length N
        # Since our edge has T positions, SemiMarkov sees N = T + 1
        # We need to pass lengths + 1 to match
        model = SemiMarkov(semiring_cls)
        result = model.logpartition(edge, lengths=lengths + 1, use_linear_scan=True)
        return result[0].squeeze(0)

    def _forward_binary_tree_sharded(
        self, scores: Tensor, lengths: Tensor, semiring: str
    ) -> Tensor:
        """Compute partition via sharded binary tree. Memory-efficient reference implementation.

        Uses CheckpointShardSemiring to reduce peak memory by splitting large matmuls
        into smaller shards that are processed sequentially with gradient checkpointing.
        This is slower than streaming but provides a reference implementation for validation.
        """
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring, MaxSemiring
        from .semirings.checkpoint import CheckpointShardSemiring

        SEMIRING_MAP = {"log": LogSemiring, "max": MaxSemiring}
        base_semiring = SEMIRING_MAP[semiring]

        # Wrap semiring with sharded checkpointing for memory efficiency
        ShardedSemiring = CheckpointShardSemiring(base_semiring, max_size=10000)

        # Build edge tensor (still O(T*K*C^2) memory, but matmuls are sharded)
        edge = self._build_edge_tensor(scores, lengths)

        # Use binary tree algorithm with sharded semiring
        model = SemiMarkov(ShardedSemiring)
        result = model._dp_binary_tree(edge, lengths=lengths + 1, force_grad=True)
        return result[0].squeeze(0)

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
    ) -> dict:
        r"""Compute partition function from encoder hidden states.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                if projection is enabled, or :math:`(\text{batch}, T, C)` if projection is ``None``.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``
            backend (str, optional): Backend selection mode:

                - ``"auto"``: Select based on memory heuristic (default)
                - ``"streaming"``: Force streaming backend (genome-scale)
                - ``"exact"``: Force exact backend via ``semimarkov.py``

        Returns:
            dict: Dictionary containing:

            - **partition** (Tensor): Log partition function of shape :math:`(\text{batch},)`.
            - **cum_scores** (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`
              for loss computation.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)  # (batch, T, C)
        else:
            scores = hidden_states

        # Debug check for gradient corruption after projection
        # This catches NaN propagation from corrupted model parameters early
        if scores.requires_grad and torch.isnan(scores).any():
            raise ValueError(
                "NaN detected in projected scores. This typically indicates gradient "
                "corruption from a previous backward pass. Check if model parameters "
                "contain NaN values (e.g., via torch.isnan(param).any() for each param)."
            )

        # Select backend
        if backend == "auto":
            backend_type, use_triton_final = self._select_backend(T, "log", use_triton)
        elif backend == "streaming":
            backend_type, use_triton_final = "streaming", use_triton
        elif backend == "exact":
            backend_type, use_triton_final = "exact", False
        elif backend == "binary_tree_sharded":
            backend_type, use_triton_final = "binary_tree_sharded", False
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'auto', 'streaming', 'exact', or 'binary_tree_sharded'."
            )

        # Build cumulative scores for prefix-sum edge retrieval
        # CRITICAL: Use float32 for numerical stability at T > 100K
        # Zero-center before cumsum to prevent magnitude drift at long sequences
        # Skip for T=1 since mean of single value zeros out content scores
        scores_float = scores.float()
        if T > 1:
            scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float32, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Debug check for NaN in cumsum (can happen if scores have extreme values)
        if cum_scores.requires_grad and torch.isnan(cum_scores).any():
            raise ValueError(
                "NaN detected in cumulative scores after cumsum. This typically indicates "
                "extreme values in the input scores or projection layer weights."
            )

        if backend_type == "streaming":
            # Compute partition function via streaming algorithm
            partition = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="log",
                use_triton=use_triton_final,
                num_warps=self.num_warps,
            )
        elif backend_type == "binary_tree_sharded":
            # Use sharded binary tree backend for memory-efficient reference implementation
            partition = self._forward_binary_tree_sharded(scores, lengths, "log")
        else:
            # Use exact backend via semimarkov.py
            partition = self._forward_exact(scores, lengths, "log")

        return {"partition": partition, "cum_scores": cum_scores}

    def compute_loss(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        labels: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
        reduction: str = "mean",
    ) -> Tensor:
        r"""Compute negative log-likelihood loss.

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
            backend (str, optional): Backend selection mode: ``"auto"``, ``"streaming"``,
                or ``"exact"``. Default: ``"auto"``
            reduction (str, optional): Reduction mode: ``"mean"``, ``"sum"``, or ``"none"``.
                Default: ``"mean"``

        Returns:
            Tensor: NLL loss. Scalar if reduction is ``"mean"`` or ``"sum"``,
            shape :math:`(\text{batch},)` if ``"none"``.
        """
        # Validate labels (hidden_states and lengths validated in forward())
        validate_labels(
            labels,
            self.num_classes,
            batch_size=hidden_states.shape[0],
            seq_length=hidden_states.shape[1],
        )

        result = self.forward(hidden_states, lengths, use_triton, backend)
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

    def parameter_penalty(self, p: float = 2.0) -> Tensor:
        r"""Compute Lp penalty on CRF parameters.

        Args:
            p (float, optional): Norm order (2.0 for L2, 1.0 for L1). Default: ``2.0``

        Returns:
            Tensor: Scalar penalty :math:`\|W_{\text{trans}}\|_p^p + \|W_{\text{dur}}\|_p^p`.
        """
        penalty = self.transition.norm(p=p).pow(p)
        penalty = penalty + self.duration_bias.norm(p=p).pow(p)
        return penalty

    def _score_gold(
        self,
        cum_scores: Tensor,
        labels: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Score the gold segmentation. Returns shape (batch,)."""
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
        backend: str = "auto",
    ) -> Tensor:
        r"""Compute Viterbi (max) score without path reconstruction.

        Args:
            hidden_states (Tensor): Shape :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Use Triton kernels. Default: ``True``
            backend (str, optional): ``"auto"``, ``"streaming"``, or ``"exact"``. Default: ``"auto"``

        Returns:
            Tensor: Best segmentation scores of shape :math:`(\text{batch},)`.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Select backend
        if backend == "auto":
            backend_type, use_triton_final = self._select_backend(T, "max", use_triton)
        elif backend == "streaming":
            backend_type, use_triton_final = "streaming", use_triton
        elif backend == "exact":
            backend_type, use_triton_final = "exact", False
        elif backend == "binary_tree_sharded":
            backend_type, use_triton_final = "binary_tree_sharded", False
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'auto', 'streaming', 'exact', or 'binary_tree_sharded'."
            )

        # Build cumulative scores
        # Zero-center before cumsum to prevent magnitude drift at long sequences
        # Skip for T=1 since mean of single value zeros out content scores
        scores_float = scores.float()
        if T > 1:
            scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float32, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        if backend_type == "streaming":
            # Use max semiring for Viterbi
            max_score = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="max",
                use_triton=use_triton_final,
                num_warps=self.num_warps,
            )
        elif backend_type == "binary_tree_sharded":
            # Use sharded binary tree backend for memory-efficient reference implementation
            max_score = self._forward_binary_tree_sharded(scores, lengths, "max")
        else:
            # Use exact backend via semimarkov.py
            max_score = self._forward_exact(scores, lengths, "max")

        return max_score

    def decode_with_traceback(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        max_traceback_length: int = 10000,
        use_triton: bool = True,
    ) -> ViterbiResult:
        r"""Viterbi decode with path reconstruction.

        Args:
            hidden_states (Tensor): Shape :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Shape :math:`(\text{batch},)`.
            max_traceback_length (int, optional): Maximum T for traceback. Longer
                sequences return empty segment lists. Default: ``10000``
            use_triton (bool, optional): Use Triton kernels for forward pass. Default: ``True``

        Returns:
            ViterbiResult: Named tuple with ``scores`` :math:`(\text{batch},)` and
            ``segments`` (list of list of :class:`Segment`).
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape
        device = hidden_states.device

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build cumulative scores
        # Zero-center before cumsum to prevent magnitude drift at long sequences
        # Skip for T=1 since mean of single value zeros out content scores
        scores_float = scores.float()
        if T > 1:
            scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, self.num_classes, dtype=torch.float32, device=device)
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Check which sequences need traceback
        needs_traceback = lengths <= max_traceback_length
        any_needs_traceback = needs_traceback.any().item()

        if any_needs_traceback:
            can_use_triton = False  # Triton backpointer disabled (memory corruption)

            # Get max scores AND backpointers in a single forward pass
            if can_use_triton:
                max_scores, bp_k, bp_c, final_labels = semi_crf_streaming_viterbi_triton(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    lengths,
                    self.max_duration,
                )
            else:
                max_scores, bp_k, bp_c, final_labels = semi_crf_streaming_viterbi_with_backpointers(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    lengths,
                    self.max_duration,
                )

            # Fast O(T) traceback using backpointers
            all_segments = self._traceback_from_backpointers(
                bp_k, bp_c, final_labels, lengths, cum_scores
            )

            # Clear segments for sequences that exceeded max_traceback_length
            for b in range(batch):
                if lengths[b].item() > max_traceback_length:
                    all_segments[b] = []
        else:
            # No sequences need traceback - just compute scores
            max_scores = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="max",
                use_triton=use_triton,
                num_warps=self.num_warps,
            )
            all_segments = [[] for _ in range(batch)]

        return ViterbiResult(scores=max_scores, segments=all_segments)

    def _traceback_single(
        self,
        cum_scores: Tensor,
        seq_len: int,
    ) -> list[Segment]:
        """Viterbi traceback for single sequence. O(TC) memory."""
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

    def _traceback_from_backpointers(
        self,
        bp_k: Tensor,
        bp_c: Tensor,
        final_labels: Tensor,
        lengths: Tensor,
        cum_scores: Tensor,
    ) -> list[list[Segment]]:
        """O(T) traceback using pre-computed backpointers."""
        batch = lengths.shape[0]
        all_segments: list[list[Segment]] = []

        for b in range(batch):
            seq_len = lengths[b].item()
            segments: list[Segment] = []

            if seq_len == 0:
                all_segments.append(segments)
                continue

            t = seq_len
            c = final_labels[b].item()

            while t > 0:
                # bp_k and bp_c are 0-indexed: position t corresponds to index t-1
                k = bp_k[b, t - 1, c].item()
                c_prev = bp_c[b, t - 1, c].item()

                if k == 0:
                    # Safety check - shouldn't happen with proper forward pass
                    break

                start = t - k
                end = t - 1  # Segment is [start, end] inclusive

                # Compute segment score
                # Always include transition, even for first segment - the forward pass
                # computes: max_{c_src} [alpha[0, c_src] + segment_score + transition[c_src, c_dest]]
                # where alpha[0, c_src] = 0, so transition from c_prev is part of the score
                seg_score = self._compute_segment_score(cum_scores[b], start, end, c, c_prev)

                segments.append(Segment(start=start, end=end, label=c, score=seg_score))

                t = start
                c = c_prev

            # Reverse to get segments in order
            segments.reverse()
            all_segments.append(segments)

        return all_segments

    def _compute_segment_score(
        self,
        cum_scores: Tensor,
        start: int,
        end: int,
        label: int,
        prev_label: Optional[int],
    ) -> float:
        """Compute content + duration_bias + transition for a segment."""
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
