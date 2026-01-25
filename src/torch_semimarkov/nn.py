r"""Neural network modules for Semi-Markov CRF.

This module provides :class:`torch.nn.Module` wrappers around the streaming Semi-CRF
kernels, making them easy to integrate with PyTorch Lightning and other training
frameworks.

Classes:
    :class:`SemiMarkovCRFHead`: CRF head for sequence labeling.

For clinical applications requiring boundary uncertainty or focused learning,
see :class:`~torch_semimarkov.uncertainty.UncertaintySemiMarkovCRFHead` which
extends this class with uncertainty methods.

Examples::

    >>> from torch_semimarkov import SemiMarkovCRFHead
    >>> model = SemiMarkovCRFHead(num_classes=5, max_duration=100, hidden_dim=64)
    >>> result = model.forward(hidden, lengths)

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
        The transition parameter has shape :math:`(C, C)` with indexing:

        - ``transition[i, j]`` = score for transitioning FROM label ``i`` TO label ``j``
        - Convention: ``(source, destination)`` ordering

        When building edge potentials internally, the matrix is transposed::

            edge[t, k, c_dest, c_src] = content[c_dest] + dur_bias[k, c_dest]
                                      + transition[c_src, c_dest]

        This matches the standard CRF transition convention used in the literature.
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
        r"""duration_bias() -> Tensor

        Duration bias tensor for segment scoring.

        Returns the duration bias of shape :math:`(K, C)` where ``K`` is the
        maximum duration and ``C`` is the number of classes. This property
        provides backward compatibility - internally calls ``self.duration_dist()``.

        Returns:
            Tensor: Duration bias tensor of shape :math:`(K, C)`.
        """
        return self.duration_dist()

    def _should_use_streaming(self, T: int) -> bool:
        r"""_should_use_streaming(T) -> bool

        Determine whether to use streaming backend based on memory heuristics.

        The streaming backend uses :math:`O(KC)` memory (ring buffer), while the
        exact backend requires :math:`O(TKC^2)` memory (edge tensor). This method
        returns ``True`` when the edge tensor would exceed the memory threshold.

        Args:
            T (int): Sequence length.

        Returns:
            bool: ``True`` if streaming backend should be used, ``False`` for exact.

        .. note::
            The default threshold is 8GB. Streaming is recommended when
            :math:`T \times K \times C^2 \times 4 > 8 \times 10^9` bytes.
        """
        K = self.max_duration
        C = self.num_classes

        # Edge tensor size in bytes: T * K * C * C * 4 (float32)
        edge_tensor_bytes = T * K * C * C * 4

        return edge_tensor_bytes > self.edge_memory_threshold

    def _select_backend(self, T: int, semiring: str, use_triton: bool) -> tuple[str, bool]:
        r"""_select_backend(T, semiring, use_triton) -> tuple[str, bool]

        Select backend based on memory heuristics and semiring requirements.

        The streaming backend supports only ``"log"`` and ``"max"`` semirings.
        Other semirings (e.g., :class:`~torch_semimarkov.semirings.EntropySemiring`)
        require the exact backend which materializes the full edge tensor.

        Args:
            T (int): Sequence length.
            semiring (str): Semiring name (``"log"``, ``"max"``, or others).
            use_triton (bool): Whether Triton acceleration is requested.

        Returns:
            tuple[str, bool]: A tuple containing:

            - **backend_type** (str): Either ``"streaming"`` or ``"exact"``.
            - **use_triton_final** (bool): Whether to use Triton (only for streaming).

        Raises:
            ValueError: If semiring requires exact backend but edge tensor exceeds
                memory threshold.
        """
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
        r"""_build_edge_tensor(scores, lengths) -> Tensor

        Build the full edge potential tensor for exact inference.

        Constructs edge potentials for all valid ``(position, duration, label_src, label_dst)``
        combinations. Each edge represents a segment starting at position ``n`` with
        duration ``k`` transitioning from ``c_src`` to ``c_dst``:

        .. math::
            \text{edge}[n, k, c_{\text{dst}}, c_{\text{src}}] =
            \text{content}[n, k, c_{\text{dst}}] + \text{duration\_bias}[k, c_{\text{dst}}]
            + \text{transition}[c_{\text{src}}, c_{\text{dst}}]

        Args:
            scores (Tensor): Projected scores of shape :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Edge potentials of shape :math:`(\text{batch}, T, K, C, C)`.

        .. warning::
            Memory complexity is :math:`O(T \times K \times C^2)`. For genome-scale
            sequences (:math:`T > 10K`), use the streaming backend instead.

        .. note::
            The edge tensor has ``T`` positions (not ``T-1``) to match the streaming
            API which can start segments at any position ``0`` to ``T-1``.
        """
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
        r"""_forward_exact(scores, lengths, semiring) -> Tensor

        Compute partition function via exact edge tensor inference.

        Uses :class:`~torch_semimarkov.SemiMarkov` with full semiring support.
        Materializes the complete edge tensor which enables arbitrary semirings
        but has :math:`O(TKC^2)` memory complexity.

        Args:
            scores (Tensor): Projected scores of shape :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            semiring (str): Semiring name (``"log"`` or ``"max"``).

        Returns:
            Tensor: Partition function values of shape :math:`(\text{batch},)`.

        .. warning::
            Requires materializing the full edge tensor. Will OOM for large ``T``.
            Use the streaming backend for genome-scale sequences.

        See Also:
            :class:`~torch_semimarkov.SemiMarkov`: Underlying inference engine
        """
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

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
    ) -> dict:
        r"""forward(hidden_states, lengths, use_triton=True, backend="auto") -> dict

        Compute partition function from encoder hidden states.

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
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'streaming', or 'exact'.")

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
        r"""compute_loss(hidden_states, lengths, labels, use_triton=True, backend="auto", reduction="mean") -> Tensor

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
        r"""parameter_penalty(p=2.0) -> Tensor

        Compute Lp penalty on CRF parameters to prevent gradient explosion.

        Semi-Markov CRFs have more complex gradient dynamics than standard CRFs
        due to: (1) edge potentials that scale with segment duration, (2) more
        learnable parameters (duration_bias, possibly duration-dependent transitions),
        and (3) partition functions with O(T × K × C²) terms vs O(T × C²).

        Without regularization, these parameters can drift to extreme values over
        many training epochs, causing numerical overflow in the backward pass
        marginal computation (exp of extreme log-marginals).

        This method returns a penalty term that should be added to the loss:

        .. math::
            \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \lambda \cdot \text{penalty}

        where :math:`\lambda` is a hyperparameter (typical values: 0.001 to 0.1).

        Args:
            p (float): The norm order. Default is 2.0 (L2 regularization).
                Use p=1.0 for L1 (sparse) regularization.

        Returns:
            Tensor: Scalar penalty value (sum of Lp norms of CRF parameters).

        Example:
            >>> crf = SemiMarkovCRFHead(num_classes=10, max_duration=30, hidden_dim=256)
            >>> loss = crf.compute_loss(hidden, lengths, labels)
            >>> total_loss = loss + 0.01 * crf.parameter_penalty()
            >>> total_loss.backward()
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
        backend: str = "auto",
    ) -> Tensor:
        r"""decode(hidden_states, lengths, use_triton=True, backend="auto") -> Tensor

        Decode best segmentation using Viterbi algorithm.

        Computes the maximum score over all valid segmentations using the
        max semiring (Viterbi decoding).

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``
            backend (str, optional): Backend selection mode: ``"auto"``, ``"streaming"``,
                or ``"exact"``. Default: ``"auto"``

        Returns:
            Tensor: Best score (max over all segmentations) of shape :math:`(\text{batch},)`.

        .. note::
            This returns the score, not the actual segmentation. For the full
            segmentation path, use the :class:`~torch_semimarkov.SemiMarkov` class
            with :class:`~torch_semimarkov.semirings.MaxSemiring` and extract via marginals.
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
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'streaming', or 'exact'.")

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
        r"""decode_with_traceback(hidden_states, lengths, max_traceback_length=10000, use_triton=True) -> ViterbiResult

        Decode best segmentation with full path reconstruction.

        Computes the maximum-scoring segmentation using Viterbi algorithm
        and returns both the score and the actual segment boundaries.

        Args:
            hidden_states (Tensor): Encoder output of shape
                :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            max_traceback_length (int, optional): Maximum sequence length for
                traceback. Sequences longer than this will have empty segment
                lists (score is still computed). Default: ``10000``
            use_triton (bool, optional): Whether to use Triton kernels for the
                forward pass. Default: ``True``

        Returns:
            ViterbiResult: Named tuple containing:

            - **scores** (Tensor): Best scores of shape :math:`(\text{batch},)`.
            - **segments** (List[List[Segment]]): Per-batch segment lists.

        .. note::
            For very long sequences (T > max_traceback_length), traceback requires
            :math:`O(T \times C)` memory which may not be feasible. In such cases, the
            returned segments list will be empty but scores are still computed.

        Examples::

            >>> result = crf.decode_with_traceback(hidden_states, lengths)
            >>> print(f"Score: {result.scores[0].item():.2f}")
            >>> for seg in result.segments[0]:
            ...     print(f"  [{seg.start}, {seg.end}] label={seg.label}")
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

        # Use fast backpointer-based traceback when possible
        if any_needs_traceback:
            # NOTE: Triton backpointer kernel disabled due to memory corruption bug
            # that causes NaN in subsequent training forward passes (see commit 256db2a).
            # Using PyTorch reference implementation until root cause is fixed.
            can_use_triton = False

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
        r"""_traceback_single(cum_scores, seq_len) -> list[Segment]

        Perform Viterbi traceback to recover the optimal segmentation path.

        Runs the forward pass with backpointer storage, then traces back from
        the final position to reconstruct the optimal segmentation.

        Args:
            cum_scores (Tensor): Cumulative scores of shape :math:`(1, T+1, C)`.
            seq_len (int): Actual sequence length.

        Returns:
            list[Segment]: List of :class:`Segment` objects forming the optimal
            segmentation, ordered from start to end.

        .. note::
            This method allocates :math:`O(T \times C)` memory for backpointers.
            For very long sequences, use :meth:`decode` which returns only the
            score without traceback.
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

    def _traceback_from_backpointers(
        self,
        bp_k: Tensor,
        bp_c: Tensor,
        final_labels: Tensor,
        lengths: Tensor,
        cum_scores: Tensor,
    ) -> list[list[Segment]]:
        r"""Fast traceback using pre-computed backpointers.

        This method performs O(T) traceback instead of O(T*K) recomputation
        by using backpointers computed during the forward pass.

        Args:
            bp_k (Tensor): Backpointer durations of shape (batch, T, C).
            bp_c (Tensor): Backpointer source labels of shape (batch, T, C).
            final_labels (Tensor): Best final label for each batch of shape (batch,).
            lengths (Tensor): Sequence lengths of shape (batch,).
            cum_scores (Tensor): Cumulative scores for segment score computation.

        Returns:
            list[list[Segment]]: Per-batch segment lists.
        """
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
        r"""_compute_segment_score(cum_scores, start, end, label, prev_label) -> float

        Compute the score contribution of a single segment.

        The segment score is the sum of content, duration bias, and transition:

        .. math::
            \text{score} = (\text{cum}[\text{end}+1, c] - \text{cum}[\text{start}, c])
            + \text{dur\_bias}[k, c] + \text{trans}[c_{\text{prev}}, c]

        Args:
            cum_scores (Tensor): Cumulative scores of shape :math:`(T+1, C)`.
            start (int): Segment start position (inclusive).
            end (int): Segment end position (inclusive).
            label (int): Segment label (class index).
            prev_label (int, optional): Previous segment's label for transition.
                ``None`` for the first segment.

        Returns:
            float: Total score contribution of this segment.
        """
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
