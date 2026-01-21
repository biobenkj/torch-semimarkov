import torch

from .helpers import _Struct


class SemiMarkov(_Struct):
    r"""Semi-Markov CRF inference for structured sequence prediction.

    Implements efficient forward algorithms for Semi-Markov Conditional Random
    Fields with explicit duration modeling. Uses streaming linear scan with
    O(KC) memory, enabling chromosome-scale sequences (T=400K+).

    Args:
        semiring: Semiring class defining the algebra for inference.
            Common choices: :class:`~torch_semimarkov.semirings.LogSemiring` (default),
            :class:`~torch_semimarkov.semirings.MaxSemiring` (Viterbi).

    .. note::
        For even more memory efficiency with large sequences, use the
        :func:`~torch_semimarkov.streaming.semi_crf_streaming_forward` API
        which computes edge potentials on-the-fly from cumulative scores.

    Input format:
        Edge potentials have shape :math:`(\text{batch}, N-1, K, C, C)` where:

        - :math:`N` is the sequence length
        - :math:`K` is the maximum segment duration
        - :math:`C` is the number of labels/classes
        - ``edge[b, n, k, c2, c1]`` is the log-potential for transitioning from
          label ``c1`` to label ``c2`` with duration ``k`` at position ``n``

    Examples::

        >>> import torch
        >>> from torch_semimarkov import SemiMarkov
        >>> from torch_semimarkov.semirings import LogSemiring
        >>> model = SemiMarkov(LogSemiring)
        >>> # batch=2, seq_len=100, max_dur=8, num_labels=4
        >>> edge = torch.randn(2, 99, 8, 4, 4)
        >>> lengths = torch.tensor([100, 100])
        >>> log_Z, potentials = model.logpartition(edge, lengths=lengths)
        >>> log_Z.shape
        torch.Size([2])

    See Also:
        :meth:`logpartition`: Compute log partition function
        :meth:`marginals`: Compute posterior marginals
        :func:`hsmm`: Convert HSMM parameters to edge potentials
        :func:`~torch_semimarkov.streaming.semi_crf_streaming_forward`:
            Memory-efficient API for very long sequences
    """

    def _check_potentials(self, edge, lengths=None):
        r"""Validate and convert edge potentials.

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``

        Returns:
            Tuple: ``(edge, batch, N, K, C, lengths)`` with validated dimensions.

        Raises:
            AssertionError: If shapes are inconsistent or lengths invalid.
        """
        batch, N_1, K, C, C2 = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        N = N_1 + 1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)
        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "At least one in batch must be length N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, K, C, lengths

    def logpartition(
        self,
        log_potentials,
        lengths=None,
        force_grad=False,
    ):
        r"""logpartition(log_potentials, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], None]

        Compute the log partition function using streaming linear scan.

        The partition function :math:`Z(x) = \sum_y \exp(\phi(x, y))` sums over
        all valid segmentations. This method returns :math:`\log Z(x)`.

        Memory: :math:`O(KC)` - independent of sequence length :math:`T`.
        Compute: :math:`O(T \times K \times C^2)`.

        Args:
            log_potentials (Tensor): Edge potentials of shape
                :math:`(\text{batch}, N-1, K, C, C)` in log-space.
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None`` (assumes all sequences have length N)
            force_grad (bool, optional): If ``True``, force gradient computation even
                when not needed. Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], None]: A tuple containing:

            - **log_Z** (Tensor): Log partition function of shape :math:`(\text{ssize}, \text{batch})`
              where ssize is the semiring size (typically 1 for LogSemiring)
            - **potentials** (List[Tensor]): List containing the input potentials for gradient computation
            - **beta** (None): Placeholder for backward compatibility (always ``None``)

        Examples::

            >>> model = SemiMarkov(LogSemiring)
            >>> edge = torch.randn(4, 99, 8, 6, 6)
            >>> lengths = torch.full((4,), 100)
            >>> log_Z, potentials, _ = model.logpartition(edge, lengths=lengths)
            >>> log_Z.shape
            torch.Size([1, 4])

        See Also:
            :func:`~torch_semimarkov.streaming.semi_crf_streaming_forward`:
                For very long sequences, use the streaming API which computes
                edges on-the-fly from cumulative scores.
        """
        return self._dp_scan_streaming(log_potentials, lengths, force_grad)

    def _dp_scan_streaming(self, edge, lengths=None, force_grad=False):
        r"""_dp_scan_streaming(edge, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], None]

        Streaming :math:`O(N)` scan with :math:`O(KC)` memory.

        Uses a ring buffer of the last :math:`K` beta values instead of storing all
        :math:`T` betas. Alpha values are computed inline and consumed immediately
        without storage.

        The recurrence computes:

        .. math::
            \beta[n, c] = \text{logsumexp}_{k=1}^{\min(K-1,n)} \sum_{c'} \left(
                \beta[n-k, c'] + \text{edge}[n-k, k, c, c']
            \right)

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None``
            force_grad (bool, optional): Force gradient computation even when not needed.
                Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], None]: A tuple containing:

            - **log_Z** (Tensor): Log partition of shape :math:`(\text{ssize}, \text{batch})`
            - **potentials** (List[Tensor]): Input potentials for gradient computation
            - **beta** (None): Placeholder (always ``None`` for streaming)

        .. note::
            Memory is :math:`O(KC)` for the ring buffer, independent of sequence length
            :math:`T`. This makes it universally applicable across all genomic parameter
            regimes where :math:`T` can exceed 400K.
        """
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Initialize beta[0] = one for all labels (paths of length 0 ending at position 0)
        beta0 = torch.zeros((ssize, batch, C), dtype=edge.dtype, device=edge.device)
        beta0 = semiring.fill(beta0, torch.tensor(True, device=edge.device), semiring.one)

        # Ring buffer with head pointer (no shifting needed)
        # beta_hist[:, :, (head - k) % K, :] holds beta from k steps ago
        ring_len = K
        beta_hist = torch.zeros((ssize, batch, ring_len, C), dtype=edge.dtype, device=edge.device)
        beta_hist = semiring.fill(beta_hist, torch.tensor(True, device=edge.device), semiring.zero)
        beta_hist[:, :, 0, :] = beta0
        head = 0  # beta[n-1] lives at beta_hist[:, :, head, :]

        # Store final beta for each batch item (at their respective lengths)
        final_beta = semiring.fill(
            torch.zeros_like(beta0),
            torch.tensor(True, device=edge.device),
            semiring.zero,
        )
        # Handle length=1 case
        mask_len1 = (lengths == 1).view(1, batch, 1)
        final_beta = torch.where(mask_len1, beta0, final_beta)

        # Pre-allocate duration indices (avoid re-creating each step)
        # max(K, 2) ensures K=1 still has duration 1 available
        dur_full = torch.arange(1, max(K, 2), device=edge.device)  # 1..max(K-1, 1)

        for n in range(1, N):
            # Number of valid durations at this position
            # k_eff = min(K-1, n) is the max duration index (durations are 1-indexed)
            k_eff = min(K - 1, n)

            # Duration indices: 1, 2, ..., k_eff (slice pre-allocated tensor)
            dur = dur_full[:k_eff]

            # Position indices where segments start: n-1, n-2, ..., n-k_eff
            start = n - dur

            # Get previous betas from ring buffer using modular indexing
            # beta[n-dur] is at index (head - (dur - 1)) % ring_len
            ring_idx = (head - (dur - 1)) % ring_len
            beta_prev = beta_hist.index_select(2, ring_idx)  # (ssize, batch, k_eff, C)

            # Get edge potentials for these (start, duration) pairs
            # edge shape: (ssize, batch, N-1, K, C, C)
            edge_slice = edge[:, :, start, dur, :, :]  # (ssize, batch, k_eff, C, C)

            # Compute: logsumexp over c_prev (last dim) of beta_prev + edge
            # beta_prev: (ssize, batch, k_eff, C) -> unsqueeze to (ssize, batch, k_eff, 1, C)
            # edge_slice: (ssize, batch, k_eff, C, C)
            # broadcast: (ssize, batch, k_eff, C, C)
            # sum over c_prev (dim=-1): (ssize, batch, k_eff, C)
            scores = semiring.sum(beta_prev.unsqueeze(-2) + edge_slice, dim=-1)

            # Sum over duration dimension to get beta[n]
            beta_n = semiring.sum(scores, dim=2)  # (ssize, batch, C)

            # Capture final beta for sequences ending at this position
            mask_end = (lengths == (n + 1)).view(1, batch, 1)
            final_beta = torch.where(mask_end, beta_n, final_beta)

            # Advance head pointer and write beta_n (no shifting!)
            head = (head + 1) % ring_len
            beta_hist[:, :, head, :] = beta_n

        # Final partition function: sum over labels
        v = semiring.sum(final_beta, dim=-1)  # (ssize, batch)
        return v, [edge], None

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        r"""to_parts(sequence, extra, lengths=None) -> Tensor

        Convert a sequence label representation to edge potentials.

        Args:
            sequence (Tensor): Label sequence of shape :math:`(\text{batch}, N)` with
                values in ``[-1, 0, ..., C-1]``. Value ``-1`` indicates continuation
                of the previous segment (no label boundary).
            extra (tuple): Tuple of ``(C, K)`` where C is number of labels and K is
                maximum duration.
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None``

        Returns:
            Tensor: Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)` where
            ``edge[b, n, k, c2, c1] = 1`` if there is a transition from label ``c1``
            to label ``c2`` with duration ``k`` at position ``n``.

        Examples::

            >>> # Sequence with labels [0, -1, 1, -1, -1, 2]
            >>> # means: label 0 (dur 2), label 1 (dur 3), label 2 (dur 1)
            >>> seq = torch.tensor([[0, -1, 1, -1, -1, 2]])
            >>> edge = SemiMarkov.to_parts(seq, (3, 4))
        """
        C, K = extra
        batch, N = sequence.shape
        labels = torch.zeros(batch, N - 1, K, C, C).long()
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        for b in range(batch):
            last = None
            c = None
            for n in range(0, N):
                if sequence[b, n] == -1:
                    assert n != 0
                    continue
                else:
                    new_c = sequence[b, n]
                    if n != 0:
                        labels[b, last, n - last, new_c, c] = 1
                    last = n
                    c = new_c
        return labels

    @staticmethod
    def from_parts(edge):
        r"""from_parts(edge) -> Tuple[Tensor, Tuple[int, int]]

        Convert edge potentials to a sequence label representation.

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
                Should contain binary indicators (0 or 1) marking segment boundaries.

        Returns:
            Tuple[Tensor, Tuple[int, int]]: A tuple containing:

            - **sequence** (Tensor): Label sequence of shape :math:`(\text{batch}, N)` with
              values in ``[-1, 0, ..., C-1]``. Value ``-1`` indicates continuation.
            - **extra** (tuple): Tuple of ``(C, K)`` for reconstructing edge shape.

        Examples::

            >>> edge = torch.zeros(1, 5, 4, 3, 3)
            >>> edge[0, 0, 2, 1, 0] = 1  # transition 0->1 at pos 0, dur 2
            >>> seq, (C, K) = SemiMarkov.from_parts(edge)
        """
        batch, N_1, K, C, _ = edge.shape
        N = N_1 + 1
        labels = torch.zeros(batch, N).long().fill_(-1)
        on = edge.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == 0:
                labels[on[i][0], on[i][1]] = on[i][4]
            labels[on[i][0], on[i][1] + on[i][2]] = on[i][3]
        return labels, (C, K)

    @staticmethod
    def hsmm(init_z_1, transition_z_to_z, transition_z_to_l, emission_n_l_z):
        r"""hsmm(init_z_1, transition_z_to_z, transition_z_to_l, emission_n_l_z) -> Tensor

        Convert Hidden Semi-Markov Model parameters to edge potentials.

        This adapter transforms standard HSMM parameterization (initial distribution,
        transition matrix, duration distribution, emissions) into the edge potential
        format expected by :meth:`logpartition`.

        Args:
            init_z_1 (Tensor): Initial state log-probabilities of shape :math:`(C,)` or
                :math:`(\text{batch}, C)`. Represents :math:`\log P(z_{-1}=i)` where
                :math:`z_{-1}` is an auxiliary state inducing the distribution over :math:`z_0`.
            transition_z_to_z (Tensor): State transition log-probabilities of shape
                :math:`(C, C)` where ``transition_z_to_z[i, j]`` = :math:`\log P(z_{n+1}=j | z_n=i)`.
            transition_z_to_l (Tensor): Duration log-probabilities of shape :math:`(C, K)`
                where ``transition_z_to_l[i, j]`` = :math:`\log P(l_n=j | z_n=i)`.
            emission_n_l_z (Tensor): Emission log-probabilities of shape
                :math:`(\text{batch}, N, K, C)` where ``emission[b, n, k, c]`` =
                :math:`\log P(x_{n:n+k} | z_n=c, l_n=k)`.

        Returns:
            Tensor: Edge potentials of shape :math:`(\text{batch}, N, K, C, C)` where:

            .. math::
                \text{edge}[b, n, k, c_2, c_1] = \log P(z_n=c_2 | z_{n-1}=c_1)
                + \log P(l_n=k | z_n=c_2) + \log P(x_{n:n+k} | z_n=c_2, l_n=k)

            with the initial state distribution added at position 0.

        Examples::

            >>> C, K, N, batch = 4, 8, 100, 2
            >>> init = torch.log_softmax(torch.randn(C), dim=-1)
            >>> trans_z = torch.log_softmax(torch.randn(C, C), dim=-1)
            >>> trans_l = torch.log_softmax(torch.randn(C, K), dim=-1)
            >>> emission = torch.randn(batch, N, K, C)
            >>> edge = SemiMarkov.hsmm(init, trans_z, trans_l, emission)
            >>> edge.shape
            torch.Size([2, 100, 8, 4, 4])
        """
        batch, N, K, C = emission_n_l_z.shape
        edges = torch.zeros(batch, N, K, C, C).type_as(emission_n_l_z)

        # initial state: log P(z_{-1})
        if init_z_1.dim() == 1:
            init_z_1 = init_z_1.unsqueeze(0).expand(batch, -1)
        edges[:, 0, :, :, :] += init_z_1.view(batch, 1, 1, C)

        # transitions: log P(z_n | z_{n-1})
        edges += transition_z_to_z.transpose(-1, -2).view(1, 1, 1, C, C)

        # l given z: log P(l_n | z_n)
        edges += transition_z_to_l.transpose(-1, -2).view(1, 1, K, C, 1)

        # emissions: log P(x_{n:n+l_n} | z_n, l_n)
        edges += emission_n_l_z.view(batch, N, K, C, 1)

        return edges
