import torch

from .blocktriangular import BlockTriangularMatrix, block_triang_matmul
from .helpers import _Struct


class SemiMarkov(_Struct):
    r"""Semi-Markov CRF inference for structured sequence prediction.

    Implements efficient forward algorithms for Semi-Markov Conditional Random
    Fields with explicit duration modeling. Supports multiple backends with
    different memory/speed tradeoffs.

    Args:
        semiring: Semiring class defining the algebra for inference.
            Common choices: :class:`~torch_semimarkov.semirings.LogSemiring` (default),
            :class:`~torch_semimarkov.semirings.MaxSemiring` (Viterbi).

    .. note::
        The default algorithm is streaming linear scan with O(KC) memory,
        which is universally applicable across all genomic parameter regimes.
        For 2-3x speedup when memory permits, use ``use_vectorized=True``.

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
    """

    def _check_potentials(self, edge, lengths=None):
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
        use_linear_scan=None,
        use_vectorized=False,
        use_banded=False,
        banded_perm: str = "auto",
        banded_bw_ratio: float = 0.6,
    ):
        r"""logpartition(log_potentials, lengths=None, force_grad=False, use_linear_scan=None, use_vectorized=False, use_banded=False) -> Tuple[Tensor, List[Tensor]]

        Compute the log partition function using the forward algorithm.

        The partition function :math:`Z(x) = \sum_y \exp(\phi(x, y))` sums over
        all valid segmentations. This method returns :math:`\log Z(x)`.

        Args:
            log_potentials (Tensor): Edge potentials of shape
                :math:`(\text{batch}, N-1, K, C, C)` in log-space.
            lengths (Tensor, optional): Sequence lengths of shape :math:`(\text{batch},)`.
                Default: ``None`` (assumes all sequences have length N).
            force_grad (bool, optional): If ``True``, force gradient computation even
                when not needed. Default: ``False``
            use_linear_scan (bool, optional): Algorithm selection:

                - ``None`` (default): Auto-select based on state space size
                  (KC > 200 triggers linear scan)
                - ``True``: Use O(N) linear scan with O(KC) memory
                - ``False``: Use O(log N) binary tree (warning: O((KC)³) memory per matmul)

            use_vectorized (bool, optional): If ``True``, use vectorized linear scan
                (2-3x faster but O(TKC) memory). Default: ``False`` (streaming with O(KC) memory)
            use_banded (bool, optional): If ``True``, attempt banded matrix path (prototype).
                Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor]]: A tuple containing:

            - **log_Z** (Tensor): Log partition function of shape :math:`(\text{batch},)`
            - **potentials** (List[Tensor]): List containing the input potentials for gradient computation

        .. warning::
            The binary tree algorithm (``use_linear_scan=False``) has O(log N) sequential
            depth but creates O((KC)³) temporary tensors per matrix multiply. For state
            spaces KC > 50-100, this typically causes OOM. Use linear scan for practical
            applications or :class:`~torch_semimarkov.semirings.CheckpointShardSemiring`
            to reduce peak memory.

        Examples::

            >>> model = SemiMarkov(LogSemiring)
            >>> edge = torch.randn(4, 99, 8, 6, 6)
            >>> lengths = torch.full((4,), 100)
            >>> # Default: streaming scan with O(KC) memory
            >>> log_Z, _ = model.logpartition(edge, lengths=lengths)
            >>> # Faster but more memory: vectorized scan
            >>> log_Z, _ = model.logpartition(edge, lengths=lengths, use_vectorized=True)
        """
        # Auto-select algorithm if not specified (before _check_potentials to get dimensions)
        if use_linear_scan is None:
            # Get dimensions to determine algorithm
            _, _, K, C, _ = self._get_dimension(log_potentials)
            K_C_product = K * C
            # Use linear scan for large state spaces (empirical threshold)
            use_linear_scan = K_C_product > 200

        # Dispatch to appropriate algorithm (each handles _check_potentials internally)
        if use_banded:
            # Banded path with optional permutation/bandwidth gating
            return self._dp_banded(
                log_potentials,
                lengths,
                force_grad,
                banded_perm=banded_perm,
                banded_bw_ratio=banded_bw_ratio,
            )
        if use_linear_scan:
            if use_vectorized:
                return self._dp_standard_vectorized(log_potentials, lengths, force_grad)
            else:
                return self._dp_scan_streaming(log_potentials, lengths, force_grad)

        # Binary tree algorithm: check and prepare potentials
        log_potentials, batch, N, K, C, lengths = self._check_potentials(log_potentials, lengths)

        # Binary tree algorithm - O((KC)^3) per matmul, use linear scan for large state spaces
        semiring = self.semiring
        ssize = semiring.size()
        log_potentials.requires_grad_(True)
        log_N, bin_N = self._bin_length(N - 1)
        init = self._chart((batch, bin_N, K - 1, K - 1, C, C), log_potentials, force_grad)

        # Init.
        mask = torch.zeros(*init.shape, device=log_potentials.device).bool()
        mask[:, :, :, 0, 0].diagonal(0, -2, -1).fill_(True)
        init = semiring.fill(init, mask, semiring.one)

        # Length mask
        big = torch.zeros(
            ssize,
            batch,
            bin_N,
            K,
            C,
            C,
            dtype=log_potentials.dtype,
            device=log_potentials.device,
        )
        big[:, :, : N - 1] = log_potentials
        c = init[:, :, :].view(ssize, batch * bin_N, K - 1, K - 1, C, C)
        lp = big[:, :, :].view(ssize, batch * bin_N, K, C, C)
        mask = torch.arange(bin_N).view(1, bin_N).expand(batch, bin_N)
        mask = mask.to(log_potentials.device)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * bin_N, 1, 1, 1).to(lp.device)
        lp.data[:] = semiring.fill(lp.data, mask, semiring.zero)
        c.data[:, :, :, 0] = semiring.fill(c.data[:, :, :, 0], (~mask), semiring.zero)
        c[:, :, : K - 1, 0] = semiring.sum(
            torch.stack([c.data[:, :, : K - 1, 0], lp[:, :, 1:K]], dim=-1)
        )
        mask = torch.zeros(*init.shape, device=log_potentials.device).bool()
        mask_length = torch.arange(bin_N).view(1, bin_N, 1).expand(batch, bin_N, C)
        mask_length = mask_length.to(log_potentials.device)
        for k in range(1, K - 1):
            mask_length_k = mask_length < (lengths - 1 - (k - 1)).view(batch, 1, 1)
            mask_length_k = semiring.convert(mask_length_k)
            mask[:, :, :, k - 1, k].diagonal(0, -2, -1).masked_fill_(mask_length_k, True)
        init = semiring.fill(init, mask, semiring.one)

        K_1 = K - 1

        # Order n, n-1
        chart = (
            init.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(-1, batch, bin_N, K_1 * C, K_1 * C)
        )

        for _level in range(1, log_N + 1):
            chart = semiring.matmul(chart[:, :, 1::2], chart[:, :, 0::2])

        final = chart.view(-1, batch, K_1, C, K_1, C)
        v = semiring.sum(semiring.sum(final[:, :, 0, :, 0, :].contiguous()))
        return v, [log_potentials]

    def _dp_standard(self, edge, lengths=None, force_grad=False):
        r"""_dp_standard(edge, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], List[Tensor]]

        Standard O(N) linear scan dynamic programming for Semi-Markov CRF.

        This is the reference implementation using list comprehensions. Kept for
        backward compatibility and correctness validation.

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``
            force_grad (bool, optional): Force gradient computation. Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: (log_Z, [edge], beta tables)

        .. note::
            For production use, prefer :meth:`_dp_scan_streaming` (O(KC) memory)
            or :meth:`_dp_standard_vectorized` (2-3x faster).
        """
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Init
        # All paths starting at N of len K
        alpha = self._make_chart(1, (batch, N, K, C), edge, force_grad)[0]

        # All paths finishing at N with label C
        beta = self._make_chart(N, (batch, C), edge, force_grad)
        beta[0] = semiring.fill(beta[0], torch.tensor(True).to(edge.device), semiring.one)

        # Main.
        for n in range(1, N):
            alpha[:, :, n - 1] = semiring.dot(
                beta[n - 1].view(ssize, batch, 1, 1, C),
                edge[:, :, n - 1].view(ssize, batch, K, C, C),
            )

            t = max(n - K, -1)
            f1 = torch.arange(n - 1, t, -1)
            f2 = torch.arange(1, len(f1) + 1)
            beta[n][:] = semiring.sum(
                torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2, strict=True)], dim=-1)
            )
        v = semiring.sum(
            torch.stack([beta[length_val - 1][:, i] for i, length_val in enumerate(lengths)], dim=1)
        )
        return v, [edge], beta

    def _dp_blocktriangular(self, edge, lengths=None, force_grad=False):
        r"""_dp_blocktriangular(edge, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], None]

        Binary-tree DP using block-triangular matrix multiplication.

        Exploits the duration constraint :math:`k_1 + k_2 \leq \text{span}` to sparsify
        the :math:`(K \cdot C, K \cdot C)` matrices via :class:`BlockTriangularMatrix`
        at each tree level.

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``
            force_grad (bool, optional): Force gradient computation. Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], None]: (log_Z, [edge], None)
        """
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        log_N, bin_N = self._bin_length(N - 1)
        init = self._chart((batch, bin_N, K - 1, K - 1, C, C), edge, force_grad)

        # Init mask (same as standard)
        mask = torch.zeros(*init.shape, device=edge.device).bool()
        mask[:, :, :, 0, 0].diagonal(0, -2, -1).fill_(True)
        init = semiring.fill(init, mask, semiring.one)

        # Length mask (same as standard)
        big = torch.zeros(
            ssize,
            batch,
            bin_N,
            K,
            C,
            C,
            dtype=edge.dtype,
            device=edge.device,
        )
        big[:, :, : N - 1] = edge
        c = init[:, :, :].view(ssize, batch * bin_N, K - 1, K - 1, C, C)
        lp = big[:, :, :].view(ssize, batch * bin_N, K, C, C)
        mask = torch.arange(bin_N).view(1, bin_N).expand(batch, bin_N)
        mask = mask.to(edge.device)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * bin_N, 1, 1, 1).to(lp.device)
        lp.data[:] = semiring.fill(lp.data, mask, semiring.zero)
        c.data[:, :, :, 0] = semiring.fill(c.data[:, :, :, 0], (~mask), semiring.zero)
        c[:, :, : K - 1, 0] = semiring.sum(
            torch.stack([c.data[:, :, : K - 1, 0], lp[:, :, 1:K]], dim=-1)
        )
        mask = torch.zeros(*init.shape, device=edge.device).bool()
        mask_length = torch.arange(bin_N).view(1, bin_N, 1).expand(batch, bin_N, C)
        mask_length = mask_length.to(edge.device)
        for k in range(1, K - 1):
            mask_length_k = mask_length < (lengths - 1 - (k - 1)).view(batch, 1, 1)
            mask_length_k = semiring.convert(mask_length_k)
            mask[:, :, :, k - 1, k].diagonal(0, -2, -1).masked_fill_(mask_length_k, True)
        init = semiring.fill(init, mask, semiring.one)

        K_1 = K - 1

        # Flatten to (K*C, K*C) - same permutation as standard algorithm
        chart = (
            init.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(-1, batch, bin_N, K_1 * C, K_1 * C)
        )

        for level in range(1, log_N + 1):
            span_length = 2 ** (level + 1)

            left = chart[:, :, 1::2]  # Odd indices
            right = chart[:, :, 0::2]  # Even indices

            result_parts = []
            for s in range(ssize):
                # Collapse batch and pair dims for BlockTriangularMatrix
                left_dense = left[s].reshape(-1, K_1 * C, K_1 * C)
                right_dense = right[s].reshape(-1, K_1 * C, K_1 * C)

                left_bt = BlockTriangularMatrix.from_dense(left_dense, K_1, C, span_length)
                right_bt = BlockTriangularMatrix.from_dense(right_dense, K_1, C, span_length)

                prod_bt = block_triang_matmul(left_bt, right_bt, semiring, span_length)
                prod_dense = prod_bt.to_dense().reshape(batch, -1, K_1 * C, K_1 * C)
                result_parts.append(prod_dense)

            chart = torch.stack(result_parts, dim=0)

        final = chart.view(-1, batch, K_1, C, K_1, C)
        v = semiring.sum(semiring.sum(final[:, :, 0, :, 0, :].contiguous()))
        return v, [edge], None

    def _dp_standard_vectorized(self, edge, lengths=None, force_grad=False):
        r"""_dp_standard_vectorized(edge, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], List[Tensor]]

        Vectorized O(N) linear scan with O(TKC) memory.

        Provides 2-3x speedup over :meth:`_dp_standard` through:

        1. Direct broadcasting instead of :meth:`semiring.dot` for alpha updates
        2. Advanced indexing instead of list comprehension for beta accumulation

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``
            force_grad (bool, optional): Force gradient computation. Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: (log_Z, [edge], beta tables)

        .. note::
            Memory usage is O(TKC) for full alpha/beta tables. For memory-constrained
            settings, use :meth:`_dp_scan_streaming` with O(KC) memory instead.
        """
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Init
        # All paths starting at N of len K
        alpha = self._make_chart(1, (batch, N, K, C), edge, force_grad)[0]

        # All paths finishing at N with label C
        beta = self._make_chart(N, (batch, C), edge, force_grad)
        beta[0] = semiring.fill(beta[0], torch.tensor(True).to(edge.device), semiring.one)

        # Main loop - vectorized operations
        for n in range(1, N):
            # Alpha update: Vectorized broadcast + logsumexp
            # alpha[n-1, k, c2] = logsumexp_c1(beta[n-1, c1] + edge[n-1, k, c2, c1])
            # Original: semiring.dot(beta[n-1].view(...), edge[n-1].view(...))
            # Optimized: Direct broadcasting with semiring.sum over last dim
            alpha[:, :, n - 1] = semiring.sum(
                beta[n - 1].view(ssize, batch, 1, 1, C) + edge[:, :, n - 1], dim=-1
            )

            # Beta accumulation: Vectorized advanced indexing
            # beta[n] = sum over k: alpha[n-k, k, :]
            # Original: torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2)])
            # Optimized: Use advanced indexing to gather all needed alpha values at once
            # Match the original loop bounds: t = max(n - K, -1); durations 1..len(f1)
            t = max(n - K, -1)
            time_indices = torch.arange(n - 1, t, -1, device=edge.device)
            dur_indices = torch.arange(1, time_indices.numel() + 1, device=edge.device)

            # Gather: alpha[:, :, time_indices[i], dur_indices[i], :]
            gathered = alpha[:, :, time_indices, dur_indices, :]  # (ssize, batch, k_eff, C)
            beta[n][:] = semiring.sum(gathered, dim=-2)  # Sum over k dimension

        # Final: Sum over sequence endpoints (keep original implementation)
        v = semiring.sum(
            torch.stack([beta[length_val - 1][:, i] for i, length_val in enumerate(lengths)], dim=1)
        )
        return v, [edge], beta

    def _dp_scan_streaming(self, edge, lengths=None, force_grad=False):
        r"""_dp_scan_streaming(edge, lengths=None, force_grad=False) -> Tuple[Tensor, List[Tensor], None]

        Streaming O(N) scan with O(KC) memory. **This is the default algorithm.**

        Uses a ring buffer of the last K beta values instead of storing all T betas.
        Alpha values are computed inline and consumed immediately without storage.

        .. math::
            \beta[n, c] = \text{logsumexp}_{k=1}^{\min(K-1,n)} \sum_{c'} \left(
                \beta[n-k, c'] + \text{edge}[n-k, k, c, c']
            \right)

        Args:
            edge (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``
            force_grad (bool, optional): Force gradient computation. Default: ``False``

        Returns:
            Tuple[Tensor, List[Tensor], None]: (log_Z, [edge], None)

        .. note::
            Memory is O(KC) for the ring buffer, independent of sequence length T.
            This makes it universally applicable across all genomic parameter regimes.
            Performance is within a few percent of the vectorized implementation.
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
        dur_full = torch.arange(1, K, device=edge.device)  # 1..K-1

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

    def _compute_bandwidth(self, span_length, K, C):
        """
        Compute bandwidth for Semi-Markov transitions over a span.

        At a binary tree level with span length S, the constraint that
        k1 + k2 <= S (duration pairs must fit within span) creates
        implicit sparsity in the (K*C, K*C) state space.

        Args:
            span_length: Length of the span being composed
            K: Maximum duration
            C: Number of labels

        Returns:
            (lu, ld): Upper and lower bandwidth
        """
        # For span S, valid duration pairs satisfy k1 + k2 <= S
        # In the worst case (all durations up to K valid), we get full density
        # In practice, shorter spans have narrower effective bandwidth

        # Tighter estimate: effective bandwidth grows roughly with half the span
        # (triangular constraint k1 + k2 <= span_length). This keeps banded
        # usable for mid-level tree nodes.
        effective_K = min(K - 1, span_length // 2)

        # State space is K*C, and we can transition between states
        # separated by at most K*C in the flattened indexing
        # The (k, c) -> (k', c') mapping creates bands
        lu = ld = effective_K * C

        return lu, ld

    def _build_adjacency(self, span_length: int, K: int, C: int, device) -> torch.Tensor:
        """
        Build a dense adjacency (K_1*C) x (K_1*C) for duration/label pairs
        under the constraint k1 + k2 <= span_length. This is used for bandwidth
        measurement and permutation selection.
        """
        K_1 = K - 1
        size = K_1 * C
        adj = torch.zeros((size, size), device=device, dtype=torch.bool)

        for k1 in range(K_1):
            max_k2 = min(K_1 - 1, span_length - k1)
            if max_k2 < 0:
                continue
            for k2 in range(max_k2 + 1):
                # connect all label pairs for these durations
                for c1 in range(C):
                    for c2 in range(C):
                        i = k1 * C + c1
                        j = k2 * C + c2
                        adj[i, j] = True
        return adj

    def _choose_banded_permutation(
        self,
        span_length: int,
        K: int,
        C: int,
        perm_mode: str,
        bw_ratio: float,
        device,
    ):
        """
        Decide whether to use banded matmul for a given span length and optionally
        return a permutation that reduces bandwidth.

        Returns:
            use_banded (bool), permutation (torch.Tensor or None), best_bw (int)
        """
        from .banded_utils import (
            apply_permutation,
            measure_effective_bandwidth,
            rcm_ordering_from_adjacency,
            snake_ordering,
        )

        K_1 = K - 1
        size = K_1 * C
        adj = self._build_adjacency(span_length, K, C, device=device)
        best_bw = measure_effective_bandwidth(adj.float(), fill_value=0.0)
        best_perm = None

        def maybe_update(perm):
            nonlocal best_bw, best_perm
            perm_bw = measure_effective_bandwidth(
                apply_permutation(adj.float(), perm), fill_value=0.0
            )
            if perm_bw < best_bw:
                best_bw = perm_bw
                best_perm = perm

        mode = perm_mode.lower() if isinstance(perm_mode, str) else "auto"
        if mode in ("auto", "snake"):
            perm = snake_ordering(K_1, C).to(device)
            maybe_update(perm)
        if mode in ("auto", "rcm"):
            perm_rcm, used = rcm_ordering_from_adjacency(adj.float().cpu())
            if used:
                maybe_update(perm_rcm.to(device))

        threshold = bw_ratio * size
        use_banded = best_bw < threshold
        return use_banded, best_perm, best_bw, threshold

    def _dp_banded(
        self, edge, lengths=None, force_grad=False, banded_perm="auto", banded_bw_ratio=0.6
    ):
        """
        PROTOTYPE AND NOT MEANT FOR PRODUCTION USE.
        Banded Semi-Markov binary tree forward pass.

        Uses BandedMatrix representations to exploit implicit sparsity from
        duration constraints. At each tree level, the constraint k1 + k2 <= span_length
        creates banded structure in the (K*C, K*C) state space.

        Memory improvement: O(N * K * C * bandwidth) vs O(N * (K*C)^2)
        where bandwidth << K*C for mid-level tree nodes.

        Uses CPU fallbacks (pure PyTorch) by default; automatically uses CUDA
        kernels when genbmm CUDA extension is compiled.

        Returns same signature as other DP methods: (v, [edge], beta)
        """
        # Use local BandedMatrix (has from_dense method)
        from .banded import BandedMatrix

        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Binary tree setup (same as standard algorithm)
        log_N, bin_N = self._bin_length(N - 1)
        init = self._chart((batch, bin_N, K - 1, K - 1, C, C), edge, force_grad)

        # Init mask (same as standard)
        mask = torch.zeros(*init.shape, device=edge.device).bool()
        mask[:, :, :, 0, 0].diagonal(0, -2, -1).fill_(True)
        init = semiring.fill(init, mask, semiring.one)

        # Length mask (same as standard)
        big = torch.zeros(
            ssize,
            batch,
            bin_N,
            K,
            C,
            C,
            dtype=edge.dtype,
            device=edge.device,
        )
        big[:, :, : N - 1] = edge
        c = init[:, :, :].view(ssize, batch * bin_N, K - 1, K - 1, C, C)
        lp = big[:, :, :].view(ssize, batch * bin_N, K, C, C)
        mask = torch.arange(bin_N).view(1, bin_N).expand(batch, bin_N)
        mask = mask.to(edge.device)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * bin_N, 1, 1, 1).to(lp.device)
        lp.data[:] = semiring.fill(lp.data, mask, semiring.zero)
        c.data[:, :, :, 0] = semiring.fill(c.data[:, :, :, 0], (~mask), semiring.zero)
        c[:, :, : K - 1, 0] = semiring.sum(
            torch.stack([c.data[:, :, : K - 1, 0], lp[:, :, 1:K]], dim=-1)
        )
        mask = torch.zeros(*init.shape, device=edge.device).bool()
        mask_length = torch.arange(bin_N).view(1, bin_N, 1).expand(batch, bin_N, C)
        mask_length = mask_length.to(edge.device)
        for k in range(1, K - 1):
            mask_length_k = mask_length < (lengths - 1 - (k - 1)).view(batch, 1, 1)
            mask_length_k = semiring.convert(mask_length_k)
            mask[:, :, :, k - 1, k].diagonal(0, -2, -1).masked_fill_(mask_length_k, True)
        init = semiring.fill(init, mask, semiring.one)

        K_1 = K - 1

        # Flatten to (K*C, K*C) - same permutation as standard algorithm
        chart = (
            init.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(-1, batch, bin_N, K_1 * C, K_1 * C)
        )

        # Binary tree with banded matmul
        for level in range(1, log_N + 1):
            # Span length at this level (doubles each iteration)
            # Level 1: span = 4, Level 2: span = 8, etc.
            span_length = 2 ** (level + 1)

            # Compute bandwidth for this level based on duration constraints
            lu, ld = self._compute_bandwidth(span_length, K, C)

            # Get left and right charts for composition
            left = chart[:, :, 1::2]  # Odd indices
            right = chart[:, :, 0::2]  # Even indices

            # Decide whether banded is worthwhile and optionally permute states
            use_banded, perm, bw, threshold = self._choose_banded_permutation(
                span_length, K, C, banded_perm, banded_bw_ratio, device=edge.device
            )
            matrix_size = K_1 * C
            if not use_banded or (lu + ld + 1) >= matrix_size:
                chart = semiring.matmul(left, right)
                continue

            if perm is not None:
                left = left[..., :, perm]
                left = left[..., perm, :]
                right = right[..., :, perm]
                right = right[..., perm, :]

            # Determine fill value based on semiring
            fill_value = semiring.zero if hasattr(semiring, "zero") else -1e9

            # Convert to banded representation for memory-efficient matmul
            # Shape: (ssize, batch, n_pairs, K*C, K*C) -> use genbmm.BandedMatrix
            ssize_val = left.shape[0]

            # Process each semiring dimension and batch
            result_parts = []
            for s in range(ssize_val):
                batch_parts = []
                for b in range(batch):
                    # Extract matrices for this semiring/batch: (n_pairs, K*C, K*C)
                    left_matrices = left[s, b]
                    right_matrices = right[s, b]

                    # Convert to BandedMatrix (CPU-only for now, CUDA via semirings later)
                    left_banded = BandedMatrix.from_dense(left_matrices, lu, ld, fill_value)
                    right_banded = BandedMatrix.from_dense(right_matrices, lu, ld, fill_value)

                    # Banded matmul using appropriate semiring operation
                    # Must use b.op(a.transpose()) to match semiring.matmul semantics
                    if hasattr(semiring, "zero") and semiring.zero == -1e9:
                        # Log semiring: use multiply_log (logsumexp)
                        result_banded = right_banded.multiply_log(left_banded.transpose())
                    elif hasattr(semiring, "one") and semiring.one == -1e9:
                        # Max semiring: use multiply_max
                        result_banded = right_banded.multiply_max(left_banded.transpose())
                    else:
                        # Standard semiring: use multiply (sum-product)
                        result_banded = right_banded.multiply(left_banded.transpose())

                    # Convert back to dense for next iteration
                    result_dense = result_banded.to_dense()
                    batch_parts.append(result_dense)

                result_parts.append(torch.stack(batch_parts, dim=0))

            chart = torch.stack(result_parts, dim=0)

        # Final extraction (same as standard)
        final = chart.view(-1, batch, K_1, C, K_1, C)
        v = semiring.sum(semiring.sum(final[:, :, 0, :, 0, :].contiguous()))

        # Return beta as None since binary tree doesn't compute it
        # This matches the signature of _dp_standard which returns (v, [edge], beta)
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
