import torch

from .blocktriangular import BlockTriangularMatrix, block_triang_matmul
from .helpers import _Struct


class SemiMarkov(_Struct):
    r"""Semi-Markov CRF for sequence segmentation with duration constraints.

    Computes partition functions and marginals over segmentations where each
    segment has a label and duration up to K. Multiple algorithm variants
    are available with different time/memory trade-offs.

    Args:
        semiring: Semiring for DP computation. Default: :class:`LogSemiring`

    Edge Potentials:
        Shape :math:`(\text{batch}, N-1, K, C, C)` where ``edge[b, n, k, c_dst, c_src]``
        is the score for a segment of duration ``k`` ending at position ``n``
        transitioning from ``c_src`` to ``c_dst``.
    """

    def _check_potentials(self, edge, lengths=None):
        """Validate edge potentials and return (edge, batch, N, K, C, lengths)."""
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
        r"""Compute log partition function via forward algorithm.

        Args:
            log_potentials (Tensor): Edge potentials of shape :math:`(\text{batch}, N-1, K, C, C)`.
            lengths (Tensor, optional): Sequence lengths. Default: ``None``
            force_grad (bool, optional): Force gradient computation. Default: ``False``
            use_linear_scan (bool, optional): Use O(N) linear scan. Auto-selected if ``None``.
            use_vectorized (bool, optional): Use vectorized scan (2-3x faster, O(TKC) memory).
            use_banded (bool, optional): Use banded matrix algorithm.
            banded_perm (str, optional): Permutation mode for banded: ``"auto"``, ``"snake"``, ``"rcm"``.
            banded_bw_ratio (float, optional): Bandwidth ratio threshold for banded.

        Returns:
            Tuple[Tensor, List[Tensor], None]: ``(log_Z, [potentials], None)``
        """
        # Auto-select algorithm if not specified
        if use_linear_scan is None:
            _, _, K, C, _ = self._get_dimension(log_potentials)
            K_C_product = K * C
            use_linear_scan = K_C_product > 200

        # Dispatch to appropriate algorithm
        if use_banded:
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

        # Binary tree algorithm
        return self._dp_binary_tree(log_potentials, lengths, force_grad)

    def _dp_binary_tree(self, log_potentials, lengths=None, force_grad=False):
        r"""Binary tree algorithm - O((KC)^3) per matmul.

        Use linear scan for large state spaces.
        """
        log_potentials, batch, N, K, C, lengths = self._check_potentials(log_potentials, lengths)
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
        return v, [log_potentials], None

    def _dp_scan_streaming(self, edge, lengths=None, force_grad=False):
        r"""O(N) streaming scan with O(KC) ring buffer memory.

        Memory-efficient linear scan using a ring buffer to store only
        the last K beta values instead of the full O(NK) table.
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

        # Pre-allocate duration values (avoid re-creating each step)
        # Durations range from 1 to K
        dur_full = torch.arange(1, K + 1, device=edge.device)  # 1..K

        for n in range(1, N):
            # Number of valid durations at this position: k = 1, 2, ..., min(K, n)
            k_eff = min(K, n)

            # Duration values: 1, 2, ..., k_eff (slice pre-allocated tensor)
            dur = dur_full[:k_eff]

            # Position indices where segments start: n-1, n-2, ..., n-k_eff
            start = n - dur

            # Get previous betas from ring buffer using modular indexing
            # beta[n-dur] is at index (head - (dur - 1)) % ring_len
            ring_idx = (head - (dur - 1)) % ring_len
            beta_prev = beta_hist.index_select(2, ring_idx)  # (ssize, batch, k_eff, C)

            # Get edge potentials for these (start, duration) pairs
            # edge shape: (ssize, batch, N-1, K, C, C)
            # Duration k uses edge index k-1
            dur_idx = dur - 1
            edge_slice = edge[:, :, start, dur_idx, :, :]  # (ssize, batch, k_eff, C, C)

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

    def _dp_standard(self, edge, lengths=None, force_grad=False):
        r"""Standard O(N) linear scan dynamic programming for Semi-Markov CRF.

        This is the reference implementation using list comprehensions. Kept for
        backward compatibility and correctness validation.

        Args:
            edge (Tensor): Edge potentials of shape (batch, N-1, K, C, C).
            lengths (Tensor, optional): Sequence lengths. Default: None
            force_grad (bool, optional): Force gradient computation. Default: False

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: (log_Z, [edge], beta tables)
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

            # Duration k (1 to K) uses index k-1 (0-based indexing)
            # Include K positions: t = n-K-1 so f1 has up to K elements
            t = max(n - K - 1, -1)
            f1 = torch.arange(n - 1, t, -1)  # time positions [n-1, n-2, ..., n-K]
            f2 = torch.arange(0, len(f1))  # duration indices [0, 1, ..., K-1]
            beta[n][:] = semiring.sum(
                torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2, strict=True)], dim=-1)
            )
        v = semiring.sum(
            torch.stack([beta[length_val - 1][:, i] for i, length_val in enumerate(lengths)], dim=1)
        )
        return v, [edge], beta

    def _dp_standard_vectorized(self, edge, lengths=None, force_grad=False):
        r"""Vectorized O(N) linear scan with O(TKC) memory.

        Provides 2-3x speedup over _dp_standard through:
        1. Direct broadcasting instead of semiring.dot for alpha updates
        2. Advanced indexing instead of list comprehension for beta accumulation

        Args:
            edge (Tensor): Edge potentials of shape (batch, N-1, K, C, C).
            lengths (Tensor, optional): Sequence lengths. Default: None
            force_grad (bool, optional): Force gradient computation. Default: False

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: (log_Z, [edge], beta tables)
        """
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Init
        alpha = self._make_chart(1, (batch, N, K, C), edge, force_grad)[0]
        beta = self._make_chart(N, (batch, C), edge, force_grad)
        beta[0] = semiring.fill(beta[0], torch.tensor(True).to(edge.device), semiring.one)

        # Main loop - vectorized operations
        for n in range(1, N):
            # Alpha update: Vectorized broadcast + logsumexp
            alpha[:, :, n - 1] = semiring.sum(
                beta[n - 1].view(ssize, batch, 1, 1, C) + edge[:, :, n - 1], dim=-1
            )

            # Beta accumulation: Vectorized advanced indexing
            # Duration k (1 to K) uses index k-1 (0-based indexing)
            # Include K positions: t = n-K-1 so time_indices has up to K elements
            t = max(n - K - 1, -1)
            time_indices = torch.arange(n - 1, t, -1, device=edge.device)
            dur_indices = torch.arange(0, time_indices.numel(), device=edge.device)

            # Gather: alpha[:, :, time_indices[i], dur_indices[i], :]
            gathered = alpha[:, :, time_indices, dur_indices, :]  # (ssize, batch, k_eff, C)
            beta[n][:] = semiring.sum(gathered, dim=-2)  # Sum over k dimension

        # Final: Sum over sequence endpoints
        v = semiring.sum(
            torch.stack([beta[length_val - 1][:, i] for i, length_val in enumerate(lengths)], dim=1)
        )
        return v, [edge], beta

    def _dp_blocktriangular(self, edge, lengths=None, force_grad=False):
        r"""Binary-tree DP using block-triangular matrix multiplication.

        Exploits the duration constraint k1 + k2 <= span to sparsify
        the (K*C, K*C) matrices via BlockTriangularMatrix at each tree level.

        Args:
            edge (Tensor): Edge potentials of shape (batch, N-1, K, C, C).
            lengths (Tensor, optional): Sequence lengths. Default: None
            force_grad (bool, optional): Force gradient computation. Default: False

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

    def _compute_bandwidth(self, span_length, K, C):
        """Compute bandwidth for Semi-Markov transitions over a span.

        At a binary tree level with span length S, the constraint that
        k1 + k2 <= S (duration pairs must fit within span) creates
        implicit sparsity in the (K*C, K*C) state space.
        """
        effective_K = min(K - 1, span_length // 2)
        lu = ld = effective_K * C
        return lu, ld

    def _build_adjacency(self, span_length: int, K: int, C: int, device) -> torch.Tensor:
        """Build a dense adjacency (K_1*C) x (K_1*C) for duration/label pairs
        under the constraint k1 + k2 <= span_length.
        """
        K_1 = K - 1
        size = K_1 * C
        adj = torch.zeros((size, size), device=device, dtype=torch.bool)

        for k1 in range(K_1):
            max_k2 = min(K_1 - 1, span_length - k1)
            if max_k2 < 0:
                continue
            for k2 in range(max_k2 + 1):
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
        """Decide whether to use banded matmul for a given span length and optionally
        return a permutation that reduces bandwidth.

        Returns:
            use_banded (bool), permutation (torch.Tensor or None), best_bw (int), threshold (float)
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
        """PROTOTYPE - Banded Semi-Markov binary tree forward pass.

        Uses BandedMatrix representations to exploit implicit sparsity from
        duration constraints. At each tree level, the constraint k1 + k2 <= span_length
        creates banded structure in the (K*C, K*C) state space.
        """
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

        # Flatten to (K*C, K*C)
        chart = (
            init.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(-1, batch, bin_N, K_1 * C, K_1 * C)
        )

        # Binary tree with banded matmul
        for level in range(1, log_N + 1):
            span_length = 2 ** (level + 1)
            lu, ld = self._compute_bandwidth(span_length, K, C)

            left = chart[:, :, 1::2]
            right = chart[:, :, 0::2]

            # Decide whether banded is worthwhile
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

            fill_value = semiring.zero if hasattr(semiring, "zero") else -1e9

            ssize_val = left.shape[0]
            result_parts = []
            for s in range(ssize_val):
                batch_parts = []
                for b in range(batch):
                    left_matrices = left[s, b]
                    right_matrices = right[s, b]

                    left_banded = BandedMatrix.from_dense(left_matrices, lu, ld, fill_value)
                    right_banded = BandedMatrix.from_dense(right_matrices, lu, ld, fill_value)

                    if hasattr(semiring, "zero") and semiring.zero == -1e9:
                        result_banded = right_banded.multiply_log(left_banded.transpose())
                    elif hasattr(semiring, "one") and semiring.one == -1e9:
                        result_banded = right_banded.multiply_max(left_banded.transpose())
                    else:
                        result_banded = right_banded.multiply(left_banded.transpose())

                    result_dense = result_banded.to_dense()
                    batch_parts.append(result_dense)

                result_parts.append(torch.stack(batch_parts, dim=0))

            chart = torch.stack(result_parts, dim=0)

        final = chart.view(-1, batch, K_1, C, K_1, C)
        v = semiring.sum(semiring.sum(final[:, :, 0, :, 0, :].contiguous()))
        return v, [edge], None

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """Convert label sequence (batch, N) with -1 continuations to edge potentials."""
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
        """Convert edge potentials to label sequence (batch, N) with -1 continuations."""
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
        """Convert HSMM params (init, trans, duration, emission) to edge potentials."""
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
