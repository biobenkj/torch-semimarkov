"""Sampling strategies for benchmark configurations."""

from __future__ import annotations

# Canonical shape buckets for compile-friendly sampling
T_BUCKETS = [64, 128, 256, 512, 1024, 2048]
KC_BUCKETS = [12, 24, 48, 72, 96, 144, 192, 288]


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _evenly_spaced(values: list, count: int) -> list:
    """Select evenly spaced elements from a list."""
    if count <= 0:
        return []
    if count == 1:
        return [values[0]]
    n = len(values)
    idxs = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    sampled: list = []
    for idx in idxs:
        if idx not in seen:
            sampled.append(values[idx])
            seen.add(idx)
    if len(sampled) < count:
        for idx in range(n):
            if idx not in seen:
                sampled.append(values[idx])
                seen.add(idx)
                if len(sampled) == count:
                    break
    return sampled


def _choose_grid_counts(t_count: int, kc_count: int, max_points: int) -> tuple[int, int]:
    """Choose balanced T and KC sample counts for a given max_points budget."""
    best_t, best_kc = 1, 1
    best_prod = 1
    best_balance = float("inf")
    for t_idx in range(1, t_count + 1):
        kc_idx = min(kc_count, max_points // t_idx)
        if kc_idx < 1:
            continue
        prod = t_idx * kc_idx
        balance = abs((t_idx / t_count) - (kc_idx / kc_count))
        if prod > best_prod or (prod == best_prod and balance < best_balance):
            best_t, best_kc = t_idx, kc_idx
            best_prod = prod
            best_balance = balance
    return best_t, best_kc


def sample_configurations(
    T_list: list[int],
    K_list: list[int],
    C_list: list[int],
    B: int,
    max_points: int,
) -> list[tuple[int, int, int]]:
    """Sample (T, K, C) configs by T and K*C, anchored at min/max BTKC."""
    full_configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
    if max_points <= 0 or max_points >= len(full_configs):
        return full_configs

    t_values = sorted(set(T_list))
    kc_to_pairs: dict[int, list[tuple[int, int]]] = {}
    for K in K_list:
        for C in C_list:
            kc_to_pairs.setdefault(K * C, []).append((K, C))
    kc_values = sorted(kc_to_pairs.keys())

    t_count = len(t_values)
    kc_count = len(kc_values)
    t_sample_count, kc_sample_count = _choose_grid_counts(t_count, kc_count, max_points)

    t_samples = _evenly_spaced(t_values, t_sample_count)
    kc_samples = _evenly_spaced(kc_values, kc_sample_count)

    sampled_pairs: set[tuple[int, int]] = {(T, KC) for T in t_samples for KC in kc_samples}
    sampled_pairs.add((t_values[0], kc_values[0]))
    sampled_pairs.add((t_values[-1], kc_values[-1]))

    if len(sampled_pairs) < max_points:
        all_pairs = [(T, KC) for T in t_values for KC in kc_values]
        all_pairs.sort(key=lambda pair: B * pair[0] * pair[1])
        for pair in all_pairs:
            if len(sampled_pairs) >= max_points:
                break
            sampled_pairs.add(pair)

    t_index_map = {T: idx for idx, T in enumerate(t_values)}
    kc_index_map = {KC: idx for idx, KC in enumerate(kc_values)}

    configs: list[tuple[int, int, int]] = []
    for T, KC in sorted(sampled_pairs, key=lambda pair: (pair[0], pair[1])):
        pairs = kc_to_pairs[KC]
        pair_idx = (t_index_map[T] + kc_index_map[KC]) % len(pairs)
        K, C = pairs[pair_idx]
        configs.append((T, K, C))

    return configs


def bucket_to_canonical_shape(T: int, K: int, C: int) -> tuple[int, int, int]:
    """
    Round (T, K, C) to canonical shapes that maximize compiled kernel reuse.

    torch.compile generates specialized kernels per unique tensor shape.
    By bucketing to canonical shapes, we reduce compilation from O(configs)
    to O(buckets), typically 8-16 unique kernels instead of 50-100+.

    Returns:
        (T_canon, K_canon, C_canon) - the canonical shape to use for compilation
    """
    # Bucket T to nearest bucket >= T (or largest if T exceeds all)
    T_canon = T_BUCKETS[-1]
    for t_bucket in T_BUCKETS:
        if t_bucket >= T:
            T_canon = t_bucket
            break

    # Bucket K*C product
    KC = K * C
    KC_canon = KC_BUCKETS[-1]
    for kc_bucket in KC_BUCKETS:
        if kc_bucket >= KC:
            KC_canon = kc_bucket
            break

    # Find K, C factors of KC_canon that are closest to original ratio
    # Prefer keeping K close to original since it affects duration modeling
    best_K, best_C = K, C
    best_score = float("inf")
    for k in range(1, KC_canon + 1):
        if KC_canon % k == 0:
            c = KC_canon // k
            # Score: prefer K close to original, C close to original
            score = abs(k - K) + abs(c - C) * 0.5
            if score < best_score:
                best_K, best_C = k, c
                best_score = score

    return T_canon, best_K, best_C


def get_canonical_shapes(
    T_list: list[int], K_list: list[int], C_list: list[int]
) -> list[tuple[int, int, int]]:
    """Get the set of unique canonical shapes for a parameter grid."""
    seen: set[tuple[int, int, int]] = set()
    canonical: list[tuple[int, int, int]] = []

    for T in T_list:
        for K in K_list:
            for C in C_list:
                canon = bucket_to_canonical_shape(T, K, C)
                if canon not in seen:
                    seen.add(canon)
                    canonical.append(canon)

    # Sort by memory footprint (T * K * C)
    canonical.sort(key=lambda x: x[0] * x[1] * x[2])
    return canonical


def sample_compile_friendly(
    T_list: list[int],
    K_list: list[int],
    C_list: list[int],
    max_canonical_shapes: int = 8,
    samples_per_shape: int = 2,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], tuple[int, int, int]]]:
    """
    Sample configurations that minimize unique compiled shapes.

    This is a two-phase approach:
    1. Select a subset of canonical shapes (compile targets)
    2. Sample actual configs that map to those canonical shapes

    Args:
        T_list: Sequence lengths to consider
        K_list: Max durations to consider
        C_list: Label counts to consider
        max_canonical_shapes: Maximum unique shapes to compile
        samples_per_shape: Actual configs to benchmark per canonical shape

    Returns:
        (sampled_configs, config_to_canonical_map)
    """
    # Build all configs and group by canonical shape
    all_configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
    shape_groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for cfg in all_configs:
        canon = bucket_to_canonical_shape(*cfg)
        shape_groups.setdefault(canon, []).append(cfg)

    # Select canonical shapes with good coverage (evenly spaced by memory)
    all_canonical = sorted(shape_groups.keys(), key=lambda s: s[0] * s[1] * s[2])

    if len(all_canonical) <= max_canonical_shapes:
        selected_canonical = all_canonical
    else:
        # Evenly spaced selection
        selected_canonical = _evenly_spaced(all_canonical, max_canonical_shapes)

    # Sample actual configs from each selected canonical group
    sampled: list[tuple[int, int, int]] = []
    config_to_canon: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for canon in selected_canonical:
        group = shape_groups.get(canon, [])
        # Sort group by actual size and take evenly spaced samples
        group_sorted = sorted(group, key=lambda c: c[0] * c[1] * c[2])

        if len(group_sorted) <= samples_per_shape:
            selected = group_sorted
        else:
            selected = _evenly_spaced(group_sorted, samples_per_shape)

        for cfg in selected:
            sampled.append(cfg)
            config_to_canon[cfg] = canon

    return sampled, config_to_canon
