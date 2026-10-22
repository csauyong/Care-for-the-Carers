"""Small statistics helpers, dependency-free: bootstrap intervals and agreement measures."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean (e.g. of 0/1 pass results)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Interval for mean(b) - mean(a), resampling each group independently."""
    if not a or not b:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs = sorted(mean(rng.choices(b, k=len(b))) - mean(rng.choices(a, k=len(a))) for _ in range(n))
    return diffs[int(alpha / 2 * n)], diffs[int((1 - alpha / 2) * n) - 1]


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Chance-corrected agreement between two raters on categorical labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (1, 2, 3, 4, 5)) -> float:
    """Quadratic-weighted kappa for ordinal scores (1-5): near misses count as partial agreement."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    weight = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1
    ra = [sum(row) for row in observed]
    rb = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight[i][j] * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    return 1.0 if den == 0 else 1 - num / den


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation (average ranks for ties)."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = mean(ra), mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return float("nan") if sa == 0 or sb == 0 else cov / (sa * sb)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks
