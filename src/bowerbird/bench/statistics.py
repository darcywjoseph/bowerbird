"""Compares the success rates of two benchmark conditions.

A condition's result is a grid of attempts: one row per training seed, one
column per starting layout. Two things vary by chance, which model a seed
gives and which layouts are tested, so the bootstrap resamples both.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000


@dataclass(frozen=True)
class Difference:
    """How much more often one condition succeeds than another.

    Attributes:
        difference: The first condition's success rate minus the second's.
        low: The lower end of the 95% interval.
        high: The upper end of the 95% interval.
        p_value: The two-sided chance of a difference at least this far from
          zero if the conditions were equally good.
    """

    difference: float
    low: float
    high: float
    p_value: float


def compare(
    first: Sequence[Sequence[bool]],
    second: Sequence[Sequence[bool]],
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> Difference:
    """Estimates the difference between two conditions' success rates.

    Seeds are resampled separately for each condition. Layouts are resampled
    once and shared, because both conditions are tested on the same layouts.

    Args:
        first: One condition's attempts, a row per training seed and a column
          per starting layout.
        second: The other condition's attempts, on the same layouts.
        resamples: How many bootstrap resamples to draw.
        seed: Fixes the resampling, so that a report can be reproduced.

    Returns:
        The difference, its 95% interval and its p-value.

    Raises:
        ValueError: A condition has no attempts, or the two were tested on
          different numbers of layouts.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or not a.size or not b.size:
        raise ValueError("Each condition needs a row of attempts per seed.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"The conditions were tested on {a.shape[1]} and {b.shape[1]} layouts, not the same."
        )
    rng = np.random.default_rng(seed)
    layouts = rng.integers(a.shape[1], size=(resamples, 1, a.shape[1]))
    seeds_a = rng.integers(a.shape[0], size=(resamples, a.shape[0], 1))
    seeds_b = rng.integers(b.shape[0], size=(resamples, b.shape[0], 1))
    draws = a[seeds_a, layouts].mean(axis=(1, 2)) - b[seeds_b, layouts].mean(axis=(1, 2))
    low, high = np.percentile(draws, [2.5, 97.5])
    tail = min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    # No number of resamples can show a chance below one in that number.
    p_value = min(1.0, max(2 * tail, 1 / resamples))
    return Difference(float(a.mean() - b.mean()), float(low), float(high), p_value)


def holm(p_values: Sequence[float]) -> list[float]:
    """Adjusts p-values for having made several comparisons at once.

    Args:
        p_values: The comparisons' p-values.

    Returns:
        The Holm-adjusted p-values, in the order given. A comparison holds at
        the 5% level across the whole family if its adjusted value is below
        0.05.
    """
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    largest = 0.0
    for rank, index in enumerate(order.tolist()):
        largest = max(largest, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = largest
    return adjusted
