"""Small numeric helpers shared by the scorers that need no GPU."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes, scores

_MAD_TO_SIGMA = 1.4826


def runs(mask: npt.NDArray[np.bool_], min_length: int = 1) -> list[tuple[int, int]]:
    """Finds the stretches of consecutive True values in a mask.

    Args:
        mask: One value per frame.
        min_length: The shortest stretch to report, in frames.

    Returns:
        The half-open frame range of each stretch, in order.
    """
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1).tolist()
    stops = np.flatnonzero(edges == -1).tolist()
    return [
        (start, stop)
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= min_length
    ]


def close_gaps(mask: npt.NDArray[np.bool_], max_gap: int) -> npt.NDArray[np.bool_]:
    """Fills short stretches of False that sit between stretches of True.

    Args:
        mask: One value per frame.
        max_gap: The longest stretch of False to fill, in frames.

    Returns:
        A copy of the mask with the short gaps set to True.
    """
    closed = mask.copy()
    for start, stop in runs(~mask):
        if start > 0 and stop < len(mask) and stop - start <= max_gap:
            closed[start:stop] = True
    return closed


def coverage(segments: Iterable[scores.Segment], length: int) -> float:
    """Computes the fraction of an episode's frames that segments cover.

    Args:
        segments: The segments, which may overlap.
        length: The number of frames in the episode.

    Returns:
        A fraction from 0 to 1, or 0 for an empty episode.
    """
    if length <= 0:
        return 0.0
    mask = np.zeros(length, dtype=bool)
    for segment in segments:
        mask[max(segment.start_frame, 0) : segment.end_frame] = True
    return float(mask.mean())


class Limits:
    """The smallest and largest value seen in each dimension of a signal."""

    def __init__(self) -> None:
        """Creates limits that have seen no values."""
        self.low: npt.NDArray[Any] | None = None
        self.high: npt.NDArray[Any] | None = None

    def update(self, values: npt.NDArray[Any]) -> None:
        """Widens the limits to include new values, ignoring NaN and infinity.

        Args:
            values: An array shaped (frames, dimensions).
        """
        if not values.size:
            return
        finite = np.where(np.isfinite(values), values, np.nan)
        if np.all(np.isnan(finite)):
            return
        low = np.nanmin(finite, axis=0)
        high = np.nanmax(finite, axis=0)
        if self.low is None or self.high is None:
            self.low, self.high = low, high
        else:
            self.low = np.fmin(self.low, low)
            self.high = np.fmax(self.high, high)

    @property
    def span(self) -> npt.NDArray[Any]:
        """Each dimension's range, with infinity where it never varies.

        Dividing by the span turns a change into a fraction of the range, and
        gives zero for a dimension that never moves.
        """
        if self.low is None or self.high is None:
            return np.full(1, np.inf)
        span = self.high - self.low
        return np.where(span > 0, span, np.inf)


def speed(values: npt.NDArray[Any], span: npt.NDArray[Any], fps: float) -> npt.NDArray[Any]:
    """Measures how fast a signal changes at each frame.

    Args:
        values: An array shaped (frames, dimensions).
        span: Each dimension's range over the dataset.
        fps: Frames per second.

    Returns:
        One value per frame: the fastest dimension's change, in fractions of
        its range per second. The first frame repeats the second.
    """
    if len(values) < 2:
        return np.zeros(len(values))
    step = np.nanmax(np.abs(np.diff(values, axis=0)) / span, axis=1) * fps
    return np.concatenate((step[:1], step))


def outlier_threshold(samples: Sequence[npt.NDArray[Any]], sigmas: float, floor: float) -> float:
    """Picks a level above which a value is unusual for the dataset.

    Args:
        samples: Per-frame values from every episode.
        sigmas: How many robust standard deviations above the median counts
          as unusual.
        floor: The lowest level to return, for datasets with almost no spread.

    Returns:
        The threshold.
    """
    if not samples:
        return floor
    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    if not values.size:
        return floor
    median = float(np.median(values))
    spread = float(np.median(np.abs(values - median))) * _MAD_TO_SIGMA
    return max(median + sigmas * spread, floor)


def seconds(frames: int, fps: float) -> str:
    """Formats a number of frames as a duration, such as "1.2 s".

    Args:
        frames: The number of frames.
        fps: Frames per second.
    """
    return f"{frames / fps:.1f} s"


def positions(episode: episodes.Episode) -> npt.NDArray[Any] | None:
    """Returns the signal that best shows where the robot is.

    Args:
        episode: The episode to read.

    Returns:
        The measured state if the episode has one, otherwise the commanded
        action, otherwise None.
    """
    for column in (episodes.STATE_COLUMN, episodes.ACTION_COLUMN):
        if episode.has_column(column):
            values = episode.column(column).astype(np.float64)
            return values.reshape(len(values), -1)
    return None
