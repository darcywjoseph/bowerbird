"""Measures how smoothly and how directly the robot moved over a whole episode.

Smoothness is the spectral arc length of the speed profile, as defined by
Balasubramanian, Melendez-Calderon, Roby-Brami and Burdet, "On the analysis
of movement smoothness" (2015). A smooth movement has a speed profile with
little high-frequency content, so its spectrum is a short curve. The measure
does not depend on how long the movement lasts or how fast it is, which a
measure built on jerk does. RINSE (2026) ranks demonstrations by a variant of
the same quantity.

It is written from the published definition and has not been checked against
the authors' own code.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# The constants of the published definition.
_CUTOFF_HZ = 10.0
_AMPLITUDE = 0.05
_PADDING = 4
_MIN_FRAMES = 8
# The spread of a normal distribution is this many median absolute deviations.
_MAD_TO_SIGMA = 1.4826


def arm_speed(
    position: npt.NDArray[Any], span: npt.NDArray[Any], arm: Sequence[int], fps: float
) -> npt.NDArray[Any]:
    """Combines the arm's dimensions into one speed per frame.

    Args:
        position: Positions shaped (frames, dimensions).
        span: Each dimension's range over the dataset.
        arm: The dimensions to include, which leaves out any gripper.
        fps: Frames per second.

    Returns:
        The speed between each pair of frames, in ranges per second, one
        fewer than the number of frames.
    """
    chosen = list(arm)
    velocity = np.diff(position[:, chosen], axis=0) / span[chosen] * fps
    return np.asarray(np.linalg.norm(np.nan_to_num(velocity), axis=1))


def spectral_arc_length(speed: npt.NDArray[Any], fps: float) -> float | None:
    """Measures the smoothness of a speed profile.

    Args:
        speed: The speed at each frame.
        fps: Frames per second.

    Returns:
        The spectral arc length, which is negative. Nearer zero is smoother:
        one smooth reach scores about -1.4. None if the profile is too short
        or the robot never moves.
    """
    if len(speed) < _MIN_FRAMES or not np.any(speed > 0):
        return None
    size = int(2 ** (np.ceil(np.log2(len(speed))) + _PADDING))
    frequency = np.arange(size) * fps / size
    spectrum = np.abs(np.fft.fft(speed, size))
    spectrum = spectrum / spectrum.max()
    low = frequency <= _CUTOFF_HZ
    frequency, spectrum = frequency[low], spectrum[low]
    # Only the band up to the last component of any size counts, so that
    # sensor noise at high frequencies does not lengthen the curve.
    strong = np.flatnonzero(spectrum >= _AMPLITUDE)
    first, last = int(strong[0]), int(strong[-1]) + 1
    frequency, spectrum = frequency[first:last], spectrum[first:last]
    if len(frequency) < 2:
        return 0.0
    band = frequency[-1] - frequency[0]
    steps = np.sqrt((np.diff(frequency) / band) ** 2 + np.diff(spectrum) ** 2)
    return float(-steps.sum())


def shortfall(values: Sequence[float | None], floor: float = 0.0) -> list[float]:
    """Measures how far each value falls below what is normal for its group.

    Args:
        values: One value per episode of a group, where higher is better.
          None marks an episode with no value.
        floor: The least spread to assume. Without it, a group whose episodes
          are nearly identical would make a trivial difference look large.

    Returns:
        For each episode, how far it is below the group's median, in robust
        standard deviations. Zero for an episode at or above the median, or
        with no value.
    """
    known = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(known) < 2:
        return [0.0] * len(values)
    median = float(np.median(known))
    spread = max(_MAD_TO_SIGMA * float(np.median(np.abs(known - median))), floor)
    if spread <= 0:
        return [0.0] * len(values)
    return [0.0 if value is None else max(0.0, (median - value) / spread) for value in values]
