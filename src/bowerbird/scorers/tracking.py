"""Measures how closely the robot follows its commands.

The robot is expected to reach each commanded position a short, steady delay
after the command is given. This module finds that delay from the data and
reports the stretches where the robot is unusually far from where it was told
to be, which happens when it is blocked, pushed by contact, or stalls.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes, scores
from bowerbird.scorers import signals

_CORRELATION = 0.9
_SIGMAS = 8.0
_FLOOR = 0.05
_SECONDS = 0.3
# The longest delay between a command and the robot's response to look for.
_MAX_LAG_SECONDS = 0.5
# A command counts as relative only if a state's change follows it this closely.
_RELATIVE_CORRELATION = 0.5
_ACTION = episodes.ACTION_COLUMN


class Correlation:
    """Running correlation between actions and the states that follow them.

    The correlation is kept for every delay up to a maximum, so that the delay
    at which the robot best follows its commands can be found.
    """

    def __init__(self, max_lag: int) -> None:
        """Creates an empty correlation.

        Args:
            max_lag: The longest delay to consider, in frames.
        """
        self._counts = [0] * (max_lag + 1)
        self._sums: list[npt.NDArray[Any] | None] = [None] * (max_lag + 1)

    def update(self, action: npt.NDArray[Any], state: npt.NDArray[Any]) -> None:
        """Adds one episode.

        Args:
            action: Actions shaped (frames, dimensions).
            state: States with the same shape, frame for frame.
        """
        for lag in range(len(self._counts)):
            if len(action) - lag < 2:
                break
            a, s = action[: len(action) - lag], state[lag:]
            keep = np.isfinite(a).all(axis=1) & np.isfinite(s).all(axis=1)
            a, s = a[keep], s[keep]
            sums = np.stack([a.sum(0), s.sum(0), (a * a).sum(0), (s * s).sum(0), (a * s).sum(0)])
            held = self._sums[lag]
            self._sums[lag] = sums if held is None else held + sums
            self._counts[lag] += len(a)

    def best_lag(self) -> int | None:
        """Finds the delay at which states follow actions most closely.

        Returns:
            The delay in frames, or None if states do not follow actions
            closely at any delay. That is the case when actions are not
            target positions, such as velocity commands.
        """
        best: tuple[float, int] | None = None
        for lag, (count, sums) in enumerate(zip(self._counts, self._sums, strict=True)):
            if sums is None or count < 2:
                continue
            sum_a, sum_s, sum_aa, sum_ss, sum_as = sums
            covariance = sum_as - sum_a * sum_s / count
            variance = (sum_aa - sum_a**2 / count) * (sum_ss - sum_s**2 / count)
            moving = variance > 0
            if not moving.any():
                continue
            correlation = float(np.median(covariance[moving] / np.sqrt(variance[moving])))
            if best is None or correlation > best[0]:
                best = (correlation, lag)
        if best is None or best[0] <= _CORRELATION:
            return None
        return best[1]


def command_and_response(
    episode: episodes.Episode,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | None:
    """Reads an episode's actions and states, if the two can be compared.

    Args:
        episode: The episode to read.

    Returns:
        The actions and states, frame for frame, or None if the episode lacks
        either or they have different shapes.
    """
    if not (
        episode.has_column(episodes.ACTION_COLUMN) and episode.has_column(episodes.STATE_COLUMN)
    ):
        return None
    action = episode.actions.astype(np.float64)
    state = episode.states.astype(np.float64)
    if action.ndim != 2 or action.shape != state.shape or len(action) < 2:
        return None
    return action, state


def error(
    action: npt.NDArray[Any],
    state: npt.NDArray[Any],
    lag: int,
    span: npt.NDArray[Any],
    names: tuple[str, ...],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Measures how far the robot is from where it was told to be.

    Each command is compared with the state `lag` frames later, which is when
    the robot is expected to have reached it. A constant offset between the
    two, as between a leader and a follower arm that are calibrated
    differently, is removed. Grippers are left out, because a gripper closing
    on an object is meant to stop short of its command.

    Args:
        action: Actions shaped (frames, dimensions).
        state: States with the same shape.
        lag: The robot's response delay in frames.
        span: Each dimension's range over the dataset.
        names: The action's dimension names, possibly empty.

    Returns:
        Two arrays with one value per frame: the largest gap in any arm joint
        as a fraction of that joint's range, and which joint it was in.
    """
    rows, width = action.shape
    grippers = signals.gripper_dimensions(names, width)
    arm = np.asarray([i for i in range(width) if i not in grippers], dtype=int)
    if not arm.size:
        return np.zeros(rows, dtype=np.float32), np.zeros(rows, dtype=int)
    gap = action[: rows - lag, arm] - state[lag:, arm]
    finite = np.where(np.isfinite(gap), gap, np.nan)
    with np.errstate(all="ignore"):
        offset = np.nan_to_num(np.nanmedian(finite, axis=0))
    per_dimension = np.nan_to_num(np.abs(finite - offset) / span[arm])
    error = per_dimension.max(axis=1)
    furthest = arm[per_dimension.argmax(axis=1)]
    # Each gap is placed at the frame of the state, which is when the robot was
    # out of position. The first frames answer to commands from before the
    # episode began, so nothing is claimed about them.
    padding = rows - len(error)
    return (
        np.concatenate((np.zeros(padding), error)).astype(np.float32),
        np.concatenate((np.repeat(furthest[:1], padding), furthest)),
    )


def find_lag(episode_list: Sequence[episodes.Episode]) -> int | None:
    """Finds how many frames the robot takes to respond to a command.

    Args:
        episode_list: Every episode of the dataset.

    Returns:
        The delay in frames, or None if states do not follow actions, in
        which case a gap between them means nothing.
    """
    fps = episode_list[0].fps if episode_list else 1.0
    correlation = Correlation(max(1, round(_MAX_LAG_SECONDS * fps)))
    for episode in episode_list:
        pair = command_and_response(episode)
        if pair is not None:
            correlation.update(*pair)
    return correlation.best_lag()


def relative_evidence(
    episode_list: Sequence[episodes.Episode],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | None:
    """Measures whether each command sets where the robot is or how it moves.

    A position command is followed by the state itself. A relative command,
    which says how far to move in the next step, is followed by the change in
    the state. Actions and states need not have the same layout: each action
    dimension is matched with whichever state dimension follows it best.

    Args:
        episode_list: Every episode of the dataset.

    Returns:
        Two arrays with one value per arm action dimension: its best
        correlation with a state one frame later, and its best correlation
        with a state's change over that frame. None if the dataset lacks
        actions or states.
    """
    count = 0
    sums: list[npt.NDArray[Any]] = []
    arm: list[int] = []
    for episode in episode_list:
        if not (episode.has_column(_ACTION) and episode.has_column(episodes.STATE_COLUMN)):
            continue
        action = episode.actions.astype(np.float64)
        state = episode.states.astype(np.float64)
        if action.ndim != 2 or state.ndim != 2 or len(action) < 3 or len(action) != len(state):
            continue
        if not sums:
            grippers = signals.gripper_dimensions(episode.names(_ACTION), action.shape[1])
            arm = [i for i in range(action.shape[1]) if i not in grippers]
        command = action[:-1, arm]
        # Columns are every state one frame on, then every state's change.
        response = np.concatenate((state[1:], state[1:] - state[:-1]), axis=1)
        if sums and (command.shape[1], response.shape[1]) != sums[4].shape:
            continue
        keep = np.isfinite(command).all(axis=1) & np.isfinite(response).all(axis=1)
        command, response = command[keep], response[keep]
        # Some builds of NumPy warn of overflow in a matrix product that has none.
        with np.errstate(all="ignore"):
            product = command.T @ response
        parts = [
            command.sum(0),
            (command**2).sum(0),
            response.sum(0),
            (response**2).sum(0),
            product,
        ]
        sums = parts if not sums else [held + part for held, part in zip(sums, parts, strict=True)]
        count += len(command)
    if count < 2 or not arm:
        return None

    sum_c, sum_cc, sum_r, sum_rr, sum_cr = sums
    covariance = sum_cr - np.outer(sum_c, sum_r) / count
    variance = np.outer(sum_cc - sum_c**2 / count, sum_rr - sum_r**2 / count)
    with np.errstate(all="ignore"):
        correlation = np.abs(np.where(variance > 0, covariance / np.sqrt(variance), 0.0))
    width = correlation.shape[1] // 2
    return correlation[:, :width].max(axis=1), correlation[:, width:].max(axis=1)


def commands_are_relative(episode_list: Sequence[episodes.Episode]) -> bool:
    """Decides whether the dataset's commands say how far to move each step.

    Args:
        episode_list: Every episode of the dataset.

    Returns:
        True if, for most arm dimensions, the change in the state follows the
        command more closely than the state itself does. False otherwise,
        including when there are no states to judge by.
    """
    evidence = relative_evidence(episode_list)
    if evidence is None:
        return False
    position, movement = evidence
    relative = (movement > position) & (movement > _RELATIVE_CORRELATION)
    return bool(relative.sum() * 2 > len(relative))


def segments(
    errors: Sequence[npt.NDArray[Any]],
    dimensions: Sequence[npt.NDArray[Any]],
    groups: Sequence[Any],
    fps: float,
) -> list[list[scores.Segment]]:
    """Turns per-frame tracking errors into flagged segments.

    An error is unusual when it is far above what is normal for episodes of
    the same group.

    Args:
        errors: Each episode's tracking error per frame, from `error`.
        dimensions: Each episode's worst joint per frame, from `error`.
        groups: Each episode's comparison group, from `signals.task_groups`.
        fps: Frames per second.

    Returns:
        Each episode's segments, in the order the episodes were given.
    """
    thresholds = {
        group: signals.outlier_threshold(
            [error for error, other in zip(errors, groups, strict=True) if other == group],
            _SIGMAS,
            _FLOOR,
        )
        for group in set(groups)
    }
    found: list[list[scores.Segment]] = []
    for episode_error, dimension, group in zip(errors, dimensions, groups, strict=True):
        flagged = []
        lagging = episode_error > thresholds[group]
        for start, stop in signals.runs(lagging, max(2, round(_SECONDS * fps))):
            worst = float(np.max(episode_error[start:stop]))
            reason = (
                f"robot was up to {worst:.0%} of its range away from its command "
                f"for {signals.seconds(stop - start, fps)}"
            )
            peak = start + int(np.argmax(episode_error[start:stop]))
            flagged.append(
                scores.Segment(start, stop, 0.0, reason, _ACTION, (int(dimension[peak]),))
            )
        found.append(flagged)
    return found
