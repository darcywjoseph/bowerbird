"""Finds episodes that are broken, as opposed to badly performed.

Checks: the number of frames against the metadata, missing values, dropped
frames in the timestamps, a robot that stops following its commands, and
cameras that freeze or lose frames. The tracking check allows for the robot's
response delay and is judged against other episodes of the same task. The
score is the fraction of the episode that no problem touches.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes, scores
from bowerbird.io import video
from bowerbird.scorers import motion, signals

_GAP_RATIO = 1.5
_TRACKING_CORRELATION = 0.9
_TRACKING_SIGMAS = 8.0
_TRACKING_FLOOR = 0.05
_TRACKING_SECONDS = 0.3
# The longest delay between a command and the robot's response to look for.
_MAX_LAG_SECONDS = 0.5
# A frozen camera repeats a picture exactly. A camera that watches a robot move
# slightly shows small changes instead, so the level is set just above zero.
_FROZEN_CHANGE = 1e-3
# The robot must be moving this fast, in fractions of its range per second, for
# an unchanging picture to be suspicious. Slow motion can be invisible on camera.
_FROZEN_ROBOT_SPEED = 5 * motion.IDLE_SPEED
_FROZEN_SECONDS = 0.5
_KEYFRAME_FRAMES = 2
_FROZEN_STILL_FRACTION = 0.8
_FRAME_COUNT_TOLERANCE = 1
_ACTION = episodes.ACTION_COLUMN


@dataclass
class _Draft:
    """What is known about one episode before dataset-wide thresholds exist."""

    index: int
    rows: int
    fps: float
    segments: list[scores.Segment] = field(default_factory=list)
    tracking_error: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0))
    tracking_dimension: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0, dtype=int))


class _Correlation:
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
        if best is None or best[0] <= _TRACKING_CORRELATION:
            return None
        return best[1]


class HealthScorer:
    """Scores an episode by how much of it is free of recording faults."""

    name = "health"
    requires: frozenset[scores.Requirement] = frozenset()

    def __init__(self, check_video: bool = True) -> None:
        """Creates the scorer.

        Args:
            check_video: Whether to decode the videos to look for frozen and
              missing frames. This is the slow part, and for a Hub dataset it
              downloads the video files.
        """
        self.check_video = check_video
        self.version = "3" if check_video else "3-novideo"

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Scores each episode.

        Args:
            episodes: The episodes to score. They are read twice.

        Yields:
            One score per episode, in the order given.
        """
        episode_list = list(episodes)
        position_limits = signals.Limits()
        fps = episode_list[0].fps if episode_list else 1.0
        correlation = _Correlation(max(1, round(_MAX_LAG_SECONDS * fps)))
        for episode in episode_list:
            position = signals.positions(episode)
            if position is not None:
                position_limits.update(position)
            pair = _command_and_response(episode)
            if pair is not None:
                correlation.update(*pair)
        lag = correlation.best_lag()

        drafts = [self._draft(episode, position_limits, lag) for episode in episode_list]
        groups = signals.task_groups(episode_list)
        thresholds = {
            group: signals.outlier_threshold(
                [
                    draft.tracking_error
                    for draft, other in zip(drafts, groups, strict=True)
                    if other == group
                ],
                _TRACKING_SIGMAS,
                _TRACKING_FLOOR,
            )
            for group in set(groups)
        }
        for draft, group in zip(drafts, groups, strict=True):
            lagging = draft.tracking_error > thresholds[group]
            for start, stop in signals.runs(lagging, max(2, round(_TRACKING_SECONDS * draft.fps))):
                worst = float(np.max(draft.tracking_error[start:stop]))
                reason = (
                    f"robot was up to {worst:.0%} of its range away from its command "
                    f"for {signals.seconds(stop - start, draft.fps)}"
                )
                peak = start + int(np.argmax(draft.tracking_error[start:stop]))
                dims = (int(draft.tracking_dimension[peak]),)
                draft.segments.append(scores.Segment(start, stop, 0.0, reason, _ACTION, dims))
            segments = sorted(draft.segments, key=lambda s: s.start_frame)
            score = 1.0 - signals.coverage(segments, draft.rows) if draft.rows else 0.0
            yield scores.EpisodeScore(draft.index, score, tuple(segments))

    def _draft(
        self, episode: episodes.Episode, position_limits: signals.Limits, lag: int | None
    ) -> _Draft:
        """Runs every check that does not need a dataset-wide threshold.

        Args:
            episode: The episode to check.
            position_limits: The range of the position signal over the dataset.
            lag: How many frames the robot takes to respond to a command in
              this dataset, or None if states do not follow actions and so a
              gap between them means nothing.

        Returns:
            The episode's segments so far, and its tracking error per frame.
        """
        rows = episode.frames.num_rows
        fps = episode.fps
        draft = _Draft(episode.index, rows, fps)
        whole = (0, max(rows, 1))

        if rows != episode.length:
            reason = f"has {rows} frames, but the metadata says {episode.length}"
            draft.segments.append(scores.Segment(*whole, 0.0, reason))
        draft.segments.extend(_missing_values(episode))
        draft.segments.extend(_timestamp_faults(episode))

        pair = _command_and_response(episode) if lag is not None else None
        if lag is not None and pair is not None and len(pair[0]) > lag + 1:
            draft.tracking_error, draft.tracking_dimension = _tracking_error(
                *pair, lag, position_limits.span, episode.names(_ACTION)
            )

        if self.check_video:
            position = signals.positions(episode)
            robot_speed = (
                signals.speed(position, position_limits.span, fps)
                if position is not None
                else np.full(rows, np.inf)
            )
            for camera in episode.cameras:
                draft.segments.extend(_video_faults(episode, camera, robot_speed))
        return draft


def _command_and_response(
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


def _tracking_error(
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


def _missing_values(episode: episodes.Episode) -> list[scores.Segment]:
    """Flags frames whose state or action holds NaN or infinity.

    Args:
        episode: The episode to check.

    Returns:
        One segment per stretch of bad frames per column.
    """
    segments = []
    for column in (episodes.STATE_COLUMN, episodes.ACTION_COLUMN):
        if not episode.has_column(column):
            continue
        values = episode.column(column).astype(np.float64)
        finite = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
        bad: npt.NDArray[np.bool_] = np.asarray(~finite)
        for start, stop in signals.runs(bad):
            reason = f"missing values in {column}"
            segments.append(scores.Segment(start, stop, 0.0, reason, column))
    return segments


def _timestamp_faults(episode: episodes.Episode) -> list[scores.Segment]:
    """Flags gaps and reversals in the frame timestamps.

    Args:
        episode: The episode to check.

    Returns:
        One segment per fault, covering the frames on either side of it.
    """
    if not episode.has_column(episodes.TIMESTAMP_COLUMN):
        return []
    steps = np.diff(episode.timestamps.astype(np.float64))
    expected = 1.0 / episode.fps
    segments = []
    for frame in np.flatnonzero(steps > _GAP_RATIO * expected).tolist():
        dropped = round(steps[frame] / expected) - 1
        reason = f"{dropped} frames dropped ({steps[frame]:.2f} s gap in the timestamps)"
        segments.append(scores.Segment(frame, frame + 2, 0.0, reason))
    backwards: npt.NDArray[np.bool_] = np.asarray(steps <= 0)
    for start, stop in signals.runs(backwards):
        segments.append(scores.Segment(start, stop + 1, 0.0, "timestamps do not increase"))
    return segments


def _video_faults(
    episode: episodes.Episode, camera: str, robot_speed: npt.NDArray[Any]
) -> list[scores.Segment]:
    """Flags a camera that froze, lost frames or cannot be read.

    A still picture is only a fault while the robot is moving: a camera that
    watches a robot at rest should show a still picture.

    Args:
        episode: The episode to check.
        camera: The camera's feature name.
        robot_speed: How fast the robot moves at each frame, in fractions of
          its range per second.

    Returns:
        The camera's fault segments.
    """
    rows = len(robot_speed)
    whole = (0, max(rows, 1))
    try:
        changes = video.frame_changes(episode.video(camera), episode.fps)
    except OSError as error:
        reason = f"video for {camera} could not be read ({error})"
        return [scores.Segment(*whole, 0.0, reason)]

    if abs(len(changes) - rows) > _FRAME_COUNT_TOLERANCE:
        reason = f"video for {camera} has {len(changes)} frames for {rows} steps"
        return [scores.Segment(*whole, 0.0, reason)]

    shared = min(len(changes), rows)
    still = np.zeros(rows, dtype=bool)
    still[:shared] = changes[:shared] < _FROZEN_CHANGE
    # A keyframe re-encodes the same picture slightly differently, which shows
    # up as a small change for a frame or two in the middle of a freeze.
    bridged = signals.close_gaps(still, _KEYFRAME_FRAMES)
    segments = []
    for start, stop in signals.runs(bridged, max(2, round(_FROZEN_SECONDS * episode.fps))):
        # Bridging must only paper over the odd keyframe. A picture that
        # changes every other frame is updating slowly, not frozen.
        if still[start:stop].mean() < _FROZEN_STILL_FRACTION:
            continue
        if robot_speed[start:stop].mean() >= _FROZEN_ROBOT_SPEED:
            reason = (
                f"{camera} frozen for {signals.seconds(stop - start, episode.fps)} "
                "while the robot moved"
            )
            segments.append(scores.Segment(start, stop, 0.0, reason))
    return segments
