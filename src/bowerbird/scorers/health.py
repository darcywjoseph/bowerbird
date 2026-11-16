"""Finds episodes that are broken, as opposed to badly performed.

Checks: the number of frames against the metadata, missing values, dropped
frames in the timestamps, a robot that stops following its commands, and
cameras that freeze or lose frames. The score is the fraction of the episode
that no problem touches.
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


@dataclass
class _Draft:
    """What is known about one episode before dataset-wide thresholds exist."""

    index: int
    rows: int
    fps: float
    segments: list[scores.Segment] = field(default_factory=list)
    tracking_error: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0))


class _Correlation:
    """Running correlation between each action dimension and the next state."""

    def __init__(self) -> None:
        self._count = 0
        self._sums: npt.NDArray[Any] | None = None

    def update(self, action: npt.NDArray[Any], next_state: npt.NDArray[Any]) -> None:
        """Adds one episode's paired values.

        Args:
            action: Actions shaped (frames, dimensions).
            next_state: The state one frame later, with the same shape.
        """
        keep = np.isfinite(action).all(axis=1) & np.isfinite(next_state).all(axis=1)
        a, s = action[keep], next_state[keep]
        sums = np.stack([a.sum(0), s.sum(0), (a * a).sum(0), (s * s).sum(0), (a * s).sum(0)])
        self._sums = sums if self._sums is None else self._sums + sums
        self._count += len(a)

    def follows(self) -> bool:
        """Returns whether the state tracks the action closely in most dimensions."""
        if self._sums is None or self._count < 2:
            return False
        sum_a, sum_s, sum_aa, sum_ss, sum_as = self._sums
        covariance = sum_as - sum_a * sum_s / self._count
        variance = (sum_aa - sum_a**2 / self._count) * (sum_ss - sum_s**2 / self._count)
        moving = variance > 0
        if not moving.any():
            return False
        correlation = covariance[moving] / np.sqrt(variance[moving])
        return bool(np.median(correlation) > _TRACKING_CORRELATION)


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
        self.version = "1" if check_video else "1-novideo"

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Scores each episode.

        Args:
            episodes: The episodes to score. They are read twice.

        Yields:
            One score per episode, in the order given.
        """
        episode_list = list(episodes)
        position_limits = signals.Limits()
        correlation = _Correlation()
        for episode in episode_list:
            position = signals.positions(episode)
            if position is not None:
                position_limits.update(position)
            pair = _command_and_response(episode)
            if pair is not None:
                correlation.update(*pair)
        check_tracking = correlation.follows()

        drafts = [self._draft(episode, position_limits, check_tracking) for episode in episode_list]
        threshold = signals.outlier_threshold(
            [draft.tracking_error for draft in drafts], _TRACKING_SIGMAS, _TRACKING_FLOOR
        )
        for draft in drafts:
            lagging = draft.tracking_error > threshold
            for start, stop in signals.runs(lagging, max(2, round(_TRACKING_SECONDS * draft.fps))):
                worst = float(np.nanmax(draft.tracking_error[start:stop]))
                reason = (
                    f"robot was up to {worst:.0%} of its range away from its command "
                    f"for {signals.seconds(stop - start, draft.fps)}"
                )
                draft.segments.append(scores.Segment(start, stop, 0.0, reason))
            segments = sorted(draft.segments, key=lambda s: s.start_frame)
            score = 1.0 - signals.coverage(segments, draft.rows) if draft.rows else 0.0
            yield scores.EpisodeScore(draft.index, score, tuple(segments))

    def _draft(
        self, episode: episodes.Episode, position_limits: signals.Limits, check_tracking: bool
    ) -> _Draft:
        """Runs every check that does not need a dataset-wide threshold.

        Args:
            episode: The episode to check.
            position_limits: The range of the position signal over the dataset.
            check_tracking: Whether states follow actions closely enough in
              this dataset for a gap between them to mean something.

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

        if check_tracking:
            pair = _command_and_response(episode)
            if pair is not None:
                action, next_state = pair
                error = np.nanmax(np.abs(action - next_state) / position_limits.span, axis=1)
                draft.tracking_error = np.concatenate((error, error[-1:])).astype(np.float32)

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
    """Pairs each action with the state one frame later.

    Args:
        episode: The episode to read.

    Returns:
        The actions and the following states, or None if the episode lacks
        either or they have different shapes and so cannot be compared.
    """
    if not (
        episode.has_column(episodes.ACTION_COLUMN) and episode.has_column(episodes.STATE_COLUMN)
    ):
        return None
    action = episode.actions.astype(np.float64)
    state = episode.states.astype(np.float64)
    if action.ndim != 2 or action.shape != state.shape or len(action) < 2:
        return None
    return action[:-1], state[1:]


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
        bad = ~np.isfinite(values.reshape(len(values), -1)).all(axis=1)
        for start, stop in signals.runs(bad):
            segments.append(scores.Segment(start, stop, 0.0, f"missing values in {column}"))
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
    for start, stop in signals.runs(steps <= 0):
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
