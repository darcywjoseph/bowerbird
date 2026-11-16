"""Finds episodes that are broken, as opposed to badly performed.

Checks: the number of frames against the metadata, missing values, dropped
frames in the timestamps, cameras that freeze, lose frames or cannot be read,
and an episode far shorter than is normal for its task. The score is the
fraction of the episode that no problem touches, less a fixed amount for a
very short episode.

How well the robot follows its commands is not judged here. A robot pushed
off its command by contact is part of doing the task, so that check belongs
to the motion scorer.
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
# An episode this much shorter than is normal for its task was probably cut
# short. The level is low on purpose: a skilled, fast demonstration is short
# too, and must not be caught.
_SHORT_RATIO = 0.25
_SHORT_PENALTY = 0.25
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
    """The faults found in one episode."""

    index: int
    rows: int
    fps: float
    segments: list[scores.Segment] = field(default_factory=list)


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
        self.version = "5" if check_video else "5-novideo"

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Scores each episode.

        Args:
            episodes: The episodes to score. They are read twice.

        Yields:
            One score per episode, in the order given.
        """
        episode_list = list(episodes)
        position_limits = signals.Limits()
        for episode in episode_list:
            position = signals.positions(episode)
            if position is not None:
                position_limits.update(position)

        groups = signals.task_groups(episode_list)
        lengths_by_group: dict[tuple[str, ...], list[int]] = {}
        for episode, group in zip(episode_list, groups, strict=True):
            if group is not None:
                lengths_by_group.setdefault(group, []).append(episode.length)
        medians = {group: float(np.median(lengths)) for group, lengths in lengths_by_group.items()}

        for episode, group in zip(episode_list, groups, strict=True):
            draft = self._draft(episode, position_limits)
            score = 1.0 - signals.coverage(draft.segments, draft.rows) if draft.rows else 0.0
            # A task with too few episodes has no reliable normal length.
            median = medians.get(group) if group is not None else None
            if median and episode.length < _SHORT_RATIO * median:
                reason = (
                    f"only {episode.length / median:.0%} of the median length for its task; "
                    "it may have been cut short"
                )
                # The whole episode is marked, but it costs a fixed amount: being
                # short does not make every frame bad.
                draft.segments.append(scores.Segment(0, max(draft.rows, 1), 0.0, reason))
                score -= _SHORT_PENALTY
            segments = sorted(draft.segments, key=lambda s: s.start_frame)
            yield scores.EpisodeScore(draft.index, max(score, 0.0), tuple(segments))

    def _draft(self, episode: episodes.Episode, position_limits: signals.Limits) -> _Draft:
        """Runs every check on one episode.

        Args:
            episode: The episode to check.
            position_limits: The range of the position signal over the dataset.

        Returns:
            The episode's segments.
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
