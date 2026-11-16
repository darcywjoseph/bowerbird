"""Scores how cleanly the operator moved, from the numbers alone.

The checks follow the motion rules used by RINSE and DQAF: time spent idle,
abrupt changes in the commanded motion, a gripper that opens and closes
repeatedly, and commands held at their limits. Thresholds are relative to
the dataset, so a score says how an episode compares with its neighbours.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird import errors
from bowerbird.core import episodes, scores
from bowerbird.scorers import signals

# Slower than this fraction of the range per second counts as not moving.
IDLE_SPEED = 0.02
_IDLE_SECONDS = 0.5
_ABRUPT_SIGMAS = 12.0
_ABRUPT_FLOOR = 1e-6
_ABRUPT_WEIGHT = 5.0
_GRIPPER_CLOSED = 0.3
_GRIPPER_OPEN = 0.7
_CHATTER_GAP_SECONDS = 1.0
_CHATTER_TOGGLES = 3
_CHATTER_PENALTY = 0.25
_PINNED_MARGIN = 0.01
_PINNED_SECONDS = 1.0
# A dimension that sits at its limit this often is resting there, not pinned.
_PINNED_RESTING_FRACTION = 0.1
_ACTION = episodes.ACTION_COLUMN


@dataclass
class _Draft:
    """What is known about one episode before dataset-wide thresholds exist."""

    index: int
    length: int
    fps: float
    segments: list[scores.Segment] = field(default_factory=list)
    change: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0))
    pinned: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    chatter_clusters: int = 0


class MotionScorer:
    """Scores an episode by how direct and smooth its motion is."""

    name = "motion"
    version = "1"
    requires: frozenset[scores.Requirement] = frozenset()

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Scores each episode against thresholds drawn from all of them.

        Args:
            episodes: The episodes to score. They are read twice.

        Yields:
            One score per episode, in the order given.

        Raises:
            ScorerError: An episode has neither states nor actions.
        """
        episode_list = list(episodes)
        position_limits = signals.Limits()
        action_limits = signals.Limits()
        for episode in episode_list:
            position = signals.positions(episode)
            if position is None:
                raise errors.ScorerError(
                    f"The motion scorer needs an 'action' or 'observation.state' column, and "
                    f"episode {episode.index} has neither. Use --scorers health,length to skip it."
                )
            position_limits.update(position)
            action = _actions(episode)
            if action is not None:
                action_limits.update(action)

        drafts = [_draft(episode, position_limits, action_limits) for episode in episode_list]
        threshold = signals.outlier_threshold(
            [draft.change for draft in drafts], _ABRUPT_SIGMAS, _ABRUPT_FLOOR
        )
        resting = _resting_dimensions(drafts)
        for draft, episode in zip(drafts, episode_list, strict=True):
            yield _finish(draft, threshold, resting, episode.names(_ACTION))


def _actions(episode: episodes.Episode) -> npt.NDArray[Any] | None:
    """Reads an episode's actions as a 2-D float array, if it has any.

    Args:
        episode: The episode to read.

    Returns:
        The actions shaped (frames, dimensions), or None.
    """
    if not episode.has_column(episodes.ACTION_COLUMN):
        return None
    values = episode.actions.astype(np.float64)
    return values.reshape(len(values), -1)


def _gripper_dimensions(names: tuple[str, ...], width: int) -> list[int]:
    """Finds which action dimensions drive a gripper, going by their names.

    Args:
        names: The action's dimension names, possibly empty.
        width: The number of action dimensions.

    Returns:
        The indices of the gripper dimensions.
    """
    return [i for i, name in enumerate(names[:width]) if "gripper" in name.lower()]


def _draft(
    episode: episodes.Episode, position_limits: signals.Limits, action_limits: signals.Limits
) -> _Draft:
    """Runs the checks that need only dataset-wide ranges.

    Args:
        episode: The episode to check.
        position_limits: The range of the position signal over the dataset.
        action_limits: The range of the actions over the dataset.

    Returns:
        The episode's idle and gripper segments, and the raw material for the
        checks that need dataset-wide thresholds.
    """
    fps = episode.fps
    draft = _Draft(episode.index, episode.length, fps)
    position = signals.positions(episode)
    if position is not None:
        idle = signals.speed(position, position_limits.span, fps) < IDLE_SPEED
        for start, stop in signals.runs(idle, max(2, round(_IDLE_SECONDS * fps))):
            reason = f"idle for {signals.seconds(stop - start, fps)}"
            draft.segments.append(scores.Segment(start, stop, 0.0, reason))

    action = _actions(episode)
    if action is None or action_limits.low is None or action_limits.high is None:
        return draft
    grippers = _gripper_dimensions(episode.names(episodes.ACTION_COLUMN), action.shape[1])
    arm = [i for i in range(action.shape[1]) if i not in grippers]
    normalised = (action - action_limits.low) / action_limits.span

    if arm and len(action) >= 3:
        change = np.nanmax(np.abs(np.diff(normalised[:, arm], n=2, axis=0)), axis=1) * fps**2
        draft.change = np.concatenate((change[:1], change, change[-1:])).astype(np.float32)
    else:
        draft.change = np.zeros(len(action), dtype=np.float32)

    for dimension in grippers:
        for start, stop, toggles in _chatter(normalised[:, dimension], fps):
            reason = f"gripper toggled {toggles} times in {signals.seconds(stop - start, fps)}"
            draft.segments.append(scores.Segment(start, stop, 0.0, reason))
            draft.chatter_clusters += 1

    at_limit = (normalised < _PINNED_MARGIN) | (normalised > 1 - _PINNED_MARGIN)
    for dimension in arm:
        pinned = signals.runs(at_limit[:, dimension], max(2, round(_PINNED_SECONDS * fps)))
        if pinned:
            draft.pinned[dimension] = pinned
    return draft


def _chatter(gripper: npt.NDArray[Any], fps: float) -> list[tuple[int, int, int]]:
    """Finds bursts where the gripper opens and closes in quick succession.

    Args:
        gripper: The gripper command, scaled so its range is 0 to 1.
        fps: Frames per second.

    Returns:
        For each burst, its first frame, one past its last frame, and how
        many times the gripper changed between open and closed.
    """
    is_open: bool | None = None
    toggles: list[int] = []
    for frame, value in enumerate(gripper.tolist()):
        if value > _GRIPPER_OPEN and is_open is not True:
            if is_open is not None:
                toggles.append(frame)
            is_open = True
        elif value < _GRIPPER_CLOSED and is_open is not False:
            if is_open is not None:
                toggles.append(frame)
            is_open = False

    bursts: list[tuple[int, int, int]] = []
    cluster: list[int] = []
    for frame in toggles:
        if cluster and frame - cluster[-1] > _CHATTER_GAP_SECONDS * fps:
            if len(cluster) >= _CHATTER_TOGGLES:
                bursts.append((cluster[0], cluster[-1] + 1, len(cluster)))
            cluster = []
        cluster.append(frame)
    if len(cluster) >= _CHATTER_TOGGLES:
        bursts.append((cluster[0], cluster[-1] + 1, len(cluster)))
    return bursts


def _resting_dimensions(drafts: list[_Draft]) -> set[int]:
    """Finds action dimensions that sit at a limit as a matter of course.

    Args:
        drafts: Every episode's draft.

    Returns:
        The dimensions whose time at a limit is too common to be a fault.
    """
    total_frames = sum(draft.length for draft in drafts)
    pinned_frames: dict[int, int] = {}
    for draft in drafts:
        for dimension, pinned in draft.pinned.items():
            frames = sum(stop - start for start, stop in pinned)
            pinned_frames[dimension] = pinned_frames.get(dimension, 0) + frames
    return {
        dimension
        for dimension, frames in pinned_frames.items()
        if total_frames and frames / total_frames > _PINNED_RESTING_FRACTION
    }


def _finish(
    draft: _Draft, threshold: float, resting: set[int], action_names: tuple[str, ...]
) -> scores.EpisodeScore:
    """Applies the dataset-wide thresholds and turns a draft into a score.

    Args:
        draft: The episode's draft.
        threshold: The level of change in the command that counts as abrupt.
        resting: Action dimensions to ignore when looking for pinned commands.
        action_names: The action's dimension names, possibly empty.

    Returns:
        The episode's score and the segments that explain it.
    """
    length = max(draft.length, 1)
    idle_segments = [s for s in draft.segments if s.reason.startswith("idle")]
    idle_fraction = signals.coverage(idle_segments, length)

    abrupt_segments = [
        scores.Segment(start, stop, 0.0, "abrupt change in the commanded motion")
        for start, stop in signals.runs(draft.change > threshold)
    ]
    abrupt_fraction = signals.coverage(abrupt_segments, length)

    pinned_segments = []
    for dimension, pinned in draft.pinned.items():
        if dimension in resting:
            continue
        label = action_names[dimension] if dimension < len(action_names) else f"#{dimension}"
        for start, stop in pinned:
            reason = (
                f"action {label} held at its limit for {signals.seconds(stop - start, draft.fps)}"
            )
            pinned_segments.append(scores.Segment(start, stop, 0.0, reason))
    pinned_fraction = signals.coverage(pinned_segments, length)

    penalty = (
        idle_fraction
        + _ABRUPT_WEIGHT * abrupt_fraction
        + _CHATTER_PENALTY * draft.chatter_clusters
        + pinned_fraction
    )
    segments = sorted(
        [*draft.segments, *abrupt_segments, *pinned_segments], key=lambda s: s.start_frame
    )
    return scores.EpisodeScore(
        draft.index, float(np.clip(1.0 - penalty, 0.0, 1.0)), tuple(segments)
    )
