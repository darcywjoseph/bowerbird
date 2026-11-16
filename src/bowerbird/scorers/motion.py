"""Scores how cleanly the operator moved, from the numbers alone.

The checks are in the spirit of the motion rules used by RINSE and DQAF: time
spent idle, a command that jumps away and straight back, a gripper that opens
and closes repeatedly, commands held at the edge of their range, and a robot
that is far from where it was told to be. Sizes
are measured against each signal's range over the whole dataset. What counts
as a joint's resting position is judged within each task.

The jump and edge-of-range checks assume a command is a position to move to.
Some robots are commanded by how far to move each step instead, and for
those a small one-frame jump or a command held at full size is ordinary. When
the states show that commands are of that kind, only a jump across much of
the range is counted and the edge-of-range check is skipped.

Two more checks compare an episode with the others on its task: how smooth
its motion is, by the spectral arc length of its speed, and how far the robot
travelled. Unlike the checks above they lower the score a little for any
episode worse than the task's median, so that the score ranks episodes and
does not only mark the faulty ones. A task needs enough episodes for its
median to mean something, so small tasks are not compared.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird import errors
from bowerbird.core import episodes, scores
from bowerbird.scorers import signals, smoothness, tracking

# Slower than this fraction of the range per second counts as not moving.
IDLE_SPEED = 0.02
_IDLE_SECONDS = 0.5
# Stillness before the task starts or after it ends can be trimmed away, so it
# counts for less than a pause in the middle of the task.
_IDLE_EDGE_WEIGHT = 0.5
# A command must leave its neighbours by this fraction of its range to be a spike.
_SPIKE_SIZE = 0.02
# A command that says how far to move is jumpy by nature, so only a jump across
# much of its range is counted.
_SPIKE_SIZE_RELATIVE = 0.4
# Some operators drive with taps, as from a keyboard, and then large jumps in
# such a command are how the data normally looks. If more than this share of
# episodes have one, they are not counted.
_SPIKE_COMMON = 0.1
# Everything around a spike must move less than this fraction of the spike's
# size. That is what separates a glitch from a quick tap or a change of direction.
_SPIKE_QUIET = 0.5
_SPIKE_PENALTY = 0.1
_GRIPPER_CLOSED = 0.3
_GRIPPER_OPEN = 0.7
_CHATTER_GAP_SECONDS = 1.0
_CHATTER_TOGGLES = 3
_CHATTER_PENALTY = 0.25
_PINNED_MARGIN = 0.01
_PINNED_SECONDS = 1.0
# A dimension that sits at its limit this often is resting there, not pinned.
_PINNED_RESTING_FRACTION = 0.1
# An episode this many robust standard deviations worse than its task's median
# takes the full penalty for a measure, and is reported from the lower level.
_RELATIVE_FULL = 3.0
_RELATIVE_REPORT = 2.0
_RELATIVE_PENALTY = 0.25
# The least spread to assume within a task: in units of spectral arc length for
# smoothness, and as a fraction of the task's median for distance.
_SMOOTHNESS_SPREAD = 0.2
_DISTANCE_SPREAD = 0.1
_ACTION = episodes.ACTION_COLUMN


@dataclass
class _Draft:
    """What is known about one episode before dataset-wide facts are applied."""

    index: int
    length: int
    fps: float
    segments: list[scores.Segment] = field(default_factory=list)
    spikes: list[scores.Segment] = field(default_factory=list)
    pinned: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    chatter_clusters: int = 0
    idle_frames: float = 0.0
    column: str = ""
    smoothness: float | None = None
    distance: float | None = None
    tracking_error: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0))
    tracking_dimension: npt.NDArray[Any] = field(default_factory=lambda: np.zeros(0, dtype=int))


class MotionScorer:
    """Scores an episode by how direct and smooth its motion is."""

    name = "motion"
    version = "9"
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

        lag = tracking.find_lag(episode_list)
        positional = not tracking.commands_are_relative(episode_list)
        drafts = [
            _draft(episode, position_limits, action_limits, lag, positional)
            for episode in episode_list
        ]
        with_spikes = sum(bool(draft.spikes) for draft in drafts)
        if not positional and with_spikes > _SPIKE_COMMON * len(drafts):
            for draft in drafts:
                draft.spikes = []
        groups = signals.task_groups(episode_list)
        fps = episode_list[0].fps if episode_list else 1.0
        off_command = tracking.segments(
            [draft.tracking_error for draft in drafts],
            [draft.tracking_dimension for draft in drafts],
            groups,
            fps,
        )
        resting = {
            group: _resting_dimensions(
                [draft for draft, other in zip(drafts, groups, strict=True) if other == group]
            )
            for group in set(groups)
        }
        relative = _relative(drafts, groups)
        for draft, episode, group, tracked, compared in zip(
            drafts, episode_list, groups, off_command, relative, strict=True
        ):
            yield _finish(draft, resting[group], episode.names(_ACTION), tracked, compared)


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


def _draft(
    episode: episodes.Episode,
    position_limits: signals.Limits,
    action_limits: signals.Limits,
    lag: int | None,
    positional: bool,
) -> _Draft:
    """Runs the checks that need only dataset-wide ranges.

    Args:
        episode: The episode to check.
        position_limits: The range of the position signal over the dataset.
        action_limits: The range of the actions over the dataset.
        lag: How many frames the robot takes to respond to a command, or None
          if states do not follow actions in this dataset.
        positional: Whether actions are positions to move to. If not, only
          large jumps are counted and the edge-of-range check is skipped.

    Returns:
        The episode's idle and gripper segments, and the raw material for the
        checks that need dataset-wide thresholds.
    """
    fps = episode.fps
    draft = _Draft(episode.index, episode.length, fps)
    position = signals.positions(episode)
    if position is not None:
        idle = signals.speed(position, position_limits.span, fps) < IDLE_SPEED
        column = signals.position_column(episode) or ""
        draft.column = column
        grippers = signals.gripper_dimensions(episode.names(column), position.shape[1])
        arm = [i for i in range(position.shape[1]) if i not in grippers]
        if arm and len(position) > 1:
            arm_speed = smoothness.arm_speed(position, position_limits.span, arm, fps)
            draft.smoothness = smoothness.spectral_arc_length(arm_speed, fps)
            draft.distance = float(arm_speed.sum() / fps)
        for start, stop in signals.runs(idle, max(2, round(_IDLE_SECONDS * fps))):
            reason, weight = _idle_kind(start, stop, len(idle), fps)
            draft.segments.append(scores.Segment(start, stop, 0.0, reason, column))
            draft.idle_frames += weight * (stop - start)

    pair = tracking.command_and_response(episode) if lag is not None else None
    if lag is not None and pair is not None and len(pair[0]) > lag + 1:
        draft.tracking_error, draft.tracking_dimension = tracking.error(
            *pair, lag, position_limits.span, episode.names(_ACTION)
        )

    action = _actions(episode)
    if action is None or action_limits.low is None or action_limits.high is None:
        return draft
    grippers = signals.gripper_dimensions(episode.names(episodes.ACTION_COLUMN), action.shape[1])
    arm = [i for i in range(action.shape[1]) if i not in grippers]
    normalised = (action - action_limits.low) / action_limits.span

    for dimension in grippers:
        for start, stop, toggles in _chatter(normalised[:, dimension], fps):
            reason = f"gripper toggled {toggles} times in {signals.seconds(stop - start, fps)}"
            draft.segments.append(scores.Segment(start, stop, 0.0, reason, _ACTION, (dimension,)))
            draft.chatter_clusters += 1
    names = episode.names(episodes.ACTION_COLUMN)
    smallest = _SPIKE_SIZE if positional else _SPIKE_SIZE_RELATIVE
    for dimension in arm:
        for frame, size in _spikes(normalised[:, dimension], smallest):
            label = names[dimension] if dimension < len(names) else f"#{dimension}"
            reason = f"action {label} jumped {size:.0%} of its range and back in one frame"
            segment = scores.Segment(frame, frame + 1, 0.0, reason, _ACTION, (dimension,))
            draft.spikes.append(segment)
    if not positional:
        return draft

    at_limit = (normalised < _PINNED_MARGIN) | (normalised > 1 - _PINNED_MARGIN)
    for dimension in arm:
        pinned = signals.runs(at_limit[:, dimension], max(2, round(_PINNED_SECONDS * fps)))
        if pinned:
            draft.pinned[dimension] = pinned
    return draft


def _idle_kind(start: int, stop: int, length: int, fps: float) -> tuple[str, float]:
    """Describes an idle stretch by where in the episode it falls.

    Args:
        start: The stretch's first frame.
        stop: One past the stretch's last frame.
        length: The number of frames in the episode.
        fps: Frames per second.

    Returns:
        The reason to report, and how much each idle frame counts against
        the episode.
    """
    duration = signals.seconds(stop - start, fps)
    if start == 0 and stop >= length:
        return "idle for the whole episode", 1.0
    if start == 0:
        return f"idle at the start for {duration}", _IDLE_EDGE_WEIGHT
    if stop >= length:
        return f"idle at the end for {duration}", _IDLE_EDGE_WEIGHT
    return f"paused for {duration} mid-episode", 1.0


def _spikes(command: npt.NDArray[Any], smallest: float) -> list[tuple[int, float]]:
    """Finds frames where a command jumps away from its neighbours and back.

    A spike is one frame that sits far from the frames either side of it,
    while those two frames agree with each other and the command is nearly
    still just before and just after. A quick tap or a change of direction
    fails the last condition, because the command is moving on at least one
    side.

    Args:
        command: One action dimension, scaled so its dataset range is 0 to 1.
        smallest: The least distance from its neighbours that counts.

    Returns:
        Each spike's frame and its size as a fraction of the range.
    """
    if len(command) < 5:
        return []
    centre = command[2:-2]
    size = np.abs(centre - (command[1:-3] + command[3:-1]) / 2)
    quiet = _SPIKE_QUIET * size
    is_spike = (
        (size > smallest)
        & (np.abs(command[3:-1] - command[1:-3]) < quiet)
        & (np.abs(command[1:-3] - command[:-4]) < quiet)
        & (np.abs(command[4:] - command[3:-1]) < quiet)
    )
    return [(int(i) + 2, float(size[i])) for i in np.flatnonzero(is_spike)]


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


def _relative(
    drafts: list[_Draft], groups: list[tuple[str, ...] | None]
) -> list[tuple[float, list[scores.Segment]]]:
    """Compares each episode's smoothness and distance with its task's.

    Args:
        drafts: Every episode's draft.
        groups: Each episode's task, or None for a task too small to compare.

    Returns:
        For each episode, the penalty to apply and the segments that explain
        it. An episode of a small task gets neither.
    """
    results: list[tuple[float, list[scores.Segment]]] = [(0.0, []) for _ in drafts]
    for group in set(groups):
        if group is None:
            continue
        members = [i for i, other in enumerate(groups) if other == group]
        distances = [drafts[i].distance for i in members]
        known = [distance for distance in distances if distance is not None]
        median = float(np.median(known)) if known else 0.0
        unsmooth = smoothness.shortfall([drafts[i].smoothness for i in members], _SMOOTHNESS_SPREAD)
        # A longer way round is worse, so distances are compared negated.
        far = smoothness.shortfall(
            [None if d is None else -d for d in distances], _DISTANCE_SPREAD * median
        )
        for i, rough, long_way, distance in zip(members, unsmooth, far, distances, strict=True):
            draft = drafts[i]
            whole = (0, max(draft.length, 1))
            segments = []
            if rough >= _RELATIVE_REPORT:
                reason = "motion is much less smooth than is normal for its task"
                segments.append(scores.Segment(*whole, 0.0, reason, draft.column))
            if long_way >= _RELATIVE_REPORT and distance is not None and median > 0:
                reason = (
                    f"robot travelled {distance / median:.1f}x the median distance for its task"
                )
                segments.append(scores.Segment(*whole, 0.0, reason, draft.column))
            penalty = sum(
                _RELATIVE_PENALTY * min(shortfall, _RELATIVE_FULL) / _RELATIVE_FULL
                for shortfall in (rough, long_way)
            )
            results[i] = (penalty, segments)
    return results


def _finish(
    draft: _Draft,
    resting: set[int],
    action_names: tuple[str, ...],
    off_command: list[scores.Segment],
    compared: tuple[float, list[scores.Segment]],
) -> scores.EpisodeScore:
    """Applies the dataset-wide facts and turns a draft into a score.

    Args:
        draft: The episode's draft.
        resting: Action dimensions to ignore when looking for pinned commands.
        action_names: The action's dimension names, possibly empty.
        off_command: The stretches where the robot was far from its command.
        compared: The penalty and segments from comparing the episode with
          its task, from `_relative`.

    Returns:
        The episode's score and the segments that explain it.
    """
    length = max(draft.length, 1)
    idle_fraction = draft.idle_frames / length

    pinned_segments = []
    for dimension, pinned in draft.pinned.items():
        if dimension in resting:
            continue
        label = action_names[dimension] if dimension < len(action_names) else f"#{dimension}"
        for start, stop in pinned:
            reason = (
                f"action {label} held at the edge of its recorded range "
                f"for {signals.seconds(stop - start, draft.fps)}"
            )
            pinned_segments.append(scores.Segment(start, stop, 0.0, reason, _ACTION, (dimension,)))
    pinned_fraction = signals.coverage(pinned_segments, length)

    penalty = (
        idle_fraction
        + _SPIKE_PENALTY * len(draft.spikes)
        + _CHATTER_PENALTY * draft.chatter_clusters
        + pinned_fraction
        + signals.coverage(off_command, length)
        + compared[0]
    )
    segments = sorted(
        [*draft.segments, *draft.spikes, *pinned_segments, *off_command, *compared[1]],
        key=lambda s: s.start_frame,
    )
    return scores.EpisodeScore(
        draft.index, float(np.clip(1.0 - penalty, 0.0, 1.0)), tuple(segments)
    )
