"""Damages a clean dataset in known ways, to test whether the scorers notice.

Each fault changes the recorded numbers of an episode. The episode is not run
again in the simulator, so a damaged command does not change the states that
follow it, as it would on a real robot. Timestamps stay regular, because a
LeRobot dataset stores a fixed frame rate.
"""

import pathlib
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bowerbird import errors
from bowerbird.bench import robomimic
from bowerbird.io import lerobot

FAULTS_FILE = "bowerbird/faults.parquet"
_GRIPPER = 6
_ARM = slice(0, 6)
_COMMAND_LIMIT = 1.0


@dataclass(frozen=True)
class Fault:
    """One damaged episode.

    Attributes:
        episode: The episode's number.
        kind: One of `KINDS`.
        start: The first damaged frame, counted in the damaged episode.
        stop: One past the last damaged frame.
    """

    episode: int
    kind: str
    start: int
    stop: int


_Applied = tuple[robomimic.Recording, int, int]
_Damage = Callable[[robomimic.Recording, int, np.random.Generator], _Applied]


def _hold(recording: robomimic.Recording, at: int, frames: int) -> robomimic.Recording:
    """Makes the robot stand still for a while at one frame.

    Args:
        recording: The episode.
        at: The frame to hold at. It may be one past the end.
        frames: How many still frames to insert.

    Returns:
        The episode with the frames inserted before `at`.
    """
    source = min(at, len(recording.actions) - 1)
    # Commands say how far to move, so standing still is a command of zero.
    still = np.zeros((frames, recording.actions.shape[1]))
    still[:, _GRIPPER] = recording.actions[source, _GRIPPER]
    return robomimic.Recording(
        np.insert(recording.state, [at] * frames, recording.state[source], axis=0),
        np.insert(recording.objects, [at] * frames, recording.objects[source], axis=0),
        np.insert(recording.actions, [at] * frames, still, axis=0),
    )


def _idle_edge(recording: robomimic.Recording, fps: int, rng: np.random.Generator) -> _Applied:
    """Holds the robot still for 2 to 5 s before it moves or after it finishes."""
    frames = round(rng.uniform(2.0, 5.0) * fps)
    at = 0 if rng.random() < 0.5 else len(recording.actions)
    return _hold(recording, at, frames), at, at + frames


def _pause(recording: robomimic.Recording, fps: int, rng: np.random.Generator) -> _Applied:
    """Holds the robot still for 1 to 3 s partway through."""
    frames = round(rng.uniform(1.0, 3.0) * fps)
    at = round(rng.uniform(0.2, 0.8) * len(recording.actions))
    return _hold(recording, at, frames), at, at + frames


def _cut_short(recording: robomimic.Recording, fps: int, rng: np.random.Generator) -> _Applied:
    """Ends the episode at 10% to 20% of its length."""
    keep = max(2, round(rng.uniform(0.1, 0.2) * len(recording.actions)))
    cut = robomimic.Recording(
        recording.state[:keep], recording.objects[:keep], recording.actions[:keep]
    )
    return cut, 0, keep


def _dropped_frames(recording: robomimic.Recording, fps: int, rng: np.random.Generator) -> _Applied:
    """Deletes a stretch of 0.2 to 1 s, leaving a jump."""
    length = len(recording.actions)
    frames = min(round(rng.uniform(0.2, 1.0) * fps), length // 3)
    at = int(rng.integers(length // 5, max(length // 5 + 1, length * 4 // 5 - frames)))
    gone = slice(at, at + frames)
    dropped = robomimic.Recording(
        np.delete(recording.state, gone, axis=0),
        np.delete(recording.objects, gone, axis=0),
        np.delete(recording.actions, gone, axis=0),
    )
    return dropped, at - 1, at + 1


def _command_glitch(recording: robomimic.Recording, fps: int, rng: np.random.Generator) -> _Applied:
    """Puts 3 to 10 single-frame jumps into one commanded value."""
    length = len(recording.actions)
    count = min(int(rng.integers(3, 11)), length - 2)
    dimension = int(rng.integers(_ARM.start, _ARM.stop))
    frames = np.sort(rng.choice(np.arange(1, length - 1), size=count, replace=False))
    actions = recording.actions.copy()
    # Each jump goes to the far limit from where the command was.
    actions[frames, dimension] = np.where(
        actions[frames, dimension] > 0, -_COMMAND_LIMIT, _COMMAND_LIMIT
    )
    glitched = robomimic.Recording(recording.state, recording.objects, actions)
    return glitched, int(frames[0]), int(frames[-1]) + 1


def _gripper_chatter(
    recording: robomimic.Recording, fps: int, rng: np.random.Generator
) -> _Applied:
    """Toggles the gripper command 4 to 8 times within 2 s."""
    length = len(recording.actions)
    window = min(2 * fps, length - 2)
    toggles = int(rng.integers(4, 9))
    at = int(rng.integers(1, max(2, length - window)))
    actions = recording.actions.copy()
    block = np.arange(window) * (toggles + 1) // window
    start = -_COMMAND_LIMIT if actions[at, _GRIPPER] > 0 else _COMMAND_LIMIT
    actions[at : at + window, _GRIPPER] = np.where(block % 2 == 0, start, -start)
    chattering = robomimic.Recording(recording.state, recording.objects, actions)
    return chattering, at, at + window


_DAMAGE: dict[str, _Damage] = {
    "idle_edge": _idle_edge,
    "pause": _pause,
    "cut_short": _cut_short,
    "dropped_frames": _dropped_frames,
    "command_glitch": _command_glitch,
    "gripper_chatter": _gripper_chatter,
}
KINDS = tuple(_DAMAGE)


def damage(
    source: pathlib.Path, target: pathlib.Path, *, share: float = 0.2, seed: int = 0
) -> list[Fault]:
    """Writes a copy of a dataset in which some episodes are damaged.

    Each damaged episode gets one fault, and the kinds of fault are used in
    equal numbers. The same seed always damages the same episodes in the same
    way. Any dataset already at `target` is replaced.

    Args:
        source: A dataset folder written by `robomimic.convert`.
        target: The folder to write the damaged copy to.
        share: The share of episodes to damage.
        seed: Chooses the episodes and the details of each fault.

    Returns:
        The faults, in episode order. They are also stored with the copy.

    Raises:
        DatasetError: The share is outside 0 to 1.
    """
    if not 0 <= share <= 1:
        raise errors.DatasetError(f"Not a share between 0 and 1: {share=}")
    dataset = lerobot.load(str(source))
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(dataset))[: round(share * len(dataset))]
    kind_of = {int(episode): KINDS[i % len(KINDS)] for i, episode in enumerate(chosen)}
    fps = round(dataset.fps)
    faults: list[Fault] = []

    def recordings() -> Iterator[robomimic.Recording]:
        for episode in dataset:
            recording = robomimic.Recording(
                episode.states.astype(np.float64),
                episode.column(robomimic.ENVIRONMENT_STATE).astype(np.float64),
                episode.actions.astype(np.float64),
            )
            kind = kind_of.get(episode.index)
            if kind is not None:
                recording, start, stop = _DAMAGE[kind](recording, fps, rng)
                faults.append(Fault(episode.index, kind, start, stop))
            yield recording

    if target.exists():
        shutil.rmtree(target)
    robomimic.write(target, recordings(), fps=fps, task=dataset[0].tasks[0])
    for name in (robomimic.LABELS_FILE, robomimic.ENVIRONMENT_FILE):
        if (source / name).is_file():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target / name)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([fault.episode for fault in faults], pa.int64()),
                "kind": pa.array([fault.kind for fault in faults], pa.string()),
                "start": pa.array([fault.start for fault in faults], pa.int64()),
                "stop": pa.array([fault.stop for fault in faults], pa.int64()),
            }
        ),
        target / FAULTS_FILE,
    )
    return faults


def read(root: pathlib.Path) -> list[Fault]:
    """Reads the faults stored with a damaged dataset.

    Args:
        root: A dataset folder written by `damage`.

    Returns:
        The faults, in episode order.

    Raises:
        DatasetError: The dataset has no record of faults.
    """
    path = root / FAULTS_FILE
    if not path.is_file():
        raise errors.DatasetError(f"{root} has no record of faults. Was it made by damage()?")
    rows = pq.read_table(path).sort_by("episode_index").to_pylist()
    return [Fault(row["episode_index"], row["kind"], row["start"], row["stop"]) for row in rows]
