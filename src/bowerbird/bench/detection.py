"""Measures whether the scorers find the faults put into a damaged dataset."""

import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird import scoring
from bowerbird.bench import faults
from bowerbird.io import lerobot

SCORERS = ("health", "motion")
# The scorer that should find each kind of fault, and the words its reason
# should contain.
EXPECTED: dict[str, tuple[str, tuple[str, ...]]] = {
    "idle_edge": ("motion", ("idle at",)),
    "pause": ("motion", ("paused",)),
    "cut_short": ("health", ("cut short",)),
    "dropped_frames": ("health", ("dropped", "jumped")),
    "command_glitch": ("motion", ("jumped",)),
    "gripper_chatter": ("motion", ("toggled",)),
}


@dataclass(frozen=True)
class Row:
    """How well one kind of fault was found.

    Attributes:
        kind: The kind of fault.
        episodes: How many episodes have it.
        named: The share for which the expected scorer gave the expected
          reason at the damaged frames.
        flagged: The share for which either scorer flagged anything at all.
        separation: The chance that a damaged episode has a lower overall
          score than an undamaged one. 0.5 is no better than a coin.
    """

    kind: str
    episodes: int
    named: float
    flagged: float
    separation: float


@dataclass(frozen=True)
class Detection:
    """What the scorers found in a damaged dataset.

    Attributes:
        rows: One row per kind of fault.
        separation: As in `Row`, for all damaged episodes together.
        false_alarms: For each scorer, the share of undamaged episodes it
          flagged.
        undamaged: How many episodes are undamaged.
    """

    rows: list[Row]
    separation: float
    false_alarms: dict[str, float]
    undamaged: int


def measure(root: pathlib.Path) -> Detection:
    """Scores a damaged dataset and compares the result with its faults.

    Args:
        root: A dataset folder written by `faults.damage`.

    Returns:
        How well each kind of fault was found.
    """
    dataset = lerobot.load(str(root))
    table = scoring.score(dataset, list(SCORERS), check_video=False, save=False)
    by_scorer = {name: table.get(name) for name in SCORERS}
    overall = np.array(
        [
            np.mean([by_scorer[name][index].score for name in SCORERS])
            for index in range(len(dataset))
        ]
    )
    injected = faults.read(root)
    damaged = {fault.episode for fault in injected}
    clean = np.array([index not in damaged for index in range(len(dataset))])

    rows = []
    for kind in faults.KINDS:
        members = [fault for fault in injected if fault.kind == kind]
        if not members:
            continue
        scorer, words = EXPECTED[kind]
        named = flagged = 0
        for fault in members:
            flagged += any(by_scorer[name][fault.episode].segments for name in SCORERS)
            named += any(
                segment.start_frame < fault.stop
                and segment.end_frame > fault.start
                and any(word in segment.reason for word in words)
                for segment in by_scorer[scorer][fault.episode].segments
            )
        indexes = [fault.episode for fault in members]
        rows.append(
            Row(
                kind,
                len(members),
                named / len(members),
                flagged / len(members),
                _separation(overall[indexes], overall[clean]),
            )
        )
    false_alarms = {
        name: float(np.mean([bool(by_scorer[name][i].segments) for i in np.flatnonzero(clean)]))
        for name in SCORERS
    }
    return Detection(
        rows,
        _separation(overall[~clean], overall[clean]),
        false_alarms,
        int(clean.sum()),
    )


def _separation(damaged: npt.NDArray[Any], clean: npt.NDArray[Any]) -> float:
    """Computes the chance that a damaged episode scores below a clean one.

    Args:
        damaged: The damaged episodes' scores.
        clean: The undamaged episodes' scores.

    Returns:
        The area under the ROC curve, counting ties as half.
    """
    if not len(damaged) or not len(clean):
        return 0.5
    below = (damaged[:, None] < clean[None, :]).sum()
    level = (damaged[:, None] == clean[None, :]).sum()
    return float((below + 0.5 * level) / (len(damaged) * len(clean)))
