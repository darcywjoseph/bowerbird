"""Scores, the scorer interface, and the table that stores scores on disk."""

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from bowerbird.core import episodes

_FINGERPRINT_KEY = b"bowerbird.dataset_fingerprint"
_VERSION_KEY = b"bowerbird.version"
_SEGMENT_TYPE = pa.struct(
    [
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("score", pa.float64()),
        ("reason", pa.string()),
    ]
)
_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("scorer", pa.string()),
        ("scorer_version", pa.string()),
        ("score", pa.float64()),
        ("segments", pa.list_(_SEGMENT_TYPE)),
    ]
)


class Requirement(Enum):
    """Something a scorer needs beyond the per-frame numbers."""

    VIDEO = "video"
    GPU = "gpu"
    POLICY = "policy"
    ROLLOUTS = "rollouts"


@dataclass(frozen=True)
class Segment:
    """A frame range inside an episode, with a score and the reason for it.

    Attributes:
        start_frame: The first frame of the range.
        end_frame: One past the last frame of the range.
        score: The range's quality, from 0 (worst) to 1 (best).
        reason: A plain-language cause, such as "stalled 4 s before grasp".
    """

    start_frame: int
    end_frame: int
    score: float
    reason: str


@dataclass(frozen=True)
class EpisodeScore:
    """One scorer's verdict on one episode.

    Attributes:
        episode_index: The episode's index in the dataset.
        score: The episode's quality, from 0 (worst) to 1 (best).
        segments: The frame ranges that explain the score.
    """

    episode_index: int
    score: float
    segments: tuple[Segment, ...] = ()


class Scorer(Protocol):
    """Anything that turns episodes into scores.

    Attributes:
        name: The scorer's short name, such as "motion".
        version: Changes whenever the scorer's output would change.
        requires: What the scorer needs beyond the per-frame numbers.
    """

    name: str
    version: str
    requires: frozenset[Requirement]

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[EpisodeScore]:
        """Scores each episode.

        Args:
            episodes: The episodes to score.

        Yields:
            One score per episode.
        """
        ...


class ScoreTable:
    """All scores for one version of a dataset, from any number of scorers.

    Attributes:
        dataset_fingerprint: Identifies the dataset version that was scored.
    """

    def __init__(self, dataset_fingerprint: str = "") -> None:
        """Creates an empty table.

        Args:
            dataset_fingerprint: Identifies the dataset version being scored.
        """
        self.dataset_fingerprint = dataset_fingerprint
        self._versions: dict[str, str] = {}
        self._scores: dict[str, dict[int, EpisodeScore]] = {}

    def __len__(self) -> int:
        return sum(len(scores) for scores in self._scores.values())

    @property
    def scorers(self) -> tuple[str, ...]:
        """The names of the scorers that have scores in the table."""
        return tuple(self._scores)

    def add(self, scorer: str, version: str, scores: Iterable[EpisodeScore]) -> None:
        """Stores a scorer's scores.

        Scores from a different version of the same scorer are dropped first,
        so that old and new scores are never mixed.

        Args:
            scorer: The scorer's name.
            version: The scorer's version.
            scores: The scores to store. A score for an episode that already
              has one from this scorer replaces it.
        """
        if self._versions.get(scorer) != version:
            self._scores[scorer] = {}
        self._versions[scorer] = version
        for score in scores:
            self._scores[scorer][score.episode_index] = score

    def get(self, scorer: str) -> dict[int, EpisodeScore]:
        """Returns one scorer's scores.

        Args:
            scorer: The scorer's name.

        Returns:
            The scorer's scores, keyed by episode index.

        Raises:
            KeyError: The table has no scores from that scorer.
        """
        if scorer not in self._scores:
            raise KeyError(
                f"No scores from {scorer!r}. Scorers in this table: "
                f"{', '.join(self.scorers) or 'none'}."
            )
        return dict(self._scores[scorer])

    def version(self, scorer: str) -> str:
        """Returns the version of a scorer whose scores are in the table.

        Args:
            scorer: The scorer's name.

        Returns:
            The scorer's version.

        Raises:
            KeyError: The table has no scores from that scorer.
        """
        self.get(scorer)
        return self._versions[scorer]

    def save(self, path: str | Path) -> None:
        """Writes the table to a Parquet file, replacing any existing file.

        Args:
            path: Where to write the file.
        """
        rows = [
            {
                "episode_index": score.episode_index,
                "scorer": scorer,
                "scorer_version": self._versions[scorer],
                "score": score.score,
                "segments": [vars(segment) for segment in score.segments],
            }
            for scorer, scores in self._scores.items()
            for score in scores.values()
        ]
        schema = _SCHEMA.with_metadata(
            {
                _FINGERPRINT_KEY: self.dataset_fingerprint.encode(),
                _VERSION_KEY: metadata.version("bowerbird").encode(),
            }
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so a crash cannot leave half a file.
        temporary = path.with_name(path.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreTable":
        """Reads a table written by `save`.

        Args:
            path: The Parquet file to read.

        Returns:
            The table.
        """
        arrow_table = pq.read_table(path)
        file_metadata = arrow_table.schema.metadata or {}
        table = cls(file_metadata.get(_FINGERPRINT_KEY, b"").decode())
        for row in arrow_table.to_pylist():
            segments = tuple(Segment(**segment) for segment in row["segments"] or ())
            score = EpisodeScore(row["episode_index"], row["score"], segments)
            table.add(row["scorer"], row["scorer_version"], [score])
        return table
