"""Scores, the scorer interface, and the table that stores scores on disk."""

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from bowerbird.core import episodes

# Scorers that exist to be compared against, and so stay out of the overall score.
BASELINES = frozenset({"length", "random"})
_FINGERPRINT_KEY = b"bowerbird.dataset_fingerprint"
_VERSION_KEY = b"bowerbird.version"
_SEGMENT_TYPE = pa.struct(
    [
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("score", pa.float64()),
        ("reason", pa.string()),
        ("column", pa.string()),
        ("dims", pa.list_(pa.int64())),
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
        column: The per-frame column the problem shows up in, such as
          "action", or an empty string if it is not about one column.
        dims: The dimensions of that column the problem shows up in, or an
          empty tuple if it is not about particular dimensions.
    """

    start_frame: int
    end_frame: int
    score: float
    reason: str
    column: str = ""
    dims: tuple[int, ...] = ()


def _segment(stored: dict[str, Any]) -> Segment:
    """Rebuilds a segment from its stored form.

    Args:
        stored: The segment's fields as read from a Parquet file. Files
          written before a field existed lack it.

    Returns:
        The segment.
    """
    return Segment(
        start_frame=stored["start_frame"],
        end_frame=stored["end_frame"],
        score=stored["score"],
        reason=stored["reason"],
        column=stored.get("column") or "",
        dims=tuple(stored.get("dims") or ()),
    )


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


@dataclass(frozen=True)
class EpisodeSummary:
    """Every scorer's verdict on one episode, combined.

    Attributes:
        index: The episode's index in the dataset.
        score: The overall score, from 0 (worst) to 1 (best).
        scores: Each scorer's score for the episode.
        segments: Each flagged frame range, with the scorer that flagged it.
    """

    index: int
    score: float
    scores: dict[str, float]
    segments: tuple[tuple[str, Segment], ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """The reasons behind the scores, longest segment first, no repeats."""
        ordered = sorted(
            (segment for _, segment in self.segments),
            key=lambda segment: segment.start_frame - segment.end_frame,
        )
        return tuple(dict.fromkeys(segment.reason for segment in ordered))


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

    def add(
        self,
        scorer: str,
        version: str,
        scores: Iterable[EpisodeScore],
        *,
        replace: bool = False,
    ) -> None:
        """Stores a scorer's scores.

        Scores from a different version of the same scorer are dropped first,
        so that old and new scores are never mixed.

        Args:
            scorer: The scorer's name.
            version: The scorer's version.
            scores: The scores to store. A score for an episode that already
              has one from this scorer replaces it.
            replace: Whether to drop all of the scorer's stored scores first.
        """
        if replace or self._versions.get(scorer) != version:
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

    def summaries(self, scorers: Sequence[str] | None = None) -> list[EpisodeSummary]:
        """Combines the scorers' verdicts into one summary per episode.

        Args:
            scorers: The scorers whose mean gives the overall score. Defaults
              to every scorer in the table except the length and random
              baselines, or to all of them if there is nothing else.

        Returns:
            One summary per episode, ordered by episode index.

        Raises:
            KeyError: The table has no scores from one of the scorers.
        """
        if scorers is None:
            scorers = [name for name in self._scores if name not in BASELINES] or self.scorers
        for scorer in scorers:
            self.get(scorer)
        indices = sorted({index for scores in self._scores.values() for index in scores})
        summaries = []
        for index in indices:
            by_scorer = {
                name: scores[index] for name, scores in self._scores.items() if index in scores
            }
            counted = [by_scorer[name].score for name in scorers if name in by_scorer]
            segments = tuple(
                (name, segment) for name, score in by_scorer.items() for segment in score.segments
            )
            summaries.append(
                EpisodeSummary(
                    index=index,
                    score=sum(counted) / len(counted) if counted else float("nan"),
                    scores={name: score.score for name, score in by_scorer.items()},
                    segments=segments,
                )
            )
        return summaries

    def worst(self, count: int, scorers: Sequence[str] | None = None) -> list[EpisodeSummary]:
        """Returns the lowest-scoring episodes.

        Args:
            count: How many episodes to return.
            scorers: The scorers whose mean gives the overall score. See
              `summaries` for the default.

        Returns:
            Up to `count` summaries, worst first.
        """
        ranked = sorted(self.summaries(scorers), key=lambda summary: (summary.score, summary.index))
        return ranked[:count]

    def keep_episodes(self, fraction: float, scorers: Sequence[str] | None = None) -> list[int]:
        """Picks the best-scoring episodes to train on.

        Args:
            fraction: The share of episodes to keep, from 0 to 1.
            scorers: The scorers whose mean gives the overall score. See
              `summaries` for the default.

        Returns:
            The indices of the episodes to keep, in ascending order.

        Raises:
            ValueError: The fraction is outside 0 to 1.
        """
        if not 0 <= fraction <= 1:
            raise ValueError(f"Not a fraction between 0 and 1: {fraction=}")
        ranked = sorted(
            self.summaries(scorers), key=lambda summary: (-summary.score, summary.index)
        )
        return sorted(summary.index for summary in ranked[: round(fraction * len(ranked))])

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
            segments = tuple(_segment(segment) for segment in row["segments"] or ())
            score = EpisodeScore(row["episode_index"], row["score"], segments)
            table.add(row["scorer"], row["scorer_version"], [score])
        return table
