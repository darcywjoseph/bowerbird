"""The two baselines every other scorer is compared with: length and chance."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes, scores

_LONG_RATIO = 1.5


class LengthScorer:
    """Scores an episode by how short it is next to others on the same task.

    The shortest episode of a task scores 1 and the longest scores 0. Keeping
    the shortest episodes is the baseline that published curation methods
    have struggled to beat.
    """

    name = "length"
    version = "1"
    requires: frozenset[scores.Requirement] = frozenset()

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Scores each episode against the others that share its task.

        Args:
            episodes: The episodes to score.

        Yields:
            One score per episode, in the order given.
        """
        lengths = [(episode.index, episode.tasks, episode.length) for episode in episodes]
        by_task: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for _, tasks, length in lengths:
            by_task[tasks].append(length)
        sorted_by_task = {tasks: np.sort(values) for tasks, values in by_task.items()}

        for index, tasks, length in lengths:
            peers = sorted_by_task[tasks]
            median = float(np.median(peers))
            segments: tuple[scores.Segment, ...] = ()
            if median > 0 and length > _LONG_RATIO * median:
                reason = f"{length / median:.1f}x the median length for its task"
                segments = (scores.Segment(0, length, 0.0, reason),)
            yield scores.EpisodeScore(index, _shortness(peers, length), segments)


def _shortness(sorted_lengths: npt.NDArray[Any], length: int) -> float:
    """Ranks a length among its peers.

    Args:
        sorted_lengths: The lengths of all episodes on the task, ascending.
        length: The length to rank.

    Returns:
        1 for the shortest, 0 for the longest, with ties sharing a rank.
    """
    if len(sorted_lengths) < 2:
        return 1.0
    below = np.searchsorted(sorted_lengths, length, side="left")
    above = np.searchsorted(sorted_lengths, length, side="right")
    rank = (below + above - 1) / 2
    return float(1.0 - rank / (len(sorted_lengths) - 1))


class RandomScorer:
    """Gives every episode a random score, as a control for comparisons."""

    name = "random"
    requires: frozenset[scores.Requirement] = frozenset()

    def __init__(self, seed: int = 0) -> None:
        """Creates the scorer.

        Args:
            seed: Scores are a fixed function of this seed and the episode
              index, so the same seed always gives the same scores.
        """
        self.seed = seed
        self.version = f"1-seed{seed}"

    def score(self, episodes: Iterable[episodes.Episode]) -> Iterator[scores.EpisodeScore]:
        """Draws a score for each episode.

        Args:
            episodes: The episodes to score.

        Yields:
            One score per episode, uniform between 0 and 1.
        """
        for episode in episodes:
            generator = np.random.default_rng([self.seed, episode.index])
            yield scores.EpisodeScore(episode.index, float(generator.random()))
