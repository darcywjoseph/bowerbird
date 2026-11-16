"""Score robot demonstrations and turn the scores into a ready-to-train dataset."""

from importlib.metadata import version

from bowerbird.core.episodes import Episode, VideoClip
from bowerbird.core.scores import (
    EpisodeScore,
    EpisodeSummary,
    Requirement,
    Scorer,
    ScoreTable,
    Segment,
)
from bowerbird.errors import DatasetError, ScorerError
from bowerbird.io.lerobot import Dataset, load
from bowerbird.scoring import score

__version__ = version("bowerbird")

__all__ = [
    "Dataset",
    "DatasetError",
    "Episode",
    "EpisodeScore",
    "EpisodeSummary",
    "Requirement",
    "ScoreTable",
    "Scorer",
    "ScorerError",
    "Segment",
    "VideoClip",
    "__version__",
    "load",
    "score",
]
