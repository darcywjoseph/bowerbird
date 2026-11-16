"""Score robot demonstrations and turn the scores into a ready-to-train dataset."""

from importlib.metadata import version

from bowerbird.core.episodes import Episode, VideoClip
from bowerbird.core.scores import EpisodeScore, Requirement, Scorer, ScoreTable, Segment
from bowerbird.errors import DatasetError
from bowerbird.io.lerobot import Dataset, load

__version__ = version("bowerbird")

__all__ = [
    "Dataset",
    "DatasetError",
    "Episode",
    "EpisodeScore",
    "Requirement",
    "ScoreTable",
    "Scorer",
    "Segment",
    "VideoClip",
    "__version__",
    "load",
]
