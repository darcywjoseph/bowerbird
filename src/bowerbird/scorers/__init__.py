"""The built-in scorers and the names they are asked for by."""

from bowerbird import errors
from bowerbird.core import scores
from bowerbird.scorers import baselines, health, motion

NAMES = ("health", "length", "motion", "random")
DEFAULT = ("health", "length", "motion")


def create(name: str, *, check_video: bool = True) -> scores.Scorer:
    """Creates a built-in scorer.

    Args:
        name: One of the names in `NAMES`.
        check_video: Whether the health scorer should decode the videos.

    Returns:
        The scorer.

    Raises:
        ScorerError: There is no scorer with that name.
    """
    if name == "health":
        return health.HealthScorer(check_video=check_video)
    if name == "length":
        return baselines.LengthScorer()
    if name == "motion":
        return motion.MotionScorer()
    if name == "random":
        return baselines.RandomScorer()
    raise errors.ScorerError(f"No scorer named {name!r}. Choose from: {', '.join(NAMES)}.")
