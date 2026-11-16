import numpy as np
import numpy.typing as npt

from bowerbird.scorers import motion, smoothness
from tests import conftest

FPS = 20.0


def _reach(frames: int, peak: float = 1.0) -> npt.NDArray[np.float64]:
    """The bell-shaped speed of one smooth reach."""
    time = np.linspace(0.0, 1.0, frames)
    bell: npt.NDArray[np.float64] = peak * 30 * time**2 * (1 - time) ** 2
    return bell


def test_spectral_arc_length_prefers_one_smooth_reach() -> None:
    smooth = smoothness.spectral_arc_length(_reach(60), FPS)
    halting = _reach(60) * (1 - 0.8 * (np.arange(60) // 6 % 2))
    stop_start = smoothness.spectral_arc_length(halting, FPS)

    assert smooth is not None and stop_start is not None
    assert -1.8 < smooth < -1.2
    assert stop_start < smooth - 1.0


def test_spectral_arc_length_ignores_duration_and_speed() -> None:
    reference = smoothness.spectral_arc_length(_reach(60), FPS)
    slow = smoothness.spectral_arc_length(_reach(120), FPS)
    fast_hand = smoothness.spectral_arc_length(_reach(60, peak=7.0), FPS)

    assert reference is not None and slow is not None and fast_hand is not None
    assert abs(slow - reference) < 0.15
    assert abs(fast_hand - reference) < 1e-9


def test_spectral_arc_length_needs_movement() -> None:
    assert smoothness.spectral_arc_length(np.zeros(60), FPS) is None
    assert smoothness.spectral_arc_length(_reach(4), FPS) is None


def test_shortfall_counts_only_values_below_the_median() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, None, -6.0]

    result = smoothness.shortfall(values)

    assert result[2:6] == [0.0, 0.0, 0.0, 0.0]
    assert 0 < result[1] < result[0] < result[6]
    assert smoothness.shortfall([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert smoothness.shortfall([None, 2.0]) == [0.0, 0.0]


def _path(frames: int, wobble: float = 0.0, detour: float = 0.0) -> npt.NDArray[np.float64]:
    """A two-joint move from 0 to 1, optionally shaky or by a long way round."""
    time = np.linspace(0.0, 1.0, frames)
    eased = 10 * time**3 - 15 * time**4 + 6 * time**5
    first = eased + wobble * np.sin(np.arange(frames) * 2.5)
    second = eased + detour * np.sin(time * np.pi * 6)
    return np.column_stack((first, second))


def test_motion_ranks_shaky_and_roundabout_episodes_below_their_task() -> None:
    generator = np.random.default_rng(0)
    states = [_path(int(frames)) for frames in generator.integers(50, 90, 24)]
    states.append(_path(70, wobble=0.03))
    states.append(_path(70, detour=0.5))
    episodes = [conftest.make_episode(index, state) for index, state in enumerate(states)]

    scores = list(motion.MotionScorer().score(episodes))

    reasons = [" | ".join(segment.reason for segment in score.segments) for score in scores]
    assert "much less smooth" in reasons[24]
    assert "the median distance for its task" in reasons[25]
    ordinary = [score.score for score in scores[:24]]
    assert scores[24].score < min(ordinary)
    assert scores[25].score < min(ordinary)
    assert not any("smooth" in reason or "distance" in reason for reason in reasons[:24])


def test_motion_does_not_compare_within_a_small_task() -> None:
    states = [_path(70) for _ in range(5)] + [_path(70, wobble=0.03)]
    episodes = [conftest.make_episode(index, state) for index, state in enumerate(states)]

    scores = list(motion.MotionScorer().score(episodes))

    reasons = [segment.reason for score in scores for segment in score.segments]
    assert not any("smooth" in reason or "distance" in reason for reason in reasons), reasons
