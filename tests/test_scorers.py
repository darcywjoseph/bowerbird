import numpy as np
import numpy.typing as npt

from bowerbird.scorers import baselines, health, motion
from tests import conftest

LENGTH = 100


def _smooth(phase: float = 0.0, length: int = LENGTH) -> npt.NDArray[np.float64]:
    """Three joints tracing sine waves that are never all still at once."""
    time = np.arange(length)[:, None] * 2 * np.pi / LENGTH
    wave: npt.NDArray[np.float64] = np.sin(time + phase + np.array([0.0, 2.0, 4.0]))
    return wave


def test_length_ranks_within_each_task() -> None:
    lengths = {0: 10, 1: 20, 2: 50, 3: 99, 4: 5}
    episodes = [
        conftest.make_episode(index, _smooth(length=length), task="b" if index == 3 else "a")
        for index, length in lengths.items()
    ]

    scores = {score.episode_index: score for score in baselines.LengthScorer().score(episodes)}

    assert scores[4].score == 1.0
    assert scores[2].score == 0.0
    assert scores[0].score > scores[1].score > scores[2].score
    assert scores[3].score == 1.0, "the only episode of its task has nothing to be longer than"
    assert "the median length" in scores[2].segments[0].reason
    assert not scores[0].segments


def test_random_is_repeatable_and_seeded() -> None:
    episodes = [conftest.make_episode(index, _smooth()) for index in range(20)]

    first = list(baselines.RandomScorer(seed=1).score(episodes))
    again = list(baselines.RandomScorer(seed=1).score(episodes))
    other = list(baselines.RandomScorer(seed=2).score(episodes))

    assert first == again
    assert first != other
    assert all(0.0 <= score.score < 1.0 for score in first)
    assert baselines.RandomScorer(seed=1).version != baselines.RandomScorer(seed=2).version


def _with_gripper(
    arm: npt.NDArray[np.float64], gripper: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.column_stack((arm, gripper))


def test_motion_flags_idling_jumps_and_gripper_chatter() -> None:
    names = ("shoulder", "elbow", "wrist", "gripper")
    one_grasp = (np.arange(LENGTH) >= 50).astype(np.float64)
    clean = [
        conftest.make_episode(
            index, _smooth(index), _with_gripper(_smooth(index), one_grasp), action_names=names
        )
        for index in range(6)
    ]

    held = np.concatenate((_smooth()[:40], np.repeat(_smooth()[40:41], 30, axis=0), _smooth()[40:]))
    idle = conftest.make_episode(
        6, held, _with_gripper(held, np.zeros(len(held))), action_names=names
    )

    jumpy_action = _smooth()
    jumpy_action[50, 0] += 1.0
    jumpy = conftest.make_episode(
        7, _smooth(), _with_gripper(jumpy_action, one_grasp), action_names=names
    )

    fumbling = one_grasp.copy()
    fumbling[20:40] = np.arange(20) // 3 % 2
    chatter = conftest.make_episode(
        8, _smooth(), _with_gripper(_smooth(), fumbling), action_names=names
    )

    scores = {
        score.episode_index: score
        for score in motion.MotionScorer().score([*clean, idle, jumpy, chatter])
    }

    for index in range(6):
        assert scores[index].score == 1.0, scores[index].segments
    reasons = {index: " | ".join(s.reason for s in scores[index].segments) for index in (6, 7, 8)}
    assert "idle for 3.0 s" in reasons[6]
    assert scores[6].score < 0.8
    assert "action shoulder jumped 50% of its range and back in one frame" in reasons[7]
    assert [s.start_frame for s in scores[7].segments] == [50]
    assert "gripper toggled" in reasons[8]
    assert scores[8].score < 1.0


def test_motion_does_not_mistake_quick_movement_for_a_glitch() -> None:
    steady = np.linspace(0.0, 1.0, LENGTH)
    tap = steady.copy()
    tap[50:] = steady[50] - (steady[50:] - steady[50]) * 8  # a sharp change of direction
    step = np.where(np.arange(LENGTH) < 50, 0.2, 0.8)  # a move to a new target in one frame
    stop = np.minimum(steady * 3, 1.0)  # fast, then a sudden halt
    episodes = [
        conftest.make_episode(index, None, np.column_stack((command, steady)))
        for index, command in enumerate((tap, step, stop))
    ]

    scores = list(motion.MotionScorer().score(episodes))

    for score in scores:
        assert not [s for s in score.segments if "jumped" in s.reason], score.segments


def test_health_flags_broken_recordings() -> None:
    def following(index: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        action = _smooth(index)
        state = np.concatenate((action[:1], action[:-1]))
        return state, action

    clean = [conftest.make_episode(index, *following(index)) for index in range(6)]

    state, action = following(0)
    state[30:50] = state[30]
    stuck = conftest.make_episode(6, state, action)

    state, action = following(0)
    action[10:13, 1] = np.nan
    gaps = conftest.make_episode(7, state, action)

    timestamps = np.arange(LENGTH) / conftest.FPS
    timestamps[60:] += 0.5
    dropped = conftest.make_episode(8, *following(0), timestamps=timestamps)

    short = conftest.make_episode(9, *following(0), length=LENGTH + 20)

    scorer = health.HealthScorer(check_video=False)
    scores = {
        score.episode_index: score for score in scorer.score([*clean, stuck, gaps, dropped, short])
    }

    for index in range(6):
        assert scores[index].score == 1.0, scores[index].segments
    reasons = {index: " | ".join(s.reason for s in scores[index].segments) for index in scores}
    assert "away from its command" in reasons[6]
    assert "missing values in action" in reasons[7]
    assert scores[7].segments[0].start_frame == 10
    assert "5 frames dropped" in reasons[8]
    assert "has 100 frames, but the metadata says 120" in reasons[9]
    assert scores[9].score == 0.0
    assert 0.0 < scores[6].score < 1.0


def test_health_ignores_tracking_when_actions_are_not_targets() -> None:
    episodes = [
        conftest.make_episode(index, np.cumsum(_smooth(index), axis=0), np.cos(_smooth(index) * 9))
        for index in range(6)
    ]

    scores = list(health.HealthScorer(check_video=False).score(episodes))

    assert all(score.score == 1.0 for score in scores)
