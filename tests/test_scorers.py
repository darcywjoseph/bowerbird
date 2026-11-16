import numpy as np
import numpy.typing as npt
import pytest

import bowerbird
from bowerbird.scorers import baselines, health, motion, signals
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
    assert "paused for 3.0 s mid-episode" in reasons[6]
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


def test_motion_skips_jump_checks_for_relative_commands() -> None:
    # The command is how far to move each step, and the state has a different
    # layout from the action, as with an arm driven by end-effector movements.
    names = ("dx", "dy", "gripper")
    generator = np.random.default_rng(0)
    episodes = []
    for index in range(6):
        steps = generator.uniform(-1.0, 1.0, (LENGTH, 2)).round()
        steps[10:40, 0] = 1.0  # full speed, held: ordinary for this kind of command
        steps[60, 1], steps[58:60, 1], steps[61:63, 1] = 1.0, 0.0, 0.0
        position = np.cumsum(steps, axis=0) * 0.01
        state = np.column_stack((position[:, 1], position[:, 0], position[:, 0] * 0))
        action = np.column_stack((steps, np.zeros(LENGTH)))
        shifted = np.concatenate((np.zeros((1, 3)), state[:-1]))
        episodes.append(conftest.make_episode(index, shifted, action, action_names=names))

    scores = list(motion.MotionScorer().score(episodes))

    for score in scores:
        flagged = [s for s in score.segments if "jumped" in s.reason or "edge" in s.reason]
        assert not flagged, flagged


def test_motion_counts_a_large_jump_in_a_relative_command_when_it_is_rare() -> None:
    names = ("dx", "dy", "gripper")
    generator = np.random.default_rng(0)
    episodes = []
    for index in range(30):
        steps = generator.normal(0.0, 0.2, (LENGTH, 2)).clip(-1.0, 1.0)
        steps[:, 0] = np.convolve(steps[:, 0], np.ones(5) / 5, mode="same")
        steps[:, 1] = np.convolve(steps[:, 1], np.ones(5) / 5, mode="same")
        steps[0], steps[-1] = (1.0, 1.0), (-1.0, -1.0)  # the command's full range
        if index == 29:
            steps[50, 0] = -1.0 if steps[50, 0] > 0 else 1.0
        position = np.cumsum(steps, axis=0) * 0.01
        state = np.concatenate((np.zeros((1, 2)), position[:-1]))
        action = np.column_stack((steps, np.zeros(LENGTH)))
        padded = np.column_stack((state, np.zeros(LENGTH)))
        episodes.append(conftest.make_episode(index, padded, action, action_names=names))

    scores = list(motion.MotionScorer().score(episodes))

    jumps = [[s for s in score.segments if "jumped" in s.reason] for score in scores]
    assert [segment.start_frame for segment in jumps[29]] == [50]
    assert not any(jumps[:29])


def test_health_flags_a_robot_that_jumps_as_if_frames_were_lost() -> None:
    whole = _smooth(length=140)
    episodes = [conftest.make_episode(index, _smooth(index)) for index in range(6)]
    episodes.append(conftest.make_episode(6, np.delete(whole, slice(60, 75), axis=0)))

    scorer = health.HealthScorer(check_video=False)
    scores = list(scorer.score(episodes))

    for score in scores[:6]:
        assert score.score == 1.0, score.segments
    assert [segment.start_frame for segment in scores[6].segments] == [59]
    assert "frames may be missing" in scores[6].segments[0].reason


def test_health_flags_broken_recordings() -> None:
    def following(index: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        action = _smooth(index)
        state = np.concatenate((action[:1], action[:-1]))
        return state, action

    clean = [conftest.make_episode(index, *following(index)) for index in range(6)]

    state, action = following(0)
    state[30:] = state[30]
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
    assert scores[6].score == 1.0, "a robot off its command is not a broken recording"
    assert "missing values in action" in reasons[7]
    assert scores[7].segments[0].start_frame == 10
    assert "5 frames dropped" in reasons[8]
    assert "has 100 frames, but the metadata says 120" in reasons[9]
    assert scores[9].score == 0.0


def _off_command(score: bowerbird.EpisodeScore) -> list[bowerbird.Segment]:
    return [segment for segment in score.segments if "away from its command" in segment.reason]


def test_tracking_is_skipped_when_actions_are_not_targets() -> None:
    episodes = [
        conftest.make_episode(index, np.cumsum(_smooth(index), axis=0), np.cos(_smooth(index) * 9))
        for index in range(6)
    ]

    scores = list(motion.MotionScorer().score(episodes))

    assert not any(_off_command(score) for score in scores)


def test_idle_at_the_edges_counts_half_as_much_as_a_pause() -> None:
    def padded(before: int, middle: int, after: int) -> npt.NDArray[np.float64]:
        wave = _smooth()
        return np.concatenate(
            (
                np.repeat(wave[:1], before, axis=0),
                wave[:50],
                np.repeat(wave[50:51], middle, axis=0),
                wave[50:],
                np.repeat(wave[-1:], after, axis=0),
            )
        )

    episodes = [
        conftest.make_episode(0, padded(20, 0, 20)),
        conftest.make_episode(1, padded(0, 40, 0)),
    ]

    edges, pause = motion.MotionScorer().score(episodes)

    reasons = [segment.reason for segment in edges.segments]
    assert reasons[0].startswith("idle at the start for 2.")
    assert reasons[-1].startswith("idle at the end for 2.")
    assert edges.segments[0].start_frame == 0
    assert edges.segments[-1].end_frame == 140
    assert [segment.reason for segment in pause.segments] == ["paused for 4.0 s mid-episode"]
    assert 1 - pause.score == pytest.approx(2 * (1 - edges.score), rel=0.15)


def test_tracking_allows_for_the_robots_response_delay() -> None:
    def delayed(
        index: int, lag: int = 4
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        action = _smooth(index)
        state = np.concatenate((np.repeat(action[:1], lag, axis=0), action[:-lag])) + 0.3
        return state, action

    clean = [conftest.make_episode(index, *delayed(index)) for index in range(6)]
    state, action = delayed(0)
    state[40:60] = state[40]
    stuck = conftest.make_episode(6, state, action)

    scores = list(motion.MotionScorer().score([*clean, stuck]))

    assert not any(_off_command(score) for score in scores[:6]), "a steady delay is normal"
    (segment,) = _off_command(scores[6])
    assert scores[6].score < 1.0
    assert 40 <= segment.start_frame <= 50 and 55 <= segment.end_frame <= 62


def test_tracking_ignores_a_gripper_that_stops_short_of_its_command() -> None:
    names = ("shoulder", "elbow", "wrist", "gripper")
    episodes = []
    for index in range(6):
        arm = _smooth(index)
        command = (np.arange(LENGTH) >= 50).astype(np.float64)
        action = np.column_stack((arm, command))
        state = np.column_stack((np.concatenate((arm[:1], arm[:-1])), command * 0.4))
        episodes.append(conftest.make_episode(index, state, action, action_names=names))

    scores = list(motion.MotionScorer().score(episodes))

    assert not any(_off_command(score) for score in scores), scores[0].segments


def test_small_tasks_are_pooled_and_large_ones_stand_alone() -> None:
    episodes = [
        conftest.make_episode(index, _smooth(), task="big" if index < 20 else f"small-{index}")
        for index in range(23)
    ]

    groups = signals.task_groups(episodes)

    assert groups[:20] == [("big",)] * 20
    assert groups[20:] == [None, None, None]


def test_health_flags_only_a_very_short_episode() -> None:
    lengths = {index: 100 for index in range(20)} | {20: 20, 21: 40}
    episodes = [conftest.make_episode(i, _smooth(length=n)) for i, n in lengths.items()]
    few = [
        conftest.make_episode(i, _smooth(length=n), task="rare") for i, n in ((30, 100), (31, 10))
    ]

    scorer = health.HealthScorer(check_video=False)
    scores = {score.episode_index: score for score in scorer.score([*episodes, *few])}

    assert scores[20].score == 0.75
    assert "only 20% of the median length" in scores[20].segments[0].reason
    assert scores[21].score == 1.0, "a fast demonstration at 40% of the median is left alone"
    assert scores[31].score == 1.0, "two episodes are too few to say what is normal"
