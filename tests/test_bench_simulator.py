import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("robosuite")

from bowerbird.bench import robomimic, simulator

# The can task reports the can's pose relative to the hand, which is the part
# of the first observation the simulator gets wrong.
SETTINGS = {
    "env_name": "PickPlaceCan",
    "env_kwargs": {"robots": ["Panda"], "control_freq": 20, "reward_shaping": False},
}


@pytest.fixture
def root(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / robomimic.ENVIRONMENT_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(SETTINGS))
    return tmp_path


def test_the_first_observation_matches_the_ones_that_follow(root: pathlib.Path) -> None:
    sim = simulator.Simulator(root)
    try:
        first = sim.reset(seed=3)
        # A zero command leaves the hand and the can almost where they were.
        second, _ = sim.step(np.zeros(7))
    finally:
        sim.close()

    for key in (robomimic.STATE, robomimic.ENVIRONMENT_STATE):
        np.testing.assert_allclose(first[key], second[key], atol=0.02)


def test_a_seed_always_gives_the_same_layout(root: pathlib.Path) -> None:
    sim = simulator.Simulator(root)
    try:
        first = sim.reset(seed=5)
        again = sim.reset(seed=5)
        other = sim.reset(seed=6)
    finally:
        sim.close()

    np.testing.assert_array_equal(
        first[robomimic.ENVIRONMENT_STATE], again[robomimic.ENVIRONMENT_STATE]
    )
    assert not np.array_equal(
        first[robomimic.ENVIRONMENT_STATE], other[robomimic.ENVIRONMENT_STATE]
    )
