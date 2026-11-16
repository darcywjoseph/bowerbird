import json
import pathlib

import numpy as np
import pytest

import bowerbird

h5py = pytest.importorskip("h5py")
pytest.importorskip("lerobot")
pytest.importorskip("robosuite")

from bowerbird.bench import detection, faults, robomimic  # noqa: E402

EPISODES = 30
LENGTH = 120


def _write_source(path: pathlib.Path) -> None:
    """Writes thirty smooth reaches driven by relative commands."""
    generator = np.random.default_rng(0)
    time = np.linspace(0.0, 1.0, LENGTH)
    eased = 10 * time**3 - 15 * time**4 + 6 * time**5
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Lift", "env_kwargs": {"control_freq": 20}}
        )
        for number in range(EPISODES):
            target = generator.uniform(0.5, 1.0, 3)
            position = eased[:, None] * target
            demo = data.create_group(f"demo_{number}")
            actions = np.zeros((LENGTH, 7))
            actions[:-1, :3] = np.diff(position, axis=0) * 40
            actions[:, 6] = np.where(np.arange(LENGTH) < LENGTH // 2, -1.0, 1.0)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("obs/robot0_eef_pos", data=position)
            quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (LENGTH, 1))
            demo.create_dataset("obs/robot0_eef_quat", data=quaternion)
            demo.create_dataset("obs/robot0_gripper_qpos", data=np.zeros((LENGTH, 2)))
            demo.create_dataset("obs/object", data=np.zeros((LENGTH, 10)))


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    folder = tmp_path_factory.mktemp("faults")
    _write_source(folder / "source.hdf5")
    return robomimic.convert(folder / "source.hdf5", folder / "clean", "lift")


def test_damage_gives_each_kind_of_fault_to_its_share_of_episodes(clean: pathlib.Path) -> None:
    target = clean.parent / "damaged"

    made = faults.damage(clean, target, share=0.2, seed=0)

    assert sorted(fault.kind for fault in made) == sorted(faults.KINDS)
    assert faults.read(target) == made
    assert (target / robomimic.ENVIRONMENT_FILE).is_file()
    lengths = [episode.length for episode in bowerbird.load(str(target))]
    by_kind = {fault.kind: fault for fault in made}
    assert lengths[by_kind["idle_edge"].episode] >= LENGTH + 40
    assert lengths[by_kind["pause"].episode] >= LENGTH + 20
    assert lengths[by_kind["cut_short"].episode] <= LENGTH // 5
    assert LENGTH - 20 <= lengths[by_kind["dropped_frames"].episode] < LENGTH
    assert lengths[by_kind["command_glitch"].episode] == LENGTH
    untouched = set(range(EPISODES)) - {fault.episode for fault in made}
    assert all(lengths[index] == LENGTH for index in untouched)


def test_damage_is_repeatable_and_seeded(clean: pathlib.Path) -> None:
    first = faults.damage(clean, clean.parent / "a", seed=1)
    again = faults.damage(clean, clean.parent / "a", seed=1)
    other = faults.damage(clean, clean.parent / "b", seed=2)

    assert first == again
    assert first != other


def test_the_scorers_find_the_faults_they_are_built_for(clean: pathlib.Path) -> None:
    target = clean.parent / "found"
    faults.damage(clean, target, share=0.2, seed=0)

    found = detection.measure(target)

    named = {row.kind: row.named for row in found.rows}
    assert named == dict.fromkeys(faults.KINDS, 1.0)
    assert found.undamaged == EPISODES - len(faults.KINDS)
    assert found.false_alarms["health"] == 0.0
    assert found.separation > 0.9


def test_reading_faults_from_an_undamaged_dataset_fails(clean: pathlib.Path) -> None:
    with pytest.raises(bowerbird.DatasetError, match="no record of faults"):
        faults.read(clean)
