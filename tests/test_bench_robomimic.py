import json
import pathlib

import numpy as np
import pytest

import bowerbird

h5py = pytest.importorskip("h5py")
pytest.importorskip("lerobot")

from bowerbird.bench import robomimic  # noqa: E402

LENGTHS = {"demo_0": 12, "demo_1": 9, "demo_10": 15, "demo_2": 7}
OPERATORS = {"better_operator_1": ["demo_0", "demo_10"], "worse_operator_2": ["demo_1"]}


def _write_source(path: pathlib.Path) -> None:
    widths = {
        "robot0_eef_pos": 3,
        "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2,
        "object": 10,
    }
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Lift", "env_kwargs": {"control_freq": 20}}
        )
        for name, length in LENGTHS.items():
            number = int(name.split("_")[-1])
            demo = data.create_group(name)
            demo.create_dataset("actions", data=np.full((length, 7), number, dtype=np.float64))
            for key, width in widths.items():
                demo.create_dataset(f"obs/{key}", data=np.zeros((length, width)))
        for mask, names in OPERATORS.items():
            file.create_dataset(f"mask/{mask}", data=np.array(names, dtype="S"))


def test_convert_keeps_episode_order_and_operator_labels(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source.hdf5"
    _write_source(source)

    root = robomimic.convert(source, tmp_path / "lift", "lift")

    dataset = bowerbird.load(str(root))
    assert dataset.fps == 20
    # Demonstrations are ordered by number, so demo_10 comes last, not third.
    assert [episode.length for episode in dataset] == [12, 9, 7, 15]
    assert [float(episode.actions[0, 0]) for episode in dataset] == [0.0, 1.0, 2.0, 10.0]
    assert dataset[0].states.shape == (12, 9)
    assert dataset[0].names("action")[-1] == "gripper"
    assert robomimic.read_labels(root) == [
        robomimic.Label("demo_0", "better", 1),
        robomimic.Label("demo_1", "worse", 2),
        robomimic.Label("demo_2", None, None),
        robomimic.Label("demo_10", "better", 1),
    ]


def test_download_rejects_an_unknown_task(tmp_path: pathlib.Path) -> None:
    with pytest.raises(bowerbird.DatasetError, match="Unknown robomimic data"):
        robomimic.download("juggle", "mixed", tmp_path)
