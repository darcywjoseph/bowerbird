"""Turns the official robomimic demonstration files into LeRobot datasets.

The mixed-skill files say which operator recorded each demonstration and how
skilled that operator was. Those labels are kept beside the dataset, because
the benchmark uses them to check that a scorer finds the weaker operators.
"""

import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any

import h5py
import huggingface_hub
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets import lerobot_dataset

from bowerbird import errors

SOURCE_REPO = "robomimic/robomimic_datasets"
SKILLS = ("better", "okay", "worse")
LABELS_FILE = "bowerbird/labels.parquet"
ENVIRONMENT_FILE = "bowerbird/environment.json"

_TASKS = {
    "lift": "Lift the cube.",
    "can": "Put the can in its bin.",
    "square": "Fit the square nut onto the square peg.",
}
_SETS = {"mixed": "mh", "skilled": "ph"}
# The observations a state-only policy is trained on, as in the robomimic and
# Diffusion Policy papers.
_ROBOT_PARTS = {
    "robot0_eef_pos": ("eef_x", "eef_y", "eef_z"),
    "robot0_eef_quat": ("eef_qx", "eef_qy", "eef_qz", "eef_qw"),
    "robot0_gripper_qpos": ("gripper_left", "gripper_right"),
}
_OBJECT_PART = "object"
_ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
_OPERATORS = (1, 2)


@dataclass(frozen=True)
class Label:
    """Who recorded an episode.

    Attributes:
        demo: The demonstration's name in the source file.
        skill: The operator's skill, one of `SKILLS`, or None if the source
          file does not say.
        operator: The operator's number within that skill, or None.
    """

    demo: str
    skill: str | None
    operator: int | None


def download(task: str, data_set: str, folder: pathlib.Path) -> pathlib.Path:
    """Fetches an official state-only demonstration file from the Hub.

    Args:
        task: One of "lift", "can" or "square".
        data_set: "mixed" for operators of mixed skill, "skilled" for one
          skilled operator.
        folder: Where to keep the file.

    Returns:
        The path to the downloaded file.

    Raises:
        DatasetError: The task or set is not one this module knows.
    """
    if task not in _TASKS or data_set not in _SETS:
        raise errors.DatasetError(
            f"Unknown robomimic data {task}/{data_set}. Tasks: {', '.join(_TASKS)}. "
            f"Sets: {', '.join(_SETS)}."
        )
    path = huggingface_hub.hf_hub_download(
        SOURCE_REPO,
        f"v1.5/{task}/{_SETS[data_set]}/low_dim_v15.hdf5",
        repo_type="dataset",
        local_dir=folder,
    )
    return pathlib.Path(path)


def convert(source: pathlib.Path, root: pathlib.Path, task: str) -> pathlib.Path:
    """Writes a robomimic file as a LeRobot dataset.

    Episodes keep the order of the demonstrations' numbers in the source file.
    Any dataset already at `root` is replaced.

    Args:
        source: A state-only robomimic file.
        root: The folder to write the dataset to.
        task: One of "lift", "can" or "square".

    Returns:
        The dataset's folder.

    Raises:
        DatasetError: The file lacks an observation this module needs.
    """
    if root.exists():
        shutil.rmtree(root)
    with h5py.File(source, "r") as file:
        data = file["data"]
        demos = sorted(data, key=lambda name: int(name.split("_")[-1]))
        environment = json.loads(data.attrs["env_args"])
        fps = int(environment["env_kwargs"]["control_freq"])
        first = data[demos[0]]["obs"]
        missing = [key for key in (*_ROBOT_PARTS, _OBJECT_PART) if key not in first]
        if missing:
            raise errors.DatasetError(f"{source} has no observation named {', '.join(missing)}.")
        object_width = int(first[_OBJECT_PART].shape[1])

        dataset = lerobot_dataset.LeRobotDataset.create(
            repo_id=f"bowerbird/robomimic_{root.name}",
            fps=fps,
            root=root,
            robot_type="panda",
            use_videos=False,
            features=_features(object_width),
        )
        for demo in demos:
            group = data[demo]
            state = np.concatenate([group["obs"][key][:] for key in _ROBOT_PARTS], axis=1)
            objects = group["obs"][_OBJECT_PART][:]
            actions = group["actions"][:]
            for frame in range(len(actions)):
                dataset.add_frame(
                    {
                        "observation.state": state[frame].astype(np.float32),
                        "observation.environment_state": objects[frame].astype(np.float32),
                        "action": actions[frame].astype(np.float32),
                        "task": _TASKS[task],
                    }
                )
            dataset.save_episode()
        dataset.finalize()
        labels = _labels(file, demos)

    side = root / LABELS_FILE
    side.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(range(len(labels)), pa.int64()),
                "demo": pa.array([label.demo for label in labels], pa.string()),
                "skill": pa.array([label.skill for label in labels], pa.string()),
                "operator": pa.array([label.operator for label in labels], pa.int64()),
            }
        ),
        side,
    )
    (root / ENVIRONMENT_FILE).write_text(json.dumps(environment, indent=2))
    return root


def read_labels(root: pathlib.Path) -> list[Label]:
    """Reads who recorded each episode of a converted dataset.

    Args:
        root: A dataset folder written by `convert`.

    Returns:
        One label per episode, in episode order.

    Raises:
        DatasetError: The dataset has no labels file.
    """
    path = root / LABELS_FILE
    if not path.is_file():
        raise errors.DatasetError(f"{root} has no operator labels. Was it made by convert()?")
    rows = pq.read_table(path).sort_by("episode_index").to_pylist()
    return [Label(row["demo"], row["skill"], row["operator"]) for row in rows]


def _features(object_width: int) -> dict[str, dict[str, Any]]:
    """Describes the dataset's columns to LeRobot.

    Args:
        object_width: The number of values in the object observation.

    Returns:
        The feature definitions.
    """
    state_names = [name for names in _ROBOT_PARTS.values() for name in names]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (object_width,),
            "names": [f"object_{number}" for number in range(object_width)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(_ACTION_NAMES),),
            "names": list(_ACTION_NAMES),
        },
    }


def _labels(file: h5py.File, demos: list[str]) -> list[Label]:
    """Finds each demonstration's operator from the file's lists of names.

    Args:
        file: The open robomimic file.
        demos: The demonstrations, in episode order.

    Returns:
        One label per demonstration. Skill and operator are None when the
        file does not record them.
    """
    found: dict[str, tuple[str, int]] = {}
    masks = file.get("mask", {})
    for skill in SKILLS:
        for operator in _OPERATORS:
            key = f"{skill}_operator_{operator}"
            if key in masks:
                for name in masks[key][:]:
                    found[name.decode()] = (skill, operator)
    return [Label(demo, *found.get(demo, (None, None))) for demo in demos]
