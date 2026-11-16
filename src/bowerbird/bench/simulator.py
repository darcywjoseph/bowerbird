"""Runs a robomimic task in simulation, with the observations a policy was trained on."""

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import robosuite

from bowerbird import errors
from bowerbird.bench import robomimic

Observation = dict[str, npt.NDArray[np.float32]]


class Simulator:
    """One simulated copy of the task a converted dataset was recorded on."""

    def __init__(self, root: pathlib.Path) -> None:
        """Builds the simulator from the settings stored with a dataset.

        Args:
            root: A dataset folder written by `robomimic.convert`.

        Raises:
            DatasetError: The dataset does not record its simulator settings.
        """
        path = root / robomimic.ENVIRONMENT_FILE
        if not path.is_file():
            raise errors.DatasetError(
                f"{root} has no simulator settings. Was it made by robomimic.convert()?"
            )
        settings = json.loads(path.read_text())
        options = dict(settings["env_kwargs"])
        options.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        self._env: Any = robosuite.make(settings["env_name"], **options)

    def reset(self, seed: int) -> Observation:
        """Starts a new attempt from the layout that belongs to a seed.

        Args:
            seed: Chooses where the objects and the robot start. The same seed
              always gives the same layout.

        Returns:
            The first observation.
        """
        # The simulator draws its layouts from NumPy's global generator.
        np.random.seed(seed)
        return _observation(self._env.reset())

    def step(self, action: npt.NDArray[Any]) -> tuple[Observation, bool]:
        """Applies one action.

        Args:
            action: The command for this step, in the dataset's action layout.

        Returns:
            The next observation, and whether the task is now complete.
        """
        raw, _, _, _ = self._env.step(np.asarray(action, dtype=np.float64))
        return _observation(raw), bool(self._env._check_success())

    def close(self) -> None:
        """Releases the simulator."""
        self._env.close()


def _observation(raw: dict[str, npt.NDArray[Any]]) -> Observation:
    """Arranges the simulator's observations as the dataset's columns.

    Args:
        raw: The simulator's observations by name.

    Returns:
        The robot state and the object state, under the dataset's column names.
    """
    state = np.concatenate([raw[key] for key in robomimic.ROBOT_OBSERVATIONS])
    return {
        robomimic.STATE: state.astype(np.float32),
        robomimic.ENVIRONMENT_STATE: raw[robomimic.OBJECT_OBSERVATION].astype(np.float32),
    }
