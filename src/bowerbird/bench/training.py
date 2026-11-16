"""Trains a Diffusion Policy on chosen episodes of a dataset, through LeRobot."""

import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

from bowerbird import errors

# The settings published for Diffusion Policy on state-only robomimic tasks.
# LeRobot's own defaults are sized for images.
_POLICY = {
    "type": "diffusion",
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "drop_n_last_frames": 7,
    "down_dims": "[256,512,1024]",
    "push_to_hub": "false",
}
BATCH_SIZE = 256
_CHECKPOINTS = 4


def train(
    root: pathlib.Path,
    episodes: Sequence[int],
    output: pathlib.Path,
    *,
    steps: int,
    seed: int,
    device: str | None = None,
    workers: int = 4,
) -> pathlib.Path:
    """Trains a policy and returns its final checkpoint.

    A run that was interrupted carries on from its last saved checkpoint. A
    run that already finished is not repeated.

    Args:
        root: The dataset folder.
        episodes: The episodes to train on.
        output: The folder for this run's checkpoints and logs.
        steps: The number of gradient steps.
        seed: The training seed.
        device: The device to train on, such as "cuda". Defaults to LeRobot's
          choice.
        workers: The number of data-loading processes.

    Returns:
        The folder holding the trained policy.

    Raises:
        ScorerError: Training failed, or finished without a final checkpoint.
    """
    final = output / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    if (final / "model.safetensors").is_file():
        return final

    last = output / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if last.is_file():
        command += [f"--config_path={last}", "--resume=true"]
    else:
        if output.exists():
            # LeRobot refuses to start in a folder that exists. One with no
            # checkpoint holds nothing worth keeping.
            _remove_empty_run(output)
        info = json.loads((root / "meta" / "info.json").read_text())
        command += [
            f"--dataset.repo_id=bowerbird/{root.name}",
            f"--dataset.root={root}",
            f"--dataset.episodes={json.dumps(sorted(episodes))}",
            f"--output_dir={output}",
            f"--steps={steps}",
            f"--seed={seed}",
            f"--batch_size={BATCH_SIZE}",
            f"--save_freq={max(1, steps // _CHECKPOINTS)}",
            f"--num_workers={workers}",
            f"--job_name={root.name}_{info['total_episodes']}",
            *(f"--policy.{key}={value}" for key, value in _POLICY.items()),
        ]
        if device is not None:
            command.append(f"--policy.device={device}")

    done = subprocess.run(command, check=False)
    if done.returncode != 0:
        raise errors.ScorerError(f"Training failed with exit status {done.returncode}.")
    if not (final / "model.safetensors").is_file():
        raise errors.ScorerError(f"Training finished without writing {final}.")
    return final


def _remove_empty_run(output: pathlib.Path) -> None:
    """Deletes a run folder that holds no checkpoint.

    Args:
        output: The run's folder.

    Raises:
        ScorerError: The folder holds a checkpoint, so it is not empty.
    """
    if any(output.glob("checkpoints/*/pretrained_model/model.safetensors")):
        raise errors.ScorerError(
            f"{output} holds checkpoints but no resume point. Remove it or choose another folder."
        )
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    output.rmdir()
