"""Tests a trained policy by letting it attempt the task in simulation."""

import pathlib
from typing import Any

import numpy as np
import torch
from lerobot.policies import factory

from bowerbird.bench import simulator

HORIZON = 400


def attempt(
    checkpoint: pathlib.Path,
    root: pathlib.Path,
    seeds: list[int],
    *,
    horizon: int = HORIZON,
    batch: int = 25,
    device: str | None = None,
) -> list[bool]:
    """Runs one attempt per seed and reports which succeeded.

    An attempt succeeds if the task is complete at any step before the
    horizon. Attempts run several at a time so the policy can decide for all
    of them in one pass.

    Args:
        checkpoint: The folder holding the trained policy.
        root: The dataset folder the policy was trained on, which records the
          simulator's settings.
        seeds: One seed per attempt. A seed fixes the starting layout.
        horizon: The most steps an attempt may take.
        batch: How many attempts to run at once.
        device: The device to run the policy on. Defaults to the best one
          available.

    Returns:
        Whether each attempt succeeded, in the order of `seeds`.
    """
    chosen = device or _best_device()
    policy_class = factory.get_policy_class("diffusion")
    policy: Any = policy_class.from_pretrained(checkpoint)
    policy.config.device = chosen
    policy.to(chosen).eval()
    preprocess, postprocess = factory.make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": chosen}},
    )

    simulators = [simulator.Simulator(root) for _ in range(min(batch, len(seeds)))]
    results: list[bool] = []
    try:
        for first in range(0, len(seeds), batch):
            group = seeds[first : first + batch]
            active = simulators[: len(group)]
            observations = [sim.reset(seed) for sim, seed in zip(active, group, strict=True)]
            succeeded = [False] * len(group)
            policy.reset()
            # Each attempt gets its own noise, fixed by its seed.
            torch.manual_seed(group[0])
            for _ in range(horizon):
                stacked = {
                    key: torch.from_numpy(np.stack([item[key] for item in observations]))
                    for key in observations[0]
                }
                with torch.inference_mode():
                    actions = postprocess(policy.select_action(preprocess(stacked)))
                commands = actions.to("cpu").numpy()
                for number, sim in enumerate(active):
                    # A finished attempt keeps its place in the batch but is
                    # no longer stepped.
                    if not succeeded[number]:
                        observations[number], succeeded[number] = sim.step(commands[number])
                if all(succeeded):
                    break
            results.extend(succeeded)
    finally:
        for sim in simulators:
            sim.close()
    return results


def _best_device() -> str:
    """Picks the fastest device this machine offers.

    Returns:
        "cuda", "mps" or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
