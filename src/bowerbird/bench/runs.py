"""One benchmark run: choose episodes by a rule, train on them, test the result."""

import json
import pathlib
import time
from collections.abc import Callable
from importlib import metadata

import numpy as np

import bowerbird
from bowerbird import errors, scoring
from bowerbird.bench import faults, robomimic, testing, training
from bowerbird.io import lerobot

RULES = ("all", "random", "length", "motion", "operator", "curated", "oracle")
RESULT_FILE = "result.json"
# The operators kept by the "operator" rule: exactly half of a mixed-skill set.
_KEPT_OPERATORS = {("better", 1), ("better", 2), ("okay", 1)}
_TIE_SEED = 0


def choose(root: pathlib.Path, rule: str, fraction: float, seed: int) -> list[int]:
    """Chooses the episodes a rule keeps.

    Args:
        root: The dataset folder.
        rule: One of `RULES`.
        fraction: The share of episodes to keep. Ignored by "all" and
          "oracle", and must be 0.5 for "operator".
        seed: The run's seed. Only "random" depends on it.

    Returns:
        The kept episodes' numbers, ascending.

    Raises:
        ScorerError: The rule is unknown, or cannot keep the fraction asked for.
    """
    dataset = lerobot.load(str(root))
    total = len(dataset)
    count = round(fraction * total)
    if rule == "all":
        return list(range(total))
    if rule == "random":
        generator = np.random.default_rng(seed)
        return sorted(generator.choice(total, size=count, replace=False).tolist())
    if rule == "operator":
        labels = robomimic.read_labels(root)
        kept = [
            index
            for index, label in enumerate(labels)
            if (label.skill, label.operator) in _KEPT_OPERATORS
        ]
        if len(kept) != count:
            raise errors.ScorerError(
                f"The operator rule keeps {len(kept)} of {total} episodes, not {fraction:.0%}."
            )
        return kept
    if rule == "oracle":
        # Drops exactly the episodes that were damaged, which only the
        # benchmark can know.
        damaged = {fault.episode for fault in faults.read(root)}
        return [index for index in range(total) if index not in damaged]
    if rule in ("length", "motion", "curated"):
        # "curated" is what a user would do: rank by every quality scorer.
        scorers = ["health", "motion"] if rule == "curated" else [rule]
        table = scoring.score(dataset, scorers, check_video=False, save=False)
        by_episode = {summary.index: summary for summary in table.summaries(scorers)}
        # Episodes with equal scores are ordered by a fixed shuffle, so that a
        # tie is not settled by the order the data happened to be recorded in.
        shuffle = np.random.default_rng(_TIE_SEED).permutation(total)
        ranked = sorted(range(total), key=lambda i: (-by_episode[i].score, shuffle[i]))
        return sorted(ranked[:count])
    raise errors.ScorerError(f"Unknown rule {rule!r}. Rules: {', '.join(RULES)}.")


def run(
    root: pathlib.Path,
    output: pathlib.Path,
    *,
    rule: str,
    fraction: float,
    seed: int,
    steps: int,
    attempts: int,
    horizon: int = testing.HORIZON,
    device: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Runs one condition with one seed and writes its result file.

    A run whose result file exists is not repeated. An interrupted run carries
    on from its last checkpoint.

    Args:
        root: The dataset folder.
        output: The folder for this run.
        rule: One of `RULES`.
        fraction: The share of episodes to keep.
        seed: The training seed, and the seed of a random subset.
        steps: The number of gradient steps.
        attempts: How many times the trained policy attempts the task. The
          starting layouts are seeds 0 to attempts - 1, whatever the rule.
        horizon: The most steps an attempt may take.
        device: The device to train and test on.
        log: Called with each line of progress.

    Returns:
        The result, as written to the result file.
    """
    say = log or (lambda line: None)
    result_path = output / RESULT_FILE
    if result_path.is_file():
        say(f"Already done: {result_path}")
        loaded: dict[str, object] = json.loads(result_path.read_text())
        return loaded

    episodes = choose(root, rule, fraction, seed)
    say(f"{rule}: training on {len(episodes)} episodes for {steps} steps (seed {seed})")
    started = time.time()
    checkpoint = training.train(root, episodes, output, steps=steps, seed=seed, device=device)
    trained = time.time()
    say(f"Testing over {attempts} attempts")
    successes = testing.attempt(
        checkpoint, root, list(range(attempts)), horizon=horizon, device=device
    )
    result: dict[str, object] = {
        "dataset": root.name,
        "rule": rule,
        "fraction": 1.0 if rule == "all" else fraction,
        "seed": seed,
        "steps": steps,
        "horizon": horizon,
        "episodes": episodes,
        "successes": successes,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "train_seconds": round(trained - started),
        "test_seconds": round(time.time() - trained),
        "versions": {
            "bowerbird": bowerbird.__version__,
            "lerobot": metadata.version("lerobot"),
            "robosuite": metadata.version("robosuite"),
            "mujoco": metadata.version("mujoco"),
        },
    }
    result_path.write_text(json.dumps(result, indent=2))
    say(f"Success rate {result['success_rate']:.1%}. Result: {result_path}")
    return result
