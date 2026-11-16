"""The full set of runs for one task, and the report made from their results.

The conditions and comparisons here are the ones written down in the
benchmark protocol before any run. They are fixed in code so that none can be
chosen after seeing results.
"""

import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

from bowerbird import errors
from bowerbird.bench import faults, robomimic, runs, statistics

FAULT_SHARE = 0.2
FAULT_SEED = 0
SIGNIFICANCE = 0.05


@dataclass(frozen=True)
class Condition:
    """One way of choosing the training data.

    Attributes:
        name: The condition's name in reports.
        data: Which dataset it draws on: "mixed", "skilled" or "damaged".
        rule: The rule that chooses episodes, one of `runs.RULES`.
        fraction: The share of episodes the rule keeps.
    """

    name: str
    data: str
    rule: str
    fraction: float


CONDITIONS = (
    # Part A: ranking, on demonstrations from operators of mixed skill.
    Condition("all", "mixed", "all", 1.0),
    Condition("random-50", "mixed", "random", 0.5),
    Condition("length-50", "mixed", "length", 0.5),
    Condition("motion-50", "mixed", "motion", 0.5),
    Condition("operator-50", "mixed", "operator", 0.5),
    Condition("random-80", "mixed", "random", 0.8),
    Condition("length-80", "mixed", "length", 0.8),
    Condition("motion-80", "mixed", "motion", 0.8),
    # Part B: faults, on demonstrations from one skilled operator.
    Condition("clean", "skilled", "all", 1.0),
    Condition("damaged", "damaged", "all", 1.0),
    Condition("curated", "damaged", "curated", 1.0 - FAULT_SHARE),
    Condition("oracle", "damaged", "oracle", 1.0 - FAULT_SHARE),
)
# Each pair is (condition, the condition it is compared against).
COMPARISONS = (
    ("length-50", "random-50"),
    ("motion-50", "random-50"),
    ("motion-50", "length-50"),
    ("random-50", "all"),
    ("length-50", "all"),
    ("motion-50", "all"),
    ("length-80", "random-80"),
    ("motion-80", "random-80"),
    ("motion-80", "length-80"),
    ("random-80", "all"),
    ("length-80", "all"),
    ("motion-80", "all"),
    ("curated", "damaged"),
    ("curated", "oracle"),
)
# The benchmark can only rank scorers if these differences are real.
CHECKS = (
    ("operator-50", "random-50", "keeping the better operators helps"),
    ("clean", "damaged", "the faults hurt"),
    ("oracle", "damaged", "removing the damaged episodes helps"),
)


def dataset(folder: pathlib.Path, task: str, data: str) -> pathlib.Path:
    """Returns where one of a task's datasets is kept.

    Args:
        folder: The benchmark's folder.
        task: The task's name.
        data: "mixed", "skilled" or "damaged".

    Returns:
        The dataset's folder.
    """
    return folder / "datasets" / f"{task}_{data}"


def prepare(folder: pathlib.Path, task: str) -> None:
    """Downloads a task's datasets and makes the damaged copy, if missing.

    Args:
        folder: The benchmark's folder.
        task: The task's name.
    """
    for data in ("mixed", "skilled"):
        root = dataset(folder, task, data)
        if not (root / robomimic.ENVIRONMENT_FILE).is_file():
            source = robomimic.download(task, data, folder / "source")
            robomimic.convert(source, root, task)
    damaged = dataset(folder, task, "damaged")
    if not (damaged / faults.FAULTS_FILE).is_file():
        faults.damage(dataset(folder, task, "skilled"), damaged, share=FAULT_SHARE, seed=FAULT_SEED)


def run_folder(folder: pathlib.Path, task: str, condition: str, seed: int) -> pathlib.Path:
    """Returns where one run keeps its checkpoints and result.

    Args:
        folder: The benchmark's folder.
        task: The task's name.
        condition: The condition's name.
        seed: The training seed.

    Returns:
        The run's folder.
    """
    return folder / "runs" / task / condition / f"seed{seed}"


def execute(
    folder: pathlib.Path,
    task: str,
    *,
    seeds: int,
    steps: int,
    attempts: int,
    shard: tuple[int, int] = (0, 1),
    device: str | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Runs every condition of a task with every seed.

    Finished runs are skipped, so the command can be repeated after an
    interruption. Several machines can share the work by each taking a shard.

    Args:
        folder: The benchmark's folder.
        task: The task's name.
        seeds: How many training seeds to run, numbered from 0.
        steps: The number of gradient steps per run.
        attempts: How many test attempts per run.
        shard: This machine's number and the number of machines. A machine
          takes the runs whose position in the list leaves its number as the
          remainder.
        device: The device to train and test on.
        log: Called with each line of progress.

    Raises:
        ScorerError: The shard is not a valid part of the whole.
    """
    index, count = shard
    if not 0 <= index < count:
        raise errors.ScorerError(f"Shard {index} of {count} is not valid.")
    say = log or (lambda line: None)
    prepare(folder, task)
    # Seeds are the outer loop, so that an unfinished round still holds
    # complete seeds for every condition.
    planned = [(seed, condition) for seed in range(seeds) for condition in CONDITIONS]
    mine = [item for position, item in enumerate(planned) if position % count == index]
    for number, (seed, condition) in enumerate(mine, start=1):
        say(f"[{number}/{len(mine)}] {task} {condition.name} seed {seed}")
        runs.run(
            dataset(folder, task, condition.data),
            run_folder(folder, task, condition.name, seed),
            rule=condition.rule,
            fraction=condition.fraction,
            seed=seed,
            steps=steps,
            attempts=attempts,
            device=device,
            log=log,
        )


def results(folder: pathlib.Path, task: str) -> dict[str, list[list[bool]]]:
    """Collects a task's finished runs.

    Args:
        folder: The benchmark's folder.
        task: The task's name.

    Returns:
        For each condition with at least one finished run, its attempts: a
        row per training seed, in seed order.
    """
    found: dict[str, list[list[bool]]] = {}
    for condition in CONDITIONS:
        base = folder / "runs" / task / condition.name
        paths = sorted(
            base.glob(f"seed*/{runs.RESULT_FILE}"), key=lambda path: int(path.parent.name[4:])
        )
        rows = [json.loads(path.read_text())["successes"] for path in paths]
        if rows:
            found[condition.name] = rows
    return found


def report(folder: pathlib.Path, task: str) -> str:
    """Writes up a task's results: success rates, checks and comparisons.

    Args:
        folder: The benchmark's folder.
        task: The task's name.

    Returns:
        The report as text.

    Raises:
        ScorerError: No run of the task has finished.
    """
    found = results(folder, task)
    if not found:
        raise errors.ScorerError(f"No finished runs of {task} under {folder}.")
    lines = [f"Task: {task}", "", f"{'condition':12} {'seeds':>5} {'success':>8}  by seed"]
    for name, rows in found.items():
        rates = [sum(row) / len(row) for row in rows]
        by_seed = " ".join(f"{rate:.0%}" for rate in rates)
        lines.append(f"{name:12} {len(rows):>5} {sum(rates) / len(rates):>8.1%}  {by_seed}")

    lines += ["", "Checks (the benchmark can only rank scorers if these hold):"]
    for first, second, meaning in CHECKS:
        if first in found and second in found:
            result = statistics.compare(found[first], found[second])
            holds = "holds" if result.low > 0 else "DOES NOT HOLD"
            lines.append(f"  {meaning}: {_describe(first, second, result)}  {holds}")
        else:
            lines.append(f"  {meaning}: not run yet")

    ready = [(a, b) for a, b in COMPARISONS if a in found and b in found]
    compared = [statistics.compare(found[a], found[b]) for a, b in ready]
    # The correction counts every declared comparison, run or not, so that a
    # partial report cannot look better than the full one will.
    padded = [result.p_value for result in compared] + [1.0] * (len(COMPARISONS) - len(ready))
    adjusted = statistics.holm(padded)
    lines += ["", "Comparisons (difference in success rate, 95% interval, corrected p):"]
    for (first, second), result, corrected in zip(ready, compared, adjusted, strict=False):
        verdict = "differs" if corrected < SIGNIFICANCE else "no difference found"
        lines.append(f"  {_describe(first, second, result)}  p={corrected:.3f}  {verdict}")
    if len(ready) < len(COMPARISONS):
        lines.append(f"  {len(COMPARISONS) - len(ready)} comparisons not run yet")
    return "\n".join(lines)


def _describe(first: str, second: str, result: statistics.Difference) -> str:
    """Formats one difference.

    Args:
        first: The condition.
        second: The condition it is compared against.
        result: The difference between them.

    Returns:
        The difference and its interval, in percentage points.
    """
    return (
        f"{first} vs {second}: {result.difference * 100:+.1f} points "
        f"({result.low * 100:+.1f} to {result.high * 100:+.1f})"
    )
