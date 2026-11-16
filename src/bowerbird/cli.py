"""The ``bowerbird`` command."""

import argparse
import pathlib
import sys
from collections.abc import Sequence

from bowerbird import errors, scoring
from bowerbird import scorers as scorer_registry
from bowerbird.core import scores
from bowerbird.io import lerobot
from bowerbird.report import html as html_report

_DEFAULT_REPORT = "bowerbird-report.html"
_REASON_WIDTH = 60


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: The arguments, without the program name. Defaults to the
          process's arguments.

    Returns:
        The exit status: 0 on success, 1 on an error the user can act on.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "score":
            _score(arguments)
        elif arguments.command == "show":
            _show(arguments)
        else:
            _bench(arguments)
    except (errors.DatasetError, errors.ScorerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bowerbird", description="Score robot demonstrations and see which ones are weak."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score a dataset and write an HTML report")
    _add_source(score)
    score.add_argument(
        "--scorers",
        default=",".join(scorer_registry.DEFAULT),
        help=f"comma-separated scorers to run, from: {', '.join(scorer_registry.NAMES)} "
        "(default: %(default)s)",
    )
    score.add_argument(
        "--no-video",
        action="store_true",
        help="skip the camera checks, which decode (and for a Hub dataset, download) the videos",
    )
    score.add_argument("--force", action="store_true", help="recompute scores already stored")
    score.add_argument(
        "--report", default=_DEFAULT_REPORT, help="where to write the report (default: %(default)s)"
    )

    show = commands.add_parser("show", help="list the lowest-scoring episodes and why")
    _add_source(show)
    show.add_argument("--worst", type=int, default=20, help="how many episodes to list")

    bench = commands.add_parser(
        "bench", help="train on curated data and test the result (needs bowerbird[bench])"
    )
    steps = bench.add_subparsers(dest="step", required=True)
    prepare = steps.add_parser("prepare", help="download a robomimic task as a LeRobot dataset")
    prepare.add_argument("task", help="lift, can or square")
    prepare.add_argument("data_set", metavar="set", help="mixed or skilled")
    prepare.add_argument("folder", help="where to keep the download and the dataset")
    damage = steps.add_parser("damage", help="write a copy of a dataset with known faults")
    damage.add_argument("dataset", help="a dataset folder made by `bowerbird bench prepare`")
    damage.add_argument("output", help="the folder for the damaged copy")
    damage.add_argument("--share", type=float, default=0.2, help="the share of episodes to damage")
    damage.add_argument("--seed", type=int, default=0, help="chooses the episodes and the faults")
    detect = steps.add_parser("detect", help="measure how many known faults the scorers find")
    detect.add_argument("dataset", help="a dataset folder made by `bowerbird bench damage`")
    run = steps.add_parser("run", help="curate by one rule, train a policy and test it")
    run.add_argument("dataset", help="a dataset folder made by `bowerbird bench prepare`")
    run.add_argument("output", help="the folder for this run's checkpoints and result")
    run.add_argument(
        "--rule", required=True, help="all, random, length, motion, operator, curated or oracle"
    )
    run.add_argument("--fraction", type=float, default=0.5, help="the share of episodes to keep")
    run.add_argument("--seed", type=int, default=0, help="the training seed")
    run.add_argument("--steps", type=int, required=True, help="the number of training steps")
    run.add_argument("--attempts", type=int, default=200, help="how many test attempts to run")
    run.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    whole = steps.add_parser("round", help="run every condition of the protocol for one task")
    whole.add_argument("folder", help="the folder for the datasets and runs")
    whole.add_argument("task", help="lift, can or square")
    whole.add_argument("--steps", type=int, required=True, help="training steps per run")
    whole.add_argument("--seeds", type=int, default=5, help="how many training seeds")
    whole.add_argument("--attempts", type=int, default=200, help="test attempts per run")
    whole.add_argument(
        "--shard", default="0/1", help="this machine's share of the runs, as number/machines"
    )
    whole.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    report = steps.add_parser("report", help="write up the finished runs of one task")
    report.add_argument("folder", help="the folder given to `bowerbird bench round`")
    report.add_argument("task", help="lift, can or square")
    return parser


def _add_source(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments that pick a dataset.

    Args:
        parser: The subcommand's parser.
    """
    parser.add_argument("source", help="a dataset folder, or a Hub name such as lerobot/pusht")
    parser.add_argument("--revision", help="the Hub branch, tag or commit to use")


def _score(arguments: argparse.Namespace) -> None:
    """Runs ``bowerbird score``.

    Args:
        arguments: The parsed arguments.
    """
    dataset = lerobot.load(arguments.source, revision=arguments.revision)
    frames = sum(episode.length for episode in dataset)
    print(f"Scoring {dataset.name}: {len(dataset)} episodes, {frames} frames")
    names = [name.strip() for name in arguments.scorers.split(",") if name.strip()]
    table = scoring.score(
        dataset,
        names,
        check_video=not arguments.no_video,
        force=arguments.force,
        log=lambda line: print(f"  {line}"),
    )
    report = html_report.write(dataset, table, arguments.report, check_video=not arguments.no_video)
    print(f"Scores: {dataset.scores_path}")
    print(f"Report: {report}")
    print()
    _print_worst(table, 5)


def _show(arguments: argparse.Namespace) -> None:
    """Runs ``bowerbird show``.

    Args:
        arguments: The parsed arguments.

    Raises:
        DatasetError: The dataset has no stored scores for its current version.
    """
    dataset = lerobot.load(arguments.source, revision=arguments.revision)
    table = None
    if dataset.scores_path.is_file():
        table = scores.ScoreTable.load(dataset.scores_path)
    if table is None or table.dataset_fingerprint != dataset.fingerprint or not table.scorers:
        raise errors.DatasetError(
            f"{dataset.name} has no scores yet. Run `bowerbird score {arguments.source}` first."
        )
    _print_worst(table, arguments.worst)


def _bench(arguments: argparse.Namespace) -> None:
    """Runs ``bowerbird bench``.

    Args:
        arguments: The parsed arguments.

    Raises:
        ScorerError: The benchmark's extra dependencies are not installed.
    """
    try:
        from bowerbird.bench import detection, faults, robomimic, rounds, runs
    except ImportError as error:
        raise errors.ScorerError(
            f"The benchmark needs extra packages ({error.name} is missing). "
            "Install them with: pip install 'bowerbird[bench]' (Python 3.12 or newer)."
        ) from error

    if arguments.step == "prepare":
        folder = pathlib.Path(arguments.folder)
        source = robomimic.download(arguments.task, arguments.data_set, folder / "source")
        root = folder / f"{arguments.task}_{arguments.data_set}"
        robomimic.convert(source, root, arguments.task)
        print(f"Dataset: {root}")
        return
    if arguments.step == "damage":
        made = faults.damage(
            pathlib.Path(arguments.dataset),
            pathlib.Path(arguments.output),
            share=arguments.share,
            seed=arguments.seed,
        )
        print(f"Damaged {len(made)} episodes. Dataset: {arguments.output}")
        return
    if arguments.step == "detect":
        found = detection.measure(pathlib.Path(arguments.dataset))
        print(f"{'fault':16} {'episodes':>8} {'named':>6} {'flagged':>8} {'separation':>11}")
        for row in found.rows:
            print(
                f"{row.kind:16} {row.episodes:>8} {row.named:>6.0%} {row.flagged:>8.0%} "
                f"{row.separation:>11.2f}"
            )
        print(f"All damaged episodes: separation {found.separation:.2f}")
        for name, share in found.false_alarms.items():
            print(f"{name} flags {share:.0%} of the {found.undamaged} undamaged episodes")
        return
    if arguments.step == "round":
        try:
            index, count = (int(part) for part in arguments.shard.split("/"))
        except ValueError:
            raise errors.ScorerError(
                f"Not a shard: {arguments.shard!r}. Write it as number/machines, such as 0/4."
            ) from None
        rounds.execute(
            pathlib.Path(arguments.folder),
            arguments.task,
            seeds=arguments.seeds,
            steps=arguments.steps,
            attempts=arguments.attempts,
            shard=(index, count),
            device=arguments.device,
            log=print,
        )
        return
    if arguments.step == "report":
        print(rounds.report(pathlib.Path(arguments.folder), arguments.task))
        return
    runs.run(
        pathlib.Path(arguments.dataset),
        pathlib.Path(arguments.output),
        rule=arguments.rule,
        fraction=arguments.fraction,
        seed=arguments.seed,
        steps=arguments.steps,
        attempts=arguments.attempts,
        device=arguments.device,
        log=print,
    )


def _print_worst(table: scores.ScoreTable, count: int) -> None:
    """Prints the lowest-scoring episodes as a table.

    Args:
        table: The dataset's scores.
        count: How many episodes to print.
    """
    worst = table.worst(count)
    print(f"Lowest-scoring {len(worst)} episodes:")
    header = f"{'episode':>7}  {'overall':>7}  "
    header += "  ".join(f"{name:>7}" for name in table.scorers)
    print(f"{header}  reasons")
    for summary in worst:
        cells = "  ".join(
            f"{summary.scores[name]:>7.2f}" if name in summary.scores else " " * 7
            for name in table.scorers
        )
        reasons = "; ".join(summary.reasons) or "-"
        if len(reasons) > _REASON_WIDTH:
            reasons = reasons[: _REASON_WIDTH - 3] + "..."
        print(f"{summary.index:>7}  {summary.score:>7.2f}  {cells}  {reasons}")


if __name__ == "__main__":
    sys.exit(main())
