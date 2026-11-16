"""Runs scorers over a dataset and keeps the results beside it."""

import time
from collections.abc import Callable, Sequence

from bowerbird import scorers as scorer_registry
from bowerbird.core import scores
from bowerbird.io import lerobot


def score(
    dataset: lerobot.Dataset,
    scorers: Sequence[str | scores.Scorer] | None = None,
    *,
    check_video: bool = True,
    force: bool = False,
    save: bool = True,
    log: Callable[[str], None] | None = None,
) -> scores.ScoreTable:
    """Scores a dataset, reusing stored scores that are still valid.

    A scorer is skipped when the stored table already has its scores for this
    version of the dataset, from this version of the scorer, for every episode.

    Args:
        dataset: The dataset to score.
        scorers: Scorer names or scorer objects. Defaults to the scorers that
          need no GPU, apart from the random control.
        check_video: Whether the health scorer should decode the videos.
        force: Whether to recompute scores that are already stored.
        save: Whether to read and write the table at `dataset.scores_path`.
        log: Called with one line of progress per scorer.

    Returns:
        The table, holding the requested scorers and any stored earlier.

    Raises:
        ScorerError: A scorer name is unknown or a scorer cannot run.
    """
    chosen = [
        scorer_registry.create(scorer, check_video=check_video)
        if isinstance(scorer, str)
        else scorer
        for scorer in (scorers if scorers is not None else scorer_registry.DEFAULT)
    ]
    table = scores.ScoreTable(dataset.fingerprint)
    if save and dataset.scores_path.is_file():
        stored = scores.ScoreTable.load(dataset.scores_path)
        if stored.dataset_fingerprint == dataset.fingerprint:
            table = stored

    indices = {episode.index for episode in dataset}
    changed = False
    for scorer in chosen:
        stored_here = scorer.name in table.scorers and table.version(scorer.name) == scorer.version
        if not force and stored_here and set(table.get(scorer.name)) == indices:
            if log:
                log(f"{scorer.name:<10} already scored")
            continue
        started = time.perf_counter()
        table.add(scorer.name, scorer.version, list(scorer.score(dataset)), replace=True)
        changed = True
        if log:
            log(f"{scorer.name:<10} {time.perf_counter() - started:.1f} s")
    if save and changed:
        table.save(dataset.scores_path)
    return table
