import json
import pathlib

import numpy as np
import pytest

import bowerbird
from bowerbird.bench import statistics

pytest.importorskip("h5py")
pytest.importorskip("lerobot")
pytest.importorskip("robosuite")

from bowerbird.bench import rounds, runs


def _attempts(rate: float, seeds: int = 5, layouts: int = 200, seed: int = 0) -> list[list[bool]]:
    generator = np.random.default_rng(seed)
    grid: list[list[bool]] = (generator.random((seeds, layouts)) < rate).tolist()
    return grid


def test_compare_finds_a_real_difference_and_not_a_false_one() -> None:
    better = statistics.compare(_attempts(0.8, seed=1), _attempts(0.6, seed=2))
    same = statistics.compare(_attempts(0.7, seed=3), _attempts(0.7, seed=4))

    assert 0.1 < better.low < better.difference < better.high < 0.3
    assert better.p_value < 0.001
    assert same.low < 0 < same.high
    assert same.p_value > 0.05


def test_compare_widens_when_seeds_disagree() -> None:
    steady = [[True] * 70 + [False] * 30] * 4
    erratic = [[True] * 100, [True] * 100, [True] * 40 + [False] * 60, [True] * 40 + [False] * 60]
    baseline = [[True] * 50 + [False] * 50] * 4

    narrow = statistics.compare(steady, baseline)
    wide = statistics.compare(erratic, baseline)

    assert narrow.difference == pytest.approx(wide.difference)
    assert wide.high - wide.low > narrow.high - narrow.low


def test_compare_is_repeatable_and_rejects_mismatched_layouts() -> None:
    first, second = _attempts(0.7, seed=1), _attempts(0.6, seed=2)

    assert statistics.compare(first, second) == statistics.compare(first, second)
    with pytest.raises(ValueError, match="layouts"):
        statistics.compare(first, [row[:50] for row in second])


def test_holm_adjusts_the_smallest_most_and_keeps_order() -> None:
    assert statistics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert statistics.holm([0.5, 0.9]) == [1.0, 1.0]
    assert statistics.holm([]) == []


def test_every_comparison_names_a_condition_of_the_round() -> None:
    names = {condition.name for condition in rounds.CONDITIONS}

    assert len(names) == len(rounds.CONDITIONS) == 12
    assert {name for pair in rounds.COMPARISONS for name in pair} <= names
    assert {name for first, second, _ in rounds.CHECKS for name in (first, second)} <= names
    assert {condition.rule for condition in rounds.CONDITIONS} <= set(runs.RULES)


def test_report_writes_up_whatever_has_finished(tmp_path: pathlib.Path) -> None:
    rates = {"random-50": 0.5, "length-50": 0.7, "operator-50": 0.8, "all": 0.75}
    for number, (condition, rate) in enumerate(rates.items()):
        for seed, row in enumerate(_attempts(rate, seed=number)):
            folder = rounds.run_folder(tmp_path, "square", condition, seed)
            folder.mkdir(parents=True)
            (folder / runs.RESULT_FILE).write_text(json.dumps({"successes": row}))

    text = rounds.report(tmp_path, "square")

    assert "length-50        5" in text
    assert "keeping the better operators helps: operator-50 vs random-50" in text
    assert "holds" in text
    assert "the faults hurt: not run yet" in text
    assert "length-50 vs random-50: +" in text
    assert "differs" in text
    assert "11 comparisons not run yet" in text
    with pytest.raises(bowerbird.ScorerError, match="No finished runs"):
        rounds.report(tmp_path, "can")
