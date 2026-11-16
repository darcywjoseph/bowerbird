import json
import pathlib

import pytest

import bowerbird
from bowerbird import cli

pytest.importorskip("h5py")
pytest.importorskip("lerobot")
pytest.importorskip("robosuite")

from bowerbird.bench import robomimic, runs
from tests import test_bench_robomimic


@pytest.fixture
def root(tmp_path: pathlib.Path) -> pathlib.Path:
    source = tmp_path / "source.hdf5"
    test_bench_robomimic._write_source(source)
    return robomimic.convert(source, tmp_path / "lift", "lift")


def test_choose_applies_each_rule(root: pathlib.Path) -> None:
    # Episode lengths are 12, 9, 7 and 15.
    assert runs.choose(root, "all", 0.5, seed=0) == [0, 1, 2, 3]
    assert runs.choose(root, "length", 0.5, seed=0) == [1, 2]

    first = runs.choose(root, "random", 0.5, seed=0)
    assert len(first) == 2
    assert runs.choose(root, "random", 0.5, seed=0) == first
    assert {tuple(runs.choose(root, "random", 0.5, seed=seed)) for seed in range(20)} != {
        tuple(first)
    }

    # The better operator recorded episodes 0 and 3, which is half.
    assert runs.choose(root, "operator", 0.5, seed=0) == [0, 3]
    with pytest.raises(bowerbird.ScorerError, match="keeps 2 of 4"):
        runs.choose(root, "operator", 0.75, seed=0)
    with pytest.raises(bowerbird.ScorerError, match="Unknown rule"):
        runs.choose(root, "best", 0.5, seed=0)


def test_a_finished_run_is_not_repeated(root: pathlib.Path, tmp_path: pathlib.Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / runs.RESULT_FILE).write_text(json.dumps({"success_rate": 0.25}))

    result = runs.run(root, output, rule="length", fraction=0.5, seed=0, steps=10, attempts=2)

    assert result == {"success_rate": 0.25}


def test_bench_command_reports_a_bad_rule(
    root: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(
        ["bench", "run", str(root), str(tmp_path / "run"), "--rule", "best", "--steps", "1"]
    )

    assert status == 1
    assert "Unknown rule" in capsys.readouterr().err
