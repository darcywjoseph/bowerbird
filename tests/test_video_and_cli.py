from pathlib import Path

import pytest

import bowerbird
from bowerbird import cli
from bowerbird.scorers import health
from tests import conftest


def test_health_finds_a_frozen_camera(tmp_path: Path) -> None:
    root = conftest.write_dataset(tmp_path / "d", lengths=(30, 30, 30), frozen={1: (8, 24)})

    scores = list(health.HealthScorer().score(bowerbird.load(root)))

    assert scores[0].score == 1.0, scores[0].segments
    assert scores[2].score == 1.0, scores[2].segments
    (segment,) = scores[1].segments
    assert "frozen for" in segment.reason
    assert conftest.CAMERA in segment.reason
    assert 8 <= segment.start_frame <= 10
    assert 23 <= segment.end_frame <= 25


def test_health_reports_an_unreadable_video(tmp_path: Path) -> None:
    root = conftest.write_dataset(tmp_path / "d", lengths=(30, 30))
    (root / f"videos/{conftest.CAMERA}/chunk-000/file-000.mp4").unlink()

    scores = list(health.HealthScorer().score(bowerbird.load(root)))

    assert scores[0].score == 0.0
    assert "could not be read" in scores[0].segments[0].reason


def test_score_then_show(
    dataset_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "out" / "report.html"

    assert cli.main(["show", str(dataset_root)]) == 1
    assert "bowerbird score" in capsys.readouterr().err

    assert cli.main(["score", str(dataset_root), "--report", str(report)]) == 0
    first = capsys.readouterr().out
    assert "5 episodes, 20 frames" in first
    assert (dataset_root / "bowerbird/scores.parquet").is_file()
    page = report.read_text()
    assert page.startswith("<!doctype html>")
    assert "http://" not in page and "https://" not in page, "the report must open offline"
    assert "Lowest-scoring episodes" in page

    assert cli.main(["score", str(dataset_root), "--report", str(report)]) == 0
    assert capsys.readouterr().out.count("already scored") == 3

    assert cli.main(["show", str(dataset_root), "--worst", "2"]) == 0
    assert "Lowest-scoring 2 episodes" in capsys.readouterr().out


def test_unknown_scorer_and_dataset_exit_with_guidance(
    dataset_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["score", str(dataset_root), "--scorers", "vibes"]) == 1
    assert "Choose from: health, length, motion, random" in capsys.readouterr().err

    assert cli.main(["score", str(tmp_path / "missing")]) == 1
    assert "No folder" in capsys.readouterr().err


def test_python_api_scores_and_selects(dataset_root: Path) -> None:
    dataset = bowerbird.load(dataset_root)

    table = bowerbird.score(dataset, ["length", "random"], save=False)

    assert not dataset.scores_path.exists()
    assert table.scorers == ("length", "random")
    worst = table.worst(2)
    assert len(worst) == 2
    assert worst[0].score <= worst[1].score
    keep = table.keep_episodes(0.6, scorers=["length"])
    assert keep == [1, 2, 4], "the three shortest episodes"
