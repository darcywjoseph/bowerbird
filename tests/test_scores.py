from pathlib import Path

import pytest

import bowerbird


def test_score_table_round_trips_through_parquet(tmp_path: Path) -> None:
    table = bowerbird.ScoreTable(dataset_fingerprint="abc123")
    stalled = bowerbird.Segment(10, 50, 0.1, "stalled 4 s before grasp")
    table.add("motion", "1", [bowerbird.EpisodeScore(0, 0.4, (stalled,))])
    table.add("motion", "1", [bowerbird.EpisodeScore(1, 0.9)])
    table.add("length", "2", [bowerbird.EpisodeScore(0, 0.7)])

    path = tmp_path / "nested" / "scores.parquet"
    table.save(path)
    loaded = bowerbird.ScoreTable.load(path)

    assert loaded.dataset_fingerprint == "abc123"
    assert loaded.scorers == ("motion", "length")
    assert len(loaded) == 3
    assert loaded.version("length") == "2"
    assert loaded.get("motion") == table.get("motion")
    assert loaded.get("motion")[0].segments == (stalled,)


def test_new_scorer_version_drops_old_scores() -> None:
    table = bowerbird.ScoreTable()
    table.add("motion", "1", [bowerbird.EpisodeScore(0, 0.4), bowerbird.EpisodeScore(1, 0.9)])

    table.add("motion", "2", [bowerbird.EpisodeScore(0, 0.5)])

    assert table.get("motion") == {0: bowerbird.EpisodeScore(0, 0.5)}
    assert table.version("motion") == "2"


def test_unknown_scorer_says_what_is_available() -> None:
    table = bowerbird.ScoreTable()
    table.add("motion", "1", [])

    with pytest.raises(KeyError, match="motion"):
        table.get("progress")
