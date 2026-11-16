from pathlib import Path

import numpy as np
import pytest

import bowerbird
from tests import conftest


def test_iterates_every_episode_with_its_own_rows(dataset_root: Path) -> None:
    dataset = bowerbird.load(dataset_root)

    assert len(dataset) == len(conftest.LENGTHS)
    assert dataset.fps == conftest.FPS
    assert dataset.cameras == (conftest.CAMERA,)
    for expected_index, (episode, length) in enumerate(zip(dataset, conftest.LENGTHS, strict=True)):
        assert episode.index == expected_index
        assert len(episode) == length
        assert episode.tasks == (conftest.TASK,)
        assert episode.states.shape == (length, conftest.STATE_SIZE)
        assert episode.states.dtype == np.float32
        assert episode.actions.shape == (length, conftest.ACTION_SIZE)
        np.testing.assert_array_equal(episode.states[:, 0], expected_index)
        np.testing.assert_array_equal(episode.states[:, 1], np.arange(length))
        np.testing.assert_allclose(episode.timestamps, np.arange(length) / conftest.FPS)


def test_reads_episodes_out_of_order_across_data_files(dataset_root: Path) -> None:
    dataset = bowerbird.load(dataset_root)

    for index in (4, 0, 3, 1):
        np.testing.assert_array_equal(dataset[index].states[:, 0], index)


def test_video_clip_locates_the_episode_in_its_file(dataset_root: Path) -> None:
    episode = bowerbird.load(dataset_root)[1]

    clip = episode.video(conftest.CAMERA)

    assert episode.cameras == (conftest.CAMERA,)
    assert clip.start_s == pytest.approx(0.5)
    assert clip.end_s == pytest.approx(0.8)
    assert clip.relative_path == f"videos/{conftest.CAMERA}/chunk-000/file-000.mp4"
    assert clip.local_path() == dataset_root / clip.relative_path


def test_unknown_names_say_what_is_available(dataset_root: Path) -> None:
    dataset = bowerbird.load(dataset_root)

    with pytest.raises(IndexError, match="numbered 0 to 4"):
        dataset[99]
    with pytest.raises(KeyError, match=r"observation\.state"):
        dataset[0].column("nope")
    with pytest.raises(KeyError, match=conftest.CAMERA):
        dataset[0].video("nope")


def test_fingerprint_changes_with_the_dataset(tmp_path: Path) -> None:
    first = conftest.write_dataset(tmp_path / "a")
    same = conftest.write_dataset(tmp_path / "b")
    different = conftest.write_dataset(tmp_path / "c", lengths=(5, 3))

    assert bowerbird.load(first).fingerprint == bowerbird.load(same).fingerprint
    assert bowerbird.load(first).fingerprint != bowerbird.load(different).fingerprint


def test_missing_folder_is_explained(tmp_path: Path) -> None:
    with pytest.raises(bowerbird.DatasetError, match="No folder"):
        bowerbird.load(tmp_path / "missing")


def test_folder_without_metadata_is_explained(tmp_path: Path) -> None:
    with pytest.raises(bowerbird.DatasetError, match=r"meta/info\.json is missing"):
        bowerbird.load(tmp_path)


def test_older_format_is_explained(tmp_path: Path) -> None:
    root = conftest.write_dataset(tmp_path / "old", codebase_version="v2.1")

    with pytest.raises(bowerbird.DatasetError, match=r"v2\.1.*Convert"):
        bowerbird.load(root)


def test_missing_data_file_is_explained(dataset_root: Path) -> None:
    (dataset_root / "data/chunk-000/file-001.parquet").unlink()
    dataset = bowerbird.load(dataset_root)

    assert dataset[0].states.shape[0] == conftest.LENGTHS[0]
    with pytest.raises(bowerbird.DatasetError, match="missing the data file"):
        _ = dataset[2].states


@pytest.mark.network
def test_loads_pusht_from_the_hub() -> None:
    dataset = bowerbird.load("lerobot/pusht")

    assert len(dataset) == dataset.info["total_episodes"]
    total_frames = 0
    for episode in dataset:
        assert episode.states.shape == (len(episode), 2)
        assert episode.actions.shape == (len(episode), 2)
        total_frames += len(episode)
    assert total_frames == dataset.info["total_frames"]


@pytest.mark.network
def test_unknown_hub_dataset_is_explained() -> None:
    with pytest.raises(bowerbird.DatasetError, match="Check the spelling"):
        bowerbird.load("bowerbird-tests/does-not-exist")
