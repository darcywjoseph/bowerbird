"""A tiny synthetic LeRobot v3 dataset, written fresh for each test."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FPS = 10
CAMERA = "observation.images.front"
TASK = "Put the cube in the bowl."
STATE_SIZE = 3
ACTION_SIZE = 2
LENGTHS = (5, 3, 4, 6, 2)


def write_dataset(
    root: Path,
    lengths: Sequence[int] = LENGTHS,
    *,
    episodes_per_file: int = 2,
    codebase_version: str = "v3.0",
) -> Path:
    """Writes a dataset whose values encode where they came from.

    State column 0 holds the episode index and column 1 the frame index, so a
    test can tell whether a reader returned the right rows.

    Args:
        root: The folder to create the dataset in.
        lengths: The number of frames in each episode.
        episodes_per_file: How many episodes share one data and video file.
        codebase_version: The format version to declare.

    Returns:
        The dataset folder.
    """
    info = {
        "codebase_version": codebase_version,
        "robot_type": "synthetic",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAMERA: {"dtype": "video", "shape": [4, 4, 3]},
            "observation.state": {"dtype": "float32", "shape": [STATE_SIZE]},
            "action": {"dtype": "float32", "shape": [ACTION_SIZE]},
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))

    episode_rows = []
    frames_by_file: dict[int, list[pa.Table]] = {}
    seconds_by_file: dict[int, float] = {}
    global_index = 0
    for episode_index, length in enumerate(lengths):
        file_index = episode_index // episodes_per_file
        video_start = seconds_by_file.get(file_index, 0.0)
        seconds_by_file[file_index] = video_start + length / FPS
        episode_rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": 0,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": file_index,
                f"videos/{CAMERA}/from_timestamp": video_start,
                f"videos/{CAMERA}/to_timestamp": seconds_by_file[file_index],
                "tasks": [TASK],
                "length": length,
                "stats/action/mean": [0.0] * ACTION_SIZE,
            }
        )
        frames_by_file.setdefault(file_index, []).append(
            _episode_frames(episode_index, length, global_index)
        )
        global_index += length

    _write_parquet(pa.Table.from_pylist(episode_rows), root / "meta/episodes/chunk-000/file-000")
    for file_index, tables in frames_by_file.items():
        _write_parquet(pa.concat_tables(tables), root / f"data/chunk-000/file-{file_index:03d}")
    return root


def _episode_frames(episode_index: int, length: int, global_index: int) -> pa.Table:
    frame_index = np.arange(length)
    state = np.zeros((length, STATE_SIZE), dtype=np.float32)
    state[:, 0] = episode_index
    state[:, 1] = frame_index
    action = np.full((length, ACTION_SIZE), episode_index + 0.5, dtype=np.float32)
    return pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(state.ravel(), STATE_SIZE),
            "action": pa.FixedSizeListArray.from_arrays(action.ravel(), ACTION_SIZE),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": frame_index.astype(np.int64),
            "timestamp": (frame_index / FPS).astype(np.float32),
            "index": (global_index + frame_index).astype(np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )


def _write_parquet(table: pa.Table, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path_without_suffix.with_suffix(".parquet"))


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "synthetic")
