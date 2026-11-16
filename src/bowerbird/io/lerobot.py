"""Reader for LeRobot v3 datasets, from a local folder or the Hugging Face Hub.

The format keeps per-frame numbers in Parquet files under ``data/``, video in
MP4 files under ``videos/``, and ties them together with ``meta/info.json``
and one metadata row per episode under ``meta/episodes/``. This module reads
those files directly, so it does not need LeRobot or PyTorch installed.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import errors as hub_errors

from bowerbird import errors
from bowerbird.core import episodes

_INFO_PATH = "meta/info.json"
_EPISODES_GLOB = "meta/episodes/*/*.parquet"
_SUPPORTED_VERSION = "v3."
_REPO_ID_PATTERN = re.compile(r"[\w.-]+/[\w.-]+")


class Dataset:
    """A LeRobot v3 dataset whose episodes are read on demand.

    Attributes:
        root: The folder that holds the dataset's ``meta`` and ``data``.
        name: The name the dataset was loaded by.
        info: The contents of ``meta/info.json``.
        scores_path: Where scores for the dataset are kept.
    """

    def __init__(
        self,
        root: Path,
        *,
        name: str | None = None,
        fetch: Callable[[str], Path] | None = None,
        fingerprint: str | None = None,
        scores_path: Path | None = None,
    ) -> None:
        """Opens a dataset folder and reads its metadata.

        Args:
            root: The folder that holds the dataset's ``meta`` and ``data``.
            name: The name to show for the dataset. Defaults to the folder.
            fetch: Turns a path relative to the root into a file on disk.
              Used for videos, which a Hub dataset downloads only when asked.
              Defaults to looking inside ``root``.
            fingerprint: An identifier for this version of the dataset.
              Defaults to a hash of the metadata files.
            scores_path: Where scores for the dataset are kept. Defaults to
              ``bowerbird/scores.parquet`` inside ``root``.

        Raises:
            DatasetError: The folder is not a LeRobot v3 dataset.
        """
        self.root = root
        self.name = name or str(root)
        self._fetch = fetch or root.joinpath
        self._fingerprint = fingerprint
        self.scores_path = scores_path or root / "bowerbird" / "scores.parquet"
        self.info = _read_info(root)
        self._rows = _read_episode_rows(root)
        self._names = {
            key: _dimension_names(feature.get("names"))
            for key, feature in self.info.get("features", {}).items()
        }
        self._cached_file: str | None = None
        self._cached_table: pa.Table | None = None
        self._cached_bounds: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[episodes.Episode]:
        for index in self._rows:
            yield self[index]

    def __getitem__(self, index: int) -> episodes.Episode:
        if index not in self._rows:
            raise IndexError(
                f"{self.name} has no episode {index}. It has {len(self)} episodes, "
                f"numbered {min(self._rows, default=0)} to {max(self._rows, default=0)}."
            )
        row = self._rows[index]
        return episodes.Episode(
            index=index,
            length=int(row["length"]),
            fps=self.fps,
            tasks=tuple(row.get("tasks") or ()),
            videos={camera: self._clip(camera, row) for camera in self.cameras},
            load_frames=lambda: self._frames(row),
            names=self._names,
        )

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, episodes={len(self)})"

    @property
    def fps(self) -> float:
        """Frames per second."""
        return float(self.info["fps"])

    @property
    def cameras(self) -> tuple[str, ...]:
        """The feature names of the dataset's video streams."""
        features: dict[str, Any] = self.info.get("features", {})
        return tuple(key for key, feature in features.items() if feature.get("dtype") == "video")

    @property
    def fingerprint(self) -> str:
        """An identifier that changes when the dataset's contents change."""
        if self._fingerprint is None:
            digest = hashlib.sha256()
            for path in [self.root / _INFO_PATH, *sorted(self.root.glob(_EPISODES_GLOB))]:
                digest.update(path.read_bytes())
            self._fingerprint = digest.hexdigest()[:16]
        return self._fingerprint

    def _clip(self, camera: str, row: dict[str, Any]) -> episodes.VideoClip:
        """Locates one episode's clip from one camera.

        Args:
            camera: The camera's feature name.
            row: The episode's metadata row.

        Returns:
            The clip's file and time range.
        """
        relative_path = self.info["video_path"].format(
            video_key=camera,
            chunk_index=row[f"videos/{camera}/chunk_index"],
            file_index=row[f"videos/{camera}/file_index"],
        )
        return episodes.VideoClip(
            camera=camera,
            start_s=float(row[f"videos/{camera}/from_timestamp"]),
            end_s=float(row[f"videos/{camera}/to_timestamp"]),
            relative_path=relative_path,
            _resolve=self._fetch,
        )

    def _frames(self, row: dict[str, Any]) -> pa.Table:
        """Reads one episode's rows from its data file.

        The most recent data file stays in memory, because consecutive
        episodes usually share a file.

        Args:
            row: The episode's metadata row.

        Returns:
            The episode's rows, one per frame.

        Raises:
            DatasetError: The data file is missing or lacks the episode.
        """
        relative_path = self.info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        if relative_path != self._cached_file:
            path = self.root / relative_path
            if not path.is_file():
                raise errors.DatasetError(
                    f"{self.name} is missing the data file {relative_path}. "
                    "The dataset is incomplete; download or copy it again."
                )
            table = pq.read_table(path, memory_map=True)
            self._cached_bounds = _episode_bounds(table, relative_path)
            self._cached_table = table
            self._cached_file = relative_path
        index = int(row["episode_index"])
        if index not in self._cached_bounds or self._cached_table is None:
            raise errors.DatasetError(
                f"{self.name}: the metadata places episode {index} in {relative_path}, "
                "but that file has no rows for it. The dataset is corrupted; "
                "download or re-export it."
            )
        start, stop = self._cached_bounds[index]
        return self._cached_table.slice(start, stop - start)


def load(source: str | Path, *, revision: str | None = None) -> Dataset:
    """Opens a LeRobot v3 dataset from a local folder or the Hugging Face Hub.

    A Hub dataset's metadata and per-frame numbers are downloaded to the
    Hugging Face cache. Its videos are downloaded only when a clip is used.

    Args:
        source: A folder path, or a Hub dataset name such as "lerobot/pusht".
          A folder that exists takes priority over a Hub name.
        revision: The Hub branch, tag or commit to load. Ignored for folders.

    Returns:
        The opened dataset.

    Raises:
        DatasetError: The dataset was not found or is not LeRobot v3.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        return Dataset(path, name=str(source))
    if isinstance(source, str) and _REPO_ID_PATTERN.fullmatch(source):
        return _load_from_hub(source, revision)
    raise errors.DatasetError(
        f"No folder at {str(source)!r}, and it is not a Hub dataset name. "
        "Pass a folder that contains meta/info.json, or a name like 'lerobot/pusht'."
    )


def _load_from_hub(repo_id: str, revision: str | None) -> Dataset:
    """Downloads a Hub dataset's metadata and numbers, then opens it.

    Args:
        repo_id: The dataset's name on the Hub.
        revision: The branch, tag or commit to load, or None for the default.

    Returns:
        The opened dataset, pinned to the commit that was downloaded.

    Raises:
        DatasetError: The dataset or revision does not exist, or is private.
    """
    try:
        snapshot = huggingface_hub.snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["meta/*", "data/*"],
        )
    except hub_errors.RevisionNotFoundError as error:
        raise errors.DatasetError(
            f"The Hub dataset {repo_id!r} has no revision {revision!r}. "
            "Check the branch, tag or commit name."
        ) from error
    except hub_errors.RepositoryNotFoundError as error:
        raise errors.DatasetError(
            f"No folder or Hub dataset named {repo_id!r}. Check the spelling. "
            "If the dataset is private, run `hf auth login` first."
        ) from error
    root = Path(snapshot)
    commit = root.name

    def fetch(relative_path: str) -> Path:
        return Path(
            huggingface_hub.hf_hub_download(
                repo_id, relative_path, repo_type="dataset", revision=commit
            )
        )

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    scores_path = cache_home / "bowerbird" / repo_id.replace("/", "--") / commit / "scores.parquet"
    return Dataset(root, name=repo_id, fetch=fetch, fingerprint=commit, scores_path=scores_path)


def _read_info(root: Path) -> dict[str, Any]:
    """Reads and checks ``meta/info.json``.

    Args:
        root: The dataset folder.

    Returns:
        The parsed file.

    Raises:
        DatasetError: The file is missing or the format version is not v3.
    """
    path = root / _INFO_PATH
    if not path.is_file():
        raise errors.DatasetError(
            f"{root} is not a LeRobot dataset: {_INFO_PATH} is missing. "
            "Pass the folder that contains the dataset's meta/ and data/ folders."
        )
    info: dict[str, Any] = json.loads(path.read_text())
    version = str(info.get("codebase_version", "unknown"))
    if not version.startswith(_SUPPORTED_VERSION):
        raise errors.DatasetError(
            f"{root} is a LeRobot {version} dataset, and bowerbird reads v3.0. "
            "Convert it with LeRobot's dataset conversion script, then try again."
        )
    return info


def _dimension_names(names: Any) -> tuple[str, ...]:
    """Flattens the ``names`` entry of a feature in ``meta/info.json``.

    Args:
        names: A list of names, a dict of lists such as ``{"motors": [...]}``,
          or None.

    Returns:
        One name per dimension, or an empty tuple if the names are not given.
    """
    if isinstance(names, dict):
        return tuple(str(name) for group in names.values() for name in group or ())
    if isinstance(names, list):
        return tuple(str(name) for name in names)
    return ()


def _read_episode_rows(root: Path) -> dict[int, dict[str, Any]]:
    """Reads the per-episode metadata, leaving out the bulky statistics.

    Args:
        root: The dataset folder.

    Returns:
        Each episode's metadata row, keyed and ordered by episode index.

    Raises:
        DatasetError: The dataset has no episode metadata.
    """
    paths = sorted(root.glob(_EPISODES_GLOB))
    if not paths:
        raise errors.DatasetError(
            f"{root} has no episode metadata under meta/episodes/. "
            "The dataset is incomplete; download or copy it again."
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        names = pq.read_schema(path).names
        wanted = [name for name in names if not name.startswith("stats/")]
        rows.extend(pq.read_table(path, columns=wanted).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return {int(row["episode_index"]): row for row in rows}


def _episode_bounds(table: pa.Table, relative_path: str) -> dict[int, tuple[int, int]]:
    """Finds where each episode's rows start and stop in a data file.

    Args:
        table: The data file's contents.
        relative_path: The file's path, for error messages.

    Returns:
        Each episode's half-open row range, keyed by episode index.

    Raises:
        DatasetError: An episode's rows are not contiguous.
    """
    indices = table.column("episode_index").to_numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(indices)) + 1))
    stops = np.concatenate((starts[1:], [len(indices)]))
    bounds: dict[int, tuple[int, int]] = {}
    if not len(indices):
        return bounds
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        index = int(indices[start])
        if index in bounds:
            raise errors.DatasetError(
                f"{relative_path} stores episode {index} in more than one block of rows. "
                "The dataset is corrupted; re-export it."
            )
        bounds[index] = (start, stop)
    return bounds
