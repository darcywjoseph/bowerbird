"""One demonstration and the video clips that belong to it."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class VideoClip:
    """The part of a video file that shows one episode from one camera.

    Attributes:
        camera: The camera's feature name, such as "observation.images.front".
        start_s: Where the episode starts in the file, in seconds.
        end_s: Where the episode ends in the file, in seconds.
        relative_path: The file's path relative to the dataset root.
    """

    camera: str
    start_s: float
    end_s: float
    relative_path: str
    _resolve: Callable[[str], Path] = field(repr=False, compare=False)

    def local_path(self) -> Path:
        """Returns the video file's path on disk, downloading it if needed."""
        return self._resolve(self.relative_path)


class Episode:
    """One demonstration, with its frames read from disk on first use.

    Attributes:
        index: The episode's index in the dataset.
        length: The number of frames, as recorded in the dataset's metadata.
        fps: Frames per second.
        tasks: The task descriptions attached to the episode.
    """

    def __init__(
        self,
        index: int,
        length: int,
        fps: float,
        tasks: tuple[str, ...],
        videos: Mapping[str, VideoClip],
        load_frames: Callable[[], pa.Table],
    ) -> None:
        """Creates an episode.

        Args:
            index: The episode's index in the dataset.
            length: The number of frames.
            fps: Frames per second.
            tasks: The task descriptions attached to the episode.
            videos: The episode's video clip for each camera.
            load_frames: Returns the episode's rows, one per frame.
        """
        self.index = index
        self.length = length
        self.fps = fps
        self.tasks = tasks
        self._videos = dict(videos)
        self._load_frames = load_frames
        self._frames: pa.Table | None = None

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return f"Episode(index={self.index}, length={self.length})"

    @property
    def frames(self) -> pa.Table:
        """The episode's rows as an Arrow table, one row per frame."""
        if self._frames is None:
            self._frames = self._load_frames()
        return self._frames

    @property
    def columns(self) -> tuple[str, ...]:
        """The names of the per-frame columns."""
        return tuple(self.frames.column_names)

    def column(self, name: str) -> npt.NDArray[Any]:
        """Reads one per-frame column.

        Args:
            name: The column name, such as "action".

        Returns:
            An array whose first axis is the frame.

        Raises:
            KeyError: The episode has no column with that name.
        """
        if name not in self.frames.column_names:
            raise KeyError(
                f"Episode {self.index} has no column {name!r}. "
                f"Available columns: {', '.join(self.columns)}."
            )
        return _to_numpy(self.frames.column(name))

    @property
    def states(self) -> npt.NDArray[Any]:
        """The robot state at each frame, shaped (frames, state size)."""
        return self.column(STATE_COLUMN)

    @property
    def actions(self) -> npt.NDArray[Any]:
        """The commanded action at each frame, shaped (frames, action size)."""
        return self.column(ACTION_COLUMN)

    @property
    def timestamps(self) -> npt.NDArray[Any]:
        """The time of each frame in seconds from the episode's start."""
        return self.column(TIMESTAMP_COLUMN)

    @property
    def cameras(self) -> tuple[str, ...]:
        """The names of the cameras that recorded the episode."""
        return tuple(self._videos)

    def video(self, camera: str) -> VideoClip:
        """Returns the episode's clip from one camera.

        Args:
            camera: The camera's feature name.

        Returns:
            The clip, which locates the episode inside a video file.

        Raises:
            KeyError: The episode has no camera with that name.
        """
        if camera not in self._videos:
            raise KeyError(
                f"Episode {self.index} has no camera {camera!r}. "
                f"Available cameras: {', '.join(self.cameras) or 'none'}."
            )
        return self._videos[camera]


def _to_numpy(column: pa.ChunkedArray) -> npt.NDArray[Any]:
    """Converts an Arrow column to a NumPy array.

    Args:
        column: A column of scalars or of equal-length lists.

    Returns:
        A 1-D array for scalars, or a 2-D array for lists of numbers.
    """
    array = column.combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        array = pa.concat_arrays(array.chunks) if array.num_chunks else pa.array([], column.type)
    is_list = (
        pa.types.is_fixed_size_list(array.type)
        or pa.types.is_list(array.type)
        or pa.types.is_large_list(array.type)
    )
    if is_list and pa.types.is_primitive(array.type.value_type):
        flat: npt.NDArray[Any] = array.flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(len(array), -1)
    if is_list:
        return np.asarray(array.to_pylist())
    values: npt.NDArray[Any] = array.to_numpy(zero_copy_only=False)
    return values
