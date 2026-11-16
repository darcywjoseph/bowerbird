"""Decodes video clips: how much the picture changes, and small stills from it."""

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import av
import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes

_THUMBNAIL_SIZE = 64


def frame_changes(clip: episodes.VideoClip, fps: float) -> npt.NDArray[Any]:
    """Measures how much the picture changes from each frame to the next.

    Frames are shrunk to small greyscale thumbnails first, which is enough to
    spot a frozen camera and keeps decoding the only real cost.

    Args:
        clip: The part of a video file to decode.
        fps: The video's frames per second.

    Returns:
        One value per decoded frame: the mean absolute difference from the
        previous frame, in grey levels from 0 to 255. The first value is NaN.
        The array's length is the number of frames in the clip.

    Raises:
        OSError: The file is missing or cannot be decoded.
    """
    # A frame belongs to the clip if its timestamp is nearer to the clip than
    # to the neighbouring episode, hence the half-frame margin.
    margin = 0.5 / fps
    changes: list[float] = []
    previous: npt.NDArray[Any] | None = None
    try:
        with av.open(str(clip.local_path())) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if clip.start_s > 0 and stream.time_base is not None:
                container.seek(int(clip.start_s / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None or frame.time < clip.start_s - margin:
                    continue
                if frame.time >= clip.end_s - margin:
                    break
                thumbnail = frame.reformat(
                    width=_THUMBNAIL_SIZE, height=_THUMBNAIL_SIZE, format="gray"
                ).to_ndarray()
                current = thumbnail.astype(np.float32)
                changes.append(
                    float("nan") if previous is None else float(np.abs(current - previous).mean())
                )
                previous = current
    except av.FFmpegError as error:
        raise OSError(str(error)) from error
    return np.asarray(changes, dtype=np.float32)


def thumbnails(
    clip: episodes.VideoClip, fps: float, frames: Iterable[int], height: int = 96
) -> dict[int, bytes]:
    """Extracts small JPEG stills of chosen frames from a clip.

    Args:
        clip: The part of a video file to decode.
        fps: The video's frames per second.
        frames: The frames to extract, counted from the clip's start.
        height: The height of each still in pixels.

    Returns:
        JPEG bytes for each requested frame that the clip contains.

    Raises:
        OSError: The file is missing or cannot be decoded.
    """
    wanted = sorted(set(frames))
    stills: dict[int, bytes] = {}
    if not wanted:
        return stills
    try:
        with av.open(str(clip.local_path())) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            first_s = clip.start_s + wanted[0] / fps
            if first_s > 0 and stream.time_base is not None:
                container.seek(int(first_s / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                index = round((frame.time - clip.start_s) * fps)
                if index > wanted[-1] or frame.time >= clip.end_s - 0.5 / fps:
                    break
                if index in wanted and index not in stills:
                    stills[index] = _jpeg(frame, height)
    except av.FFmpegError as error:
        raise OSError(str(error)) from error
    return stills


def _jpeg(frame: av.VideoFrame, height: int) -> bytes:
    """Encodes a frame as a JPEG of a given height.

    Args:
        frame: The decoded frame.
        height: The height of the result in pixels.

    Returns:
        The JPEG file's bytes.
    """
    # JPEG wants even dimensions.
    width = max(2, round(frame.width * height / frame.height / 2) * 2)
    small = frame.reformat(width=width, height=height, format="yuvj420p")
    small.pts = None
    # Typed as Any because older PyAV releases do not type the video encoder.
    encoder: Any = av.CodecContext.create("mjpeg", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuvj420p"
    encoder.time_base = Fraction(1, 1)
    packets = [*encoder.encode(small), *encoder.encode(None)]
    return b"".join(bytes(packet) for packet in packets)
