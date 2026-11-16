"""Shows what a flagged segment looks like: camera stills and the signal behind it."""

import base64
import html
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import episodes, scores
from bowerbird.io import video

SEGMENTS_PER_EPISODE = 3
_STILLS = 6
_PLOT_WIDTH = 720
_PLOT_HEIGHT = 130
_PLOT_LEFT = 4
_PLOT_RIGHT = 210
_PLOT_TOP = 8
_PLOT_BOTTOM = 18
_LABEL_GAP = 13
_MAX_LINES = 3
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

Line = tuple[str, npt.NDArray[Any], bool]


def render(episode: episodes.Episode, summary: scores.EpisodeSummary, check_video: bool) -> str:
    """Builds the evidence for one episode's most significant flagged segments.

    Args:
        episode: The episode.
        summary: The episode's combined scores and segments.
        check_video: Whether to decode the videos for camera stills.

    Returns:
        The episode's section of the report, as HTML.
    """
    chosen = _choose(summary)
    stills: dict[str, dict[int, bytes]] = {}
    video_note = ""
    if check_video:
        wanted = {
            frame for _, segment in chosen for frame, _ in _still_frames(segment, len(episode))
        }
        for camera in episode.cameras:
            try:
                stills[camera] = video.thumbnails(episode.video(camera), episode.fps, wanted)
            except OSError as error:
                video_note = f"<p class=note>No stills: {html.escape(str(error))}</p>"

    findings = []
    for scorer, segment in chosen:
        seconds = (
            f"{segment.start_frame / episode.fps:.1f} to {segment.end_frame / episode.fps:.1f} s"
        )
        findings.append(
            f"<div class=finding><p><b>{html.escape(segment.reason)}</b><br>"
            f"<span class=note>{html.escape(scorer)} &middot; frames {segment.start_frame} to "
            f"{segment.end_frame} ({seconds})</span></p>"
            f"{_filmstrips(segment, len(episode), stills)}{_plot(episode, segment)}</div>"
        )
    hidden = len(summary.segments) - len(chosen)
    more = f"<p class=note>{hidden} more flagged segments not shown.</p>" if hidden > 0 else ""
    return (
        f'<section id="e{episode.index}"><h3>Episode {episode.index}</h3>'
        f"<p class=note>Overall {summary.score:.2f} &middot; {len(episode)} frames &middot; "
        f"{len(episode) / episode.fps:.1f} s</p>{video_note}{''.join(findings)}{more}</section>"
    )


def _choose(summary: scores.EpisodeSummary) -> list[tuple[str, scores.Segment]]:
    """Picks the segments to show: the longest of each kind of problem.

    Args:
        summary: The episode's combined scores and segments.

    Returns:
        Up to `SEGMENTS_PER_EPISODE` segments with their scorers, longest first.
    """
    longest: dict[tuple[str, str], tuple[str, scores.Segment]] = {}
    for scorer, segment in summary.segments:
        kind = (scorer, _NUMBER.sub("N", segment.reason))
        length = segment.end_frame - segment.start_frame
        held = longest.get(kind)
        if held is None or length > held[1].end_frame - held[1].start_frame:
            longest[kind] = (scorer, segment)
    ranked = sorted(longest.values(), key=lambda item: item[1].start_frame - item[1].end_frame)
    return ranked[:SEGMENTS_PER_EPISODE]


def _still_frames(segment: scores.Segment, length: int) -> list[tuple[int, bool]]:
    """Chooses which frames to show for a segment.

    A long segment is sampled evenly. A short one is shown with the frames
    either side of it, so that there is something to compare it with.

    Args:
        segment: The flagged segment.
        length: The number of frames in the episode.

    Returns:
        Frame numbers in order, each with whether it lies inside the segment.
    """
    start = max(segment.start_frame, 0)
    stop = min(segment.end_frame, length)
    if stop - start >= _STILLS:
        frames = np.linspace(start, stop - 1, _STILLS).round().astype(int).tolist()
    else:
        first = max(0, min(start - (_STILLS - (stop - start)) // 2, length - _STILLS))
        frames = list(range(first, min(first + _STILLS, length)))
    return [(frame, start <= frame < stop) for frame in frames]


def _filmstrips(segment: scores.Segment, length: int, stills: dict[str, dict[int, bytes]]) -> str:
    """Lays out a row of stills per camera for a segment.

    Args:
        segment: The flagged segment.
        length: The number of frames in the episode.
        stills: JPEG bytes by camera and frame number.

    Returns:
        The rows' HTML, or an empty string if there are no stills.
    """
    rows = []
    for camera, by_frame in stills.items():
        figures = []
        for frame, flagged in _still_frames(segment, length):
            if frame not in by_frame:
                continue
            data = base64.b64encode(by_frame[frame]).decode("ascii")
            figures.append(
                f"<figure{' class=in' if flagged else ''}>"
                f'<img alt="Frame {frame}" src="data:image/jpeg;base64,{data}">'
                f"<figcaption>{frame}</figcaption></figure>"
            )
        if figures:
            rows.append(
                f"<div class=strip><span class=note>{html.escape(camera)}</span>"
                f"<div>{''.join(figures)}</div></div>"
            )
    return "".join(rows)


def _lines(episode: episodes.Episode, segment: scores.Segment) -> list[Line]:
    """Picks the signals that show a segment's problem.

    Args:
        episode: The episode.
        segment: The flagged segment.

    Returns:
        Up to three lines as (label, values over the whole episode scaled to
        0 to 1, dashed). Empty if the segment is not about a numeric column.
    """
    if not segment.column or not episode.has_column(segment.column):
        return []
    values = episode.column(segment.column).astype(np.float64)
    values = values.reshape(len(values), -1)
    if len(values) < 2:
        return []
    low = np.nanmin(values, axis=0)
    span = np.nanmax(values, axis=0) - low
    span = np.where(span > 0, span, 1.0)
    scaled = (values - low) / span

    dims = [dim for dim in segment.dims if dim < values.shape[1]]
    if not dims:
        # No dimension was named, so show the ones that travel furthest.
        travel = np.nansum(np.abs(np.diff(scaled, axis=0)), axis=0)
        dims = np.argsort(-travel)[:_MAX_LINES].tolist()
    names = episode.names(segment.column)
    lines: list[Line] = [
        (names[dim] if dim < len(names) else f"{segment.column}[{dim}]", scaled[:, dim], False)
        for dim in dims[:_MAX_LINES]
    ]

    # For a single commanded value, add what the robot measured, on the same scale.
    is_command = segment.column == episodes.ACTION_COLUMN and len(lines) == 1
    if is_command and episode.has_column(episodes.STATE_COLUMN):
        state = episode.states.astype(np.float64)
        state = state.reshape(len(state), -1)
        if state.shape == values.shape:
            dim = dims[0]
            lines = [
                (f"{lines[0][0]}, commanded", lines[0][1], False),
                (f"{lines[0][0]}, measured", (state[:, dim] - low[dim]) / span[dim], True),
            ]
    return lines


def _plot(episode: episodes.Episode, segment: scores.Segment) -> str:
    """Draws the signals behind a segment, with the flagged range shaded.

    Args:
        episode: The episode.
        segment: The flagged segment.

    Returns:
        The chart's SVG, or an empty string if there is nothing to draw.
    """
    lines = _lines(episode, segment)
    if not lines:
        return ""
    length = len(lines[0][1])
    start, stop = max(segment.start_frame, 0), min(segment.end_frame, length)
    padding = max(round(episode.fps), stop - start)
    first, last = max(0, start - padding), min(length, stop + padding)
    if last - first < 2:
        return ""

    inner_width = _PLOT_WIDTH - _PLOT_LEFT - _PLOT_RIGHT
    inner_height = _PLOT_HEIGHT - _PLOT_TOP - _PLOT_BOTTOM
    window = np.concatenate([values[first:last] for _, values, _ in lines])
    finite = window[np.isfinite(window)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5

    def x(frame: float) -> float:
        return _PLOT_LEFT + (frame - first) / (last - 1 - first) * inner_width

    def y(value: float) -> float:
        return _PLOT_TOP + (1 - (value - low) / (high - low)) * inner_height

    parts = [
        f'<rect class=flag x="{x(start):.1f}" y="{_PLOT_TOP}" '
        f'width="{max(x(stop - 1) - x(start), 2):.1f}" height="{inner_height}"/>'
    ]
    label_ys = _spread(
        [y(float(np.nan_to_num(values[last - 1], nan=low))) for _, values, _ in lines]
    )
    for number, ((label, values, dashed), label_y) in enumerate(zip(lines, label_ys, strict=True)):
        points = " ".join(
            f"{x(frame):.1f},{y(value):.1f}"
            for frame, value in enumerate(values[first:last].tolist(), start=first)
            if np.isfinite(value)
        )
        style = f"l{number + 1}" + (" dash" if dashed else "")
        parts.append(f'<polyline class="{style}" points="{points}"/>')
        label_x = _PLOT_WIDTH - _PLOT_RIGHT + 8
        parts.append(
            f'<line class="{style}" x1="{label_x}" x2="{label_x + 14}" y1="{label_y:.1f}" '
            f'y2="{label_y:.1f}"/><text x="{label_x + 19}" y="{label_y + 4:.1f}">'
            f"{html.escape(label)}</text>"
        )
    axis_y = _PLOT_HEIGHT - _PLOT_BOTTOM
    parts.append(
        f'<line class=axis x1="{_PLOT_LEFT}" x2="{_PLOT_WIDTH - _PLOT_RIGHT}" y1="{axis_y}" '
        f'y2="{axis_y}"/><text x="{_PLOT_LEFT}" y="{_PLOT_HEIGHT - 4}">frame {first}</text>'
        f'<text x="{_PLOT_WIDTH - _PLOT_RIGHT}" y="{_PLOT_HEIGHT - 4}" text-anchor="end">'
        f"{last - 1}</text>"
    )
    description = html.escape(f"{segment.column} around frames {start} to {stop}")
    return (
        f'<svg class=plot viewBox="0 0 {_PLOT_WIDTH} {_PLOT_HEIGHT}" role="img" '
        f'aria-label="{description}">{"".join(parts)}</svg>'
        "<p class=note>Each line is scaled to its own range over the episode. "
        "The shaded band is the flagged segment.</p>"
    )


def _spread(positions: list[float]) -> list[float]:
    """Moves label positions apart so that no two overlap.

    Args:
        positions: The vertical position each label would like.

    Returns:
        Positions in the same order, at least a label's height apart and
        inside the plot.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = list(positions)
    floor = float(_PLOT_TOP + 4)
    for i in order:
        placed[i] = max(placed[i], floor)
        floor = placed[i] + _LABEL_GAP
    overflow = floor - _LABEL_GAP - (_PLOT_HEIGHT - _PLOT_BOTTOM)
    if overflow > 0:
        placed = [position - overflow for position in placed]
    return placed
