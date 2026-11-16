"""Writes a dataset's scores as one HTML file that opens offline."""

import html
import re
from collections import Counter
from pathlib import Path

import numpy as np

from bowerbird.core import scores
from bowerbird.io import lerobot

_BINS = 10
_CHART_WIDTH = 260
_CHART_HEIGHT = 96
_BAR_GAP = 2
_LANE_HEIGHT = 6
_LANE_GAP = 2
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Scorers take colours in this fixed order, so a scorer keeps its colour
# whichever others are present.
_SERIES = ("health", "motion", "length", "random")

_STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --raised: #f3f2ef; --rule: #e2e1dc;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19; --raised: #242423; --rule: #383835;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px 64px; background: var(--surface); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; overflow-wrap: anywhere; }
h2 { font-size: 18px; margin: 40px 0 4px; }
p { margin: 0 0 12px; color: var(--text-2); max-width: 70ch; }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
.tiles, .charts { display: grid; gap: 12px; margin-top: 16px; }
.tiles { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.charts { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.tile, .chart { background: var(--raised); border-radius: 8px; padding: 14px 16px; }
.tile b { display: block; font-size: 26px; line-height: 1.2; font-variant-numeric: tabular-nums; }
.tile span, .chart span { color: var(--text-2); font-size: 13px; }
.chart h3 { margin: 0; font-size: 15px; }
.chart svg { display: block; width: 100%; height: auto; margin-top: 10px; }
.chart text { fill: var(--text-3); font-size: 11px; }
.axis { stroke: var(--rule); stroke-width: 1; }
.scroll { overflow-x: auto; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px 8px 0; text-align: left; vertical-align: top; }
th { color: var(--text-2); font-size: 13px; font-weight: 600; white-space: nowrap; }
tr { border-bottom: 1px solid var(--rule); }
td.number, th.number { text-align: right; }
td.why { min-width: 280px; }
td.why ul { margin: 6px 0 0; padding-left: 18px; color: var(--text-2); font-size: 13px; }
.timeline { display: block; width: 100%; min-width: 240px; border-radius: 2px; }
.track { fill: var(--rule); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 0; font-size: 13px; }
.legend i {
  display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
}
.s-health { fill: var(--series-1); background: var(--series-1); }
.s-motion { fill: var(--series-2); background: var(--series-2); }
.s-length { fill: var(--series-3); background: var(--series-3); }
.s-random, .s-other { fill: var(--series-4); background: var(--series-4); }
footer { margin-top: 48px; color: var(--text-3); font-size: 13px; }
"""


def write(
    dataset: lerobot.Dataset, table: scores.ScoreTable, path: str | Path, worst: int = 50
) -> Path:
    """Writes the report for a scored dataset.

    Args:
        dataset: The dataset that was scored.
        table: The dataset's scores.
        path: Where to write the HTML file.
        worst: How many of the lowest-scoring episodes to list.

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(dataset, table, worst), encoding="utf-8")
    return path


def render(dataset: lerobot.Dataset, table: scores.ScoreTable, worst: int = 50) -> str:
    """Builds the report's HTML.

    Args:
        dataset: The dataset that was scored.
        table: The dataset's scores.
        worst: How many of the lowest-scoring episodes to list.

    Returns:
        A complete HTML document with no external resources.
    """
    summaries = table.summaries()
    lengths = {episode.index: episode.length for episode in dataset}
    frames = sum(lengths.values())
    flagged = sum(1 for summary in summaries if summary.segments)
    name = html.escape(dataset.name)
    tiles = [
        (f"{len(lengths):,}", "episodes"),
        (f"{frames:,}", "frames"),
        (_duration(frames / dataset.fps), f"recorded at {dataset.fps:g} fps"),
        (f"{len(dataset.cameras)}", "camera" if len(dataset.cameras) == 1 else "cameras"),
        (f"{flagged:,}", "episodes with a flagged segment"),
    ]
    tile_html = "".join(
        f"<div class=tile><b>{big}</b><span>{small}</span></div>" for big, small in tiles
    )
    charts = "".join(_histogram(scorer, table) for scorer in table.scorers)
    overall = ", ".join(n for n in table.scorers if n not in scores.BASELINES) or "all scorers"
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bowerbird report: {name}</title>
<style>{_STYLE}</style>
<main>
<h1>{name}</h1>
<p>Demonstration quality report. Scores run from 0 (worst) to 1 (best).</p>
<div class=tiles>{tile_html}</div>

<h2>Score distributions</h2>
<p>How many episodes fall in each score range, per scorer. Hover a bar for its count.</p>
<div class=charts>{charts}</div>

<h2>Most common problems</h2>
{_problems(summaries)}

<h2>Lowest-scoring episodes</h2>
<p>The overall score is the mean of: {html.escape(overall)}. The length and random scorers are
baselines for comparison and stay out of it. Each bar shows the episode from first frame to last,
with flagged segments marked. Hover a mark for its reason.</p>
{_legend(table)}
{_episodes(table.worst(worst), lengths, table, len(summaries))}

<footer>Dataset version <code>{html.escape(table.dataset_fingerprint)}</code>. Scorer versions:
{html.escape(", ".join(f"{n} {table.version(n)}" for n in table.scorers))}.</footer>
</main>
</html>
"""


def _duration(seconds: float) -> str:
    """Formats a duration as hours, minutes or seconds.

    Args:
        seconds: The duration in seconds.
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


def _series_class(scorer: str) -> str:
    """Returns the CSS class that colours a scorer's marks.

    Args:
        scorer: The scorer's name.
    """
    return f"s-{scorer}" if scorer in _SERIES else "s-other"


def _histogram(scorer: str, table: scores.ScoreTable) -> str:
    """Draws one scorer's score distribution as a bar chart.

    Args:
        scorer: The scorer's name.
        table: The dataset's scores.

    Returns:
        The chart's HTML.
    """
    values = np.array([score.score for score in table.get(scorer).values()])
    counts, _ = np.histogram(values, bins=_BINS, range=(0.0, 1.0))
    tallest = max(int(counts.max()), 1)
    slot = _CHART_WIDTH / _BINS
    bars = []
    for i, count in enumerate(counts.tolist()):
        height = round(count / tallest * (_CHART_HEIGHT - 4), 1) if count else 0
        label = f"{i / _BINS:.1f} to {(i + 1) / _BINS:.1f}: {count} episodes"
        bars.append(
            f'<rect class="{_series_class(scorer)}" x="{i * slot + _BAR_GAP / 2:.1f}" '
            f'y="{_CHART_HEIGHT - height}" width="{slot - _BAR_GAP:.1f}" height="{height}" rx="2">'
            f"<title>{label}</title></rect>"
            # A full-height invisible target, so short bars are still easy to hover.
            f'<rect x="{i * slot:.1f}" y="0" width="{slot:.1f}" height="{_CHART_HEIGHT}" '
            f'fill="transparent"><title>{label}</title></rect>'
        )
    baseline = "baseline, " if scorer in scores.BASELINES else ""
    return f"""<div class=chart>
<h3>{html.escape(scorer)}</h3>
<span>{baseline}median {np.median(values):.2f}, lowest {values.min():.2f}</span>
<svg viewBox="0 0 {_CHART_WIDTH} {_CHART_HEIGHT + 18}" role="img"
  aria-label="Histogram of {html.escape(scorer)} scores">
{"".join(bars)}
<line class=axis x1="0" x2="{_CHART_WIDTH}" y1="{_CHART_HEIGHT}" y2="{_CHART_HEIGHT}"/>
<text x="0" y="{_CHART_HEIGHT + 14}">0</text>
<text x="{_CHART_WIDTH / 2}" y="{_CHART_HEIGHT + 14}" text-anchor="middle">0.5</text>
<text x="{_CHART_WIDTH}" y="{_CHART_HEIGHT + 14}" text-anchor="end">1</text>
</svg></div>"""


def _problems(summaries: list[scores.EpisodeSummary]) -> str:
    """Tabulates the kinds of flagged segment by how many episodes have them.

    Args:
        summaries: Every episode's summary.

    Returns:
        The table's HTML, or a sentence saying nothing was flagged.
    """
    episodes_with: Counter[tuple[str, str]] = Counter()
    for summary in summaries:
        kinds = {(scorer, _NUMBER.sub("N", segment.reason)) for scorer, segment in summary.segments}
        episodes_with.update(kinds)
    if not episodes_with:
        return "<p>No segments were flagged.</p>"
    rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{html.escape(scorer)}</td>"
        f"<td class=number>{count}</td><td class=number>{count / len(summaries):.0%}</td></tr>"
        for (scorer, reason), count in episodes_with.most_common()
    )
    return f"""<p>Numbers in a reason are replaced by N so that similar problems group together.</p>
<div class=scroll><table>
<tr><th>Problem</th><th>Scorer</th><th class=number>Episodes</th><th class=number>Share</th></tr>
{rows}</table></div>"""


def _legend(table: scores.ScoreTable) -> str:
    """Lists which colour marks which scorer's segments.

    Args:
        table: The dataset's scores.

    Returns:
        The legend's HTML.
    """
    items = "".join(
        f'<span><i class="{_series_class(scorer)}"></i>{html.escape(scorer)}</span>'
        for scorer in table.scorers
        if any(score.segments for score in table.get(scorer).values())
    )
    return f"<div class=legend>{items}</div>" if items else ""


def _timeline(summary: scores.EpisodeSummary, length: int, lanes: list[str]) -> str:
    """Draws an episode as a bar with its flagged segments marked.

    Args:
        summary: The episode's summary.
        length: The number of frames in the episode.
        lanes: The scorers that get a lane each, top to bottom.

    Returns:
        The timeline's HTML.
    """
    length = max(length, 1)
    height = len(lanes) * (_LANE_HEIGHT + _LANE_GAP) - _LANE_GAP
    marks = []
    for lane, scorer in enumerate(lanes):
        y = lane * (_LANE_HEIGHT + _LANE_GAP)
        marks.append(f'<rect class=track x="0" y="{y}" width="{length}" height="{_LANE_HEIGHT}"/>')
        for owner, segment in summary.segments:
            if owner != scorer:
                continue
            title = html.escape(
                f"{scorer}: {segment.reason} (frames {segment.start_frame} to {segment.end_frame})"
            )
            width = max(segment.end_frame - segment.start_frame, length / 200)
            marks.append(
                f'<rect class="{_series_class(scorer)}" x="{segment.start_frame}" y="{y}" '
                f'width="{width:.2f}" height="{_LANE_HEIGHT}"><title>{title}</title></rect>'
            )
    return (
        f'<svg class=timeline viewBox="0 0 {length} {height}" height="{height}" '
        f'preserveAspectRatio="none" role="img" aria-label="Flagged segments">'
        f"{''.join(marks)}</svg>"
    )


def _episodes(
    worst: list[scores.EpisodeSummary],
    lengths: dict[int, int],
    table: scores.ScoreTable,
    total: int,
) -> str:
    """Tabulates the lowest-scoring episodes.

    Args:
        worst: The summaries to list, worst first.
        lengths: Each episode's number of frames.
        table: The dataset's scores.
        total: How many episodes the dataset has.

    Returns:
        The table's HTML.
    """
    lanes = [
        scorer
        for scorer in table.scorers
        if any(score.segments for score in table.get(scorer).values())
    ] or list(table.scorers[:1])
    header = "".join(f"<th class=number>{html.escape(scorer)}</th>" for scorer in table.scorers)
    rows = []
    for summary in worst:
        cells = "".join(
            f"<td class=number>{summary.scores[scorer]:.2f}</td>"
            if scorer in summary.scores
            else "<td></td>"
            for scorer in table.scorers
        )
        reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in summary.reasons[:4])
        more = len(summary.reasons) - 4
        if more > 0:
            reasons += f"<li>and {more} more</li>"
        length = lengths.get(summary.index, 0)
        rows.append(
            f"<tr><td class=number>{summary.index}</td>"
            f"<td class=number><b>{summary.score:.2f}</b></td>{cells}"
            f"<td class=number>{length}</td>"
            f"<td class=why>{_timeline(summary, length, lanes)}<ul>{reasons}</ul></td></tr>"
        )
    note = f"<p>Showing {len(worst)} of {total} episodes.</p>" if len(worst) < total else ""
    return f"""<div class=scroll><table>
<tr><th class=number>Episode</th><th class=number>Overall</th>{header}
<th class=number>Frames</th><th>Flagged segments</th></tr>
{"".join(rows)}</table></div>{note}"""
