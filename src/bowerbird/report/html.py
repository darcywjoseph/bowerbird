"""Writes a dataset's scores as one HTML file that opens offline."""

import html
import re
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bowerbird.core import scores
from bowerbird.io import lerobot
from bowerbird.report import evidence

_BINS = 10
_SPARK_WIDTH = 120
_SPARK_HEIGHT = 22
_LANE_HEIGHT = 7
_REASONS_SHOWN = 4
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_STYLE = """
:root {
  color-scheme: light dark;
  --paper: #fff; --ink: #111; --muted: #666; --rule: #cfcfcf; --track: #e9e9e9;
  --l1: #2a78d6; --l2: #eb6834; --l3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #161616; --ink: #e6e6e6; --muted: #9a9a9a; --rule: #444; --track: #2e2e2e;
    --l1: #3987e5; --l2: #d95926; --l3: #199e70;
  }
}
body {
  margin: 0; padding: 40px 20px 80px; background: var(--paper); color: var(--ink);
  font: 16px/1.5 Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
}
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 2px; overflow-wrap: anywhere; }
h2 { font-size: 17px; font-weight: 600; margin: 36px 0 6px; }
p { margin: 0 0 10px; }
.meta, .note { color: var(--muted); font-size: 14px; }
code { font: 13px ui-monospace, Menlo, Consolas, monospace; }
.scroll { overflow-x: auto; }
table {
  border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums;
}
th, td { padding: 5px 14px 5px 0; text-align: left; vertical-align: top; }
th { font-weight: 600; border-bottom: 1px solid var(--ink); white-space: nowrap; }
td { border-bottom: 1px solid var(--rule); }
.n { text-align: right; }
svg { display: block; }
.bar, .mark { fill: var(--ink); }
.track { fill: var(--track); }
.why { min-width: 300px; }
.lane { display: grid; grid-template-columns: 50px 1fr; align-items: center; margin-bottom: 2px; }
.lane span { color: var(--muted); font-size: 12px; }
.lane svg { width: 100%; }
.reasons { color: var(--muted); font-size: 14px; margin: 3px 0 2px; }
a { color: inherit; }
h3 { font-size: 16px; font-weight: 600; margin: 28px 0 0; }
section { border-top: 1px solid var(--rule); margin-top: 24px; }
.finding { margin: 14px 0 22px; }
.strip { margin: 6px 0; }
.strip > div { display: flex; gap: 4px; overflow-x: auto; }
figure { margin: 0; border: 2px solid transparent; }
figure.in { border-color: var(--ink); }
figure img { display: block; height: 96px; }
figcaption { color: var(--muted); font-size: 12px; text-align: center; }
.plot { width: 100%; max-width: 720px; height: auto; margin-top: 8px; }
.plot text { fill: var(--muted); font: 11px ui-sans-serif, system-ui, sans-serif; }
.plot polyline, .plot line { fill: none; stroke-width: 1.5; stroke-linejoin: round; }
.plot .axis { stroke: var(--rule); stroke-width: 1; }
.flag { fill: var(--track); }
.l1 { stroke: var(--l1); } .l2 { stroke: var(--l2); } .l3 { stroke: var(--l3); }
.dash { stroke-dasharray: 4 3; }
"""


def write(
    dataset: lerobot.Dataset,
    table: scores.ScoreTable,
    path: str | Path,
    *,
    worst: int = 50,
    detailed: int = 12,
    check_video: bool = True,
) -> Path:
    """Writes the report for a scored dataset.

    Args:
        dataset: The dataset that was scored.
        table: The dataset's scores.
        path: Where to write the HTML file.
        worst: How many of the lowest-scoring episodes to list.
        detailed: How many of those to show stills and signal plots for.
        check_video: Whether to decode the videos for the stills.

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page = render(dataset, table, worst=worst, detailed=detailed, check_video=check_video)
    path.write_text(page, encoding="utf-8")
    return path


def render(
    dataset: lerobot.Dataset,
    table: scores.ScoreTable,
    *,
    worst: int = 50,
    detailed: int = 12,
    check_video: bool = True,
) -> str:
    """Builds the report's HTML.

    Args:
        dataset: The dataset that was scored.
        table: The dataset's scores.
        worst: How many of the lowest-scoring episodes to list.
        detailed: How many of those to show stills and signal plots for.
        check_video: Whether to decode the videos for the stills.

    Returns:
        A complete HTML document with no external resources.
    """
    summaries = table.summaries()
    lengths = {episode.index: episode.length for episode in dataset}
    frames = sum(lengths.values())
    flagged = sum(1 for summary in summaries if summary.segments)
    cameras = len(dataset.cameras)
    listed = table.worst(worst)
    shown = [summary for summary in listed if summary.segments][:detailed]
    sections = "".join(
        evidence.render(dataset[summary.index], summary, check_video) for summary in shown
    )
    if sections:
        sections = f"""<h2>What the flagged segments look like</h2>
<p class=note>The {len(shown)} lowest-scoring episodes with a flagged segment. Up to
{evidence.SEGMENTS_PER_EPISODE} segments are shown for each: the longest of each kind of problem.
Stills are numbered by frame, and those inside the flagged segment are outlined.</p>
{sections}"""
    name = html.escape(dataset.name)
    versions = ", ".join(f"{scorer} {table.version(scorer)}" for scorer in _shown(table))
    overall = " and ".join(n for n in _shown(table) if n not in scores.BASELINES) or "all scorers"
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} - bowerbird</title>
<style>{_STYLE}</style>
<main>
<h1>{name}</h1>
<p class=meta>bowerbird {html.escape(metadata.version("bowerbird"))} &middot;
dataset version <code>{html.escape(table.dataset_fingerprint[:12])}</code> &middot;
scorers: {html.escape(versions)}</p>

<p>{len(lengths):,} episodes, {frames:,} frames ({_duration(frames / dataset.fps)} at
{dataset.fps:g} fps), {cameras} camera{"" if cameras == 1 else "s"}.
{flagged:,} episodes have at least one flagged segment.</p>

<h2>Scores</h2>
<p class=note>Each score runs from 0 (worst) to 1 (best). The distribution column counts episodes
in ten equal score ranges, low to high.</p>
{_score_table(table)}

<h2>Problems found</h2>
{_problems(summaries)}

<h2>Lowest-scoring episodes</h2>
<p class=note>Overall is the mean of {html.escape(overall)}. The length scorer is a baseline for
comparison and is left out of it. Each bar is the episode from first frame to last, with flagged
segments filled in.</p>
{_episodes(listed, lengths, table, len(summaries), {summary.index for summary in shown})}

{sections}
</main>
</html>
"""


def _shown(table: scores.ScoreTable) -> list[str]:
    """Lists the scorers the report covers.

    The random control is left out: it exists for benchmark comparisons and
    says nothing about the dataset.

    Args:
        table: The dataset's scores.
    """
    return [scorer for scorer in table.scorers if scorer != "random"]


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


def _score_table(table: scores.ScoreTable) -> str:
    """Tabulates each scorer's score distribution.

    Args:
        table: The dataset's scores.

    Returns:
        The table's HTML.
    """
    rows = []
    for scorer in _shown(table):
        values = np.array([score.score for score in table.get(scorer).values()])
        label = html.escape(scorer) + (" (baseline)" if scorer in scores.BASELINES else "")
        rows.append(
            f"<tr><td>{label}</td><td>{_sparkline(values)}</td>"
            f"<td class=n>{np.median(values):.2f}</td><td class=n>{values.min():.2f}</td>"
            f"<td class=n>{int((values < 0.5).sum())}</td></tr>"
        )
    return f"""<div class=scroll><table>
<tr><th>Scorer</th><th>Distribution</th><th class=n>Median</th><th class=n>Lowest</th>
<th class=n>Episodes below 0.5</th></tr>
{"".join(rows)}</table></div>"""


def _sparkline(values: npt.NDArray[Any]) -> str:
    """Draws a score distribution as a small bar chart.

    Args:
        values: One score per episode.

    Returns:
        The chart's SVG. Hovering a bar gives its range and count.
    """
    counts, _ = np.histogram(values, bins=_BINS, range=(0.0, 1.0))
    tallest = max(int(counts.max()), 1)
    slot = _SPARK_WIDTH / _BINS
    bars = []
    for i, count in enumerate(counts.tolist()):
        # Every non-empty range gets at least a sliver, so that it is visible.
        height = max(round(count / tallest * _SPARK_HEIGHT, 1), 1.5) if count else 0
        label = f"{i / _BINS:.1f} to {(i + 1) / _BINS:.1f}: {count} episodes"
        bars.append(
            f'<rect class=bar x="{i * slot + 0.5:.1f}" y="{_SPARK_HEIGHT - height}" '
            f'width="{slot - 1:.1f}" height="{height}"/>'
            f'<rect x="{i * slot:.1f}" y="0" width="{slot:.1f}" height="{_SPARK_HEIGHT}" '
            f'fill="transparent"><title>{label}</title></rect>'
        )
    return (
        f'<svg width="{_SPARK_WIDTH}" height="{_SPARK_HEIGHT + 1}" role="img" '
        f'aria-label="Score distribution">{"".join(bars)}'
        f'<rect class=track x="0" y="{_SPARK_HEIGHT}" width="{_SPARK_WIDTH}" height="1"/></svg>'
    )


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
        f"<td class=n>{count}</td><td class=n>{count / len(summaries):.0%}</td></tr>"
        for (scorer, reason), count in episodes_with.most_common()
    )
    return f"""<p class=note>Each description comes from a fixed rule in the scorer named beside
it. Numbers are shown as N so that like problems share a row.</p>
<div class=scroll><table>
<tr><th>Problem</th><th>Scorer</th><th class=n>Episodes</th><th class=n>Share</th></tr>
{rows}</table></div>"""


def _timeline(summary: scores.EpisodeSummary, length: int, lanes: list[str]) -> str:
    """Draws an episode as one bar per scorer, with flagged segments filled in.

    Args:
        summary: The episode's summary.
        length: The number of frames in the episode.
        lanes: The scorers that get a bar each, top to bottom.

    Returns:
        The timeline's HTML. Hovering a segment gives its reason and frames.
    """
    length = max(length, 1)
    rows = []
    for scorer in lanes:
        marks = [f'<rect class=track x="0" y="0" width="{length}" height="{_LANE_HEIGHT}"/>']
        for owner, segment in summary.segments:
            if owner != scorer:
                continue
            title = html.escape(
                f"{segment.reason} (frames {segment.start_frame} to {segment.end_frame})"
            )
            # A one-frame segment in a long episode would otherwise be invisible.
            width = max(segment.end_frame - segment.start_frame, length / 250)
            marks.append(
                f'<rect class=mark x="{segment.start_frame}" y="0" width="{width:.2f}" '
                f'height="{_LANE_HEIGHT}"><title>{title}</title></rect>'
            )
        rows.append(
            f"<div class=lane><span>{html.escape(scorer)}</span>"
            f'<svg viewBox="0 0 {length} {_LANE_HEIGHT}" height="{_LANE_HEIGHT}" '
            f'preserveAspectRatio="none" role="img" aria-label="{html.escape(scorer)} segments">'
            f"{''.join(marks)}</svg></div>"
        )
    return "".join(rows)


def _episodes(
    worst: list[scores.EpisodeSummary],
    lengths: dict[int, int],
    table: scores.ScoreTable,
    total: int,
    linked: set[int],
) -> str:
    """Tabulates the lowest-scoring episodes.

    Args:
        worst: The summaries to list, worst first.
        lengths: Each episode's number of frames.
        table: The dataset's scores.
        total: How many episodes the dataset has.
        linked: The episodes that have a section further down to link to.

    Returns:
        The table's HTML.
    """
    lanes = [
        scorer
        for scorer in _shown(table)
        if any(score.segments for score in table.get(scorer).values())
    ]
    header = "".join(f"<th class=n>{html.escape(scorer)}</th>" for scorer in _shown(table))
    rows = []
    for summary in worst:
        cells = "".join(
            f"<td class=n>{summary.scores[scorer]:.2f}</td>"
            if scorer in summary.scores
            else "<td></td>"
            for scorer in _shown(table)
        )
        reasons = list(summary.reasons[:_REASONS_SHOWN])
        if len(summary.reasons) > _REASONS_SHOWN:
            reasons.append(f"and {len(summary.reasons) - _REASONS_SHOWN} more")
        length = lengths.get(summary.index, 0)
        number = (
            f'<a href="#e{summary.index}">{summary.index}</a>'
            if summary.index in linked
            else str(summary.index)
        )
        why = _timeline(summary, length, lanes) if summary.segments else ""
        why += f"<p class=reasons>{html.escape('; '.join(reasons)) or 'nothing flagged'}</p>"
        rows.append(
            f"<tr><td class=n>{number}</td><td class=n>{summary.score:.2f}</td>{cells}"
            f"<td class=n>{length}</td><td class=why>{why}</td></tr>"
        )
    note = (
        f"<p class=note>Showing {len(worst)} of {total} episodes.</p>" if len(worst) < total else ""
    )
    return f"""<div class=scroll><table>
<tr><th class=n>Episode</th><th class=n>Overall</th>{header}
<th class=n>Frames</th><th>Flagged segments</th></tr>
{"".join(rows)}</table></div>{note}"""
