"""PDF reports over one or more runs, rendered for the viewer's report dialog.

`render_report` turns a `ReportRequest` into PDF bytes: a header with the runs, a table of
final metric values, one chart per metric with every run overlaid, the media of each tag
at one step and the config files of each run. The page is laid out as HTML and printed by
WeasyPrint, so charts never split across pages and page numbers come from CSS paged media.
Charts are matplotlib SVGs, so they stay vector in the PDF. Videos become one grid image
of their frames. `build_report_html` returns the intermediate HTML.
"""
from __future__ import annotations

import base64
import html
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from pydantic import Field

from ldtrain.utils.files import load_video
from ldtrain.viewer import reader

MAX_GRID_FRAMES = 64          # longer videos are subsampled evenly to this many frames
MAX_GRID_WIDTH_PX = 2048      # frames are downscaled so a video grid stays below this width
GRID_GAP_PX = 2
MAX_POINTS_WITH_MARKERS = 12  # like the viewer: sparse series get hollow markers
UNGROUPED = ''
AXIS_COLOR = '#656d78'
GRID_COLOR = '#e6e8eb'
CHART_COLUMNS = {'portrait': 2, 'landscape': 3}
CHART_SIZE_IN = {'portrait': (3.5, 2.1), 'landscape': (3.4, 2.0)}

HexColor = Annotated[str, Field(pattern=r'^#[0-9a-fA-F]{6}$')]

#---------------------------------------------------------------------
# request
#---------------------------------------------------------------------

@dataclass
class ReportRun:
    """One run of the report, drawn like its viewer chip.

    Attributes:
        path: Run path relative to the viewer root.
        color: Line and swatch color, `#rrggbb`.
        dashed: Whether its lines are dashed, as for palette slots 8 and above.
    """
    path: str
    color: HexColor
    dashed: bool = False


@dataclass
class ReportRequest:
    """What the report holds, as configured in the viewer's report dialog.

    Attributes:
        runs: Runs in legend order, at least one.
        title: Document title. Empty derives one from the runs.
        metrics: Metric columns to chart, in order. Consecutive metrics sharing a group
            path share a heading. Columns no run has are skipped.
        smoothing: Weight of the debiased EMA drawn over each raw series, in [0, 0.99].
            0 draws the raw series only.
        log_metrics: Metrics drawn on a log y axis, where every value is above zero.
        x_range: Iteration range `[min, max]` shared by every chart, None for the full range.
        media_tags: Visual tags to show, in order. Tags no run has are skipped.
        media_step: Step shown for every tag, each run at its latest step ≤ it. None
            shows the latest step.
        include_summary: Whether to add the table of final, minimum and maximum values.
        include_config: Whether to add the config files of every run.
        orientation: Page orientation of the A4 pages.
    """
    runs: Annotated[list[ReportRun], Field(min_length=1)]
    title: Annotated[str, Field(max_length=200)] = ''
    metrics: list[str] = field(default_factory=list)
    smoothing: Annotated[float, Field(ge=0, le=0.99)] = 0.6
    log_metrics: list[str] = field(default_factory=list)
    x_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    media_tags: list[str] = field(default_factory=list)
    media_step: int | None = None
    include_summary: bool = True
    include_config: bool = True
    orientation: Literal['portrait', 'landscape'] = 'portrait'


@dataclass
class RunSeries:
    """The metric series of one run, non-null rows only."""
    run: ReportRun
    run_dir: Path
    series: dict[str, tuple[np.ndarray, np.ndarray]]  # metric -> (iterations, values), each shape (N,)

#---------------------------------------------------------------------
# entry points
#---------------------------------------------------------------------

def render_report(run_dirs: list[Path], request: ReportRequest) -> bytes:
    """Render the report as PDF bytes.

    Args:
        run_dirs: Resolved directory of each run in `request.runs`, same order. The
            caller checks they lie inside the viewer root.
        request: Report configuration.

    Raises:
        OSError: If WeasyPrint's system libraries (Pango) are missing.
    """
    import weasyprint  # imported late: it loads Pango, which a training-only node may lack

    return weasyprint.HTML(string=build_report_html(run_dirs, request)).write_pdf()


def build_report_html(run_dirs: list[Path], request: ReportRequest) -> str:
    """Return the report as a standalone HTML page, images referenced by `file://` URL."""
    runs = [RunSeries(run, run_dir, read_series(run_dir)) for run, run_dir in zip(request.runs, run_dirs)]
    metrics = [metric for metric in dict.fromkeys(request.metrics) if any(metric in run.series for run in runs)]
    title = request.title.strip() or default_title(request.runs)
    parts = [render_header(title, runs, request)]
    if request.include_summary and metrics:
        parts.append(render_summary(runs, metrics))
    if metrics:
        parts.append(render_metrics(runs, metrics, request))
    media = render_media(runs, request)
    if media:
        parts.append(media)
    if request.include_config:
        parts.append(render_configs(runs))
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>'
            f'<style>{page_css(request.orientation)}</style></head><body>{"".join(parts)}</body></html>')


def default_title(runs: list[ReportRun]) -> str:
    return runs[0].path if len(runs) == 1 else f'Comparison of {len(runs)} runs'

#---------------------------------------------------------------------
# data
#---------------------------------------------------------------------

def read_series(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read every metric column of a run, dropping rows where the value or iteration is empty."""
    chunk = reader.read_metrics(run_dir)
    iterations = chunk.data.get('iteration', [])
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in chunk.columns:
        if column == 'iteration':
            continue
        rows = [(it, value) for it, value in zip(iterations, chunk.data[column]) if it is not None and value is not None]
        if rows:
            values = np.array(rows, dtype=np.float64)  # (N, 2)
            series[column] = (values[:, 0], values[:, 1])
    return series


def smooth_ema(values: np.ndarray, weight: float) -> np.ndarray:
    """Debiased exponential moving average, identical to `smoothEma` in the viewer's util.js.

    Args:
        values: Raw series, shape (N,).
        weight: Weight of the previous average, in [0, 1). 0 returns `values`.

    Returns:
        Smoothed series, shape (N,).
    """
    if weight <= 0:
        return values
    smoothed = np.empty_like(values)
    last = 0.0
    for ii, value in enumerate(values):
        last = last * weight + (1 - weight) * value
        smoothed[ii] = last / (1 - weight ** (ii + 1))
    return smoothed


def resolve_visual(visuals: list[reader.VisualStep], tag: str, step: int | None) -> tuple[int, reader.VisualFile] | None:
    """The file of `tag` at the latest step ≤ `step` (any step when None), like the viewer's media cards."""
    for visual_step in reversed(visuals):
        if step is not None and visual_step.step > step:
            continue
        for visual_file in visual_step.files:
            if visual_file.tag == tag:
                return visual_step.step, visual_file
    return None

#---------------------------------------------------------------------
# formatting
#---------------------------------------------------------------------

def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def format_value(value: float | None) -> str:
    """Format like the viewer: 4 significant digits, exponent form for tiny or huge values."""
    if value is None or not math.isfinite(value):
        return 'n/a'
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
        return f'{value:.2e}'
    return f'{float(f"{value:.4g}"):g}'


def format_iteration(value: float) -> str:
    """Format an iteration compactly: 950, 13.4k, 2.1M."""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f'{value / 1e6:.1f}'.rstrip('0').rstrip('.') + 'M'
    if magnitude >= 1e3:
        return f'{value / 1e3:.1f}'.rstrip('0').rstrip('.') + 'k'
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def swatch(run: ReportRun) -> str:
    return f'<span class="sw{" dash" if run.dashed else ""}" style="--c:{run.color}"></span>'


def group_path_of(key: str) -> str:
    """Everything before the last `/` of a key, '' for an ungrouped key, as in groups.js."""
    index = key.rfind('/')
    return key[:index] if index > 0 else UNGROUPED


def card_title(key: str) -> str:
    group_path = group_path_of(key)
    if not group_path:
        return f'<span class="ct">{esc(key)}</span>'
    return f'<span class="ct"><span class="pre">{esc(group_path)}/</span>{esc(key[len(group_path) + 1:])}</span>'


def grouped(keys: list[str]) -> list[tuple[str, list[str]]]:
    """Split keys into runs of consecutive keys sharing a group path, keeping their order."""
    groups: list[tuple[str, list[str]]] = []
    for key in keys:
        group_path = group_path_of(key)
        if groups and groups[-1][0] == group_path:
            groups[-1][1].append(key)
        else:
            groups.append((group_path, [key]))
    return groups

#---------------------------------------------------------------------
# sections
#---------------------------------------------------------------------

def render_header(title: str, runs: list[RunSeries], request: ReportRequest) -> str:
    rows = []
    for run in runs:
        node = reader.scan_directory(run.run_dir, run.run_dir)  # the run as its own root: mtime and last iteration
        updated = time.strftime('%Y-%m-%d %H:%M', time.localtime(node.mtime)) if node and node.mtime else 'n/a'
        last_iteration = format_iteration(node.last_iteration) if node and node.last_iteration is not None else 'n/a'
        rows.append(f'<tr><td>{swatch(run.run)}</td><td class="path">{esc(run.run.path)}</td>'
                    f'<td class="num">{esc(last_iteration)}</td>'
                    f'<td class="num">{len(run.series)}</td><td class="num">{esc(updated)}</td></tr>')
    settings = [f'smoothing {request.smoothing:.2f}']
    if request.x_range:
        settings.append(f'iterations {format_iteration(request.x_range[0])} to {format_iteration(request.x_range[1])}')
    return (f'<header><h1>{esc(title)}</h1>'
            f'<p class="meta">Generated {esc(time.strftime("%Y-%m-%d %H:%M"))} by ldtrain · {esc(", ".join(settings))}</p>'
            '<table class="runs"><thead><tr><th></th><th>Run</th><th class="num">Last iteration</th><th class="num">Metrics</th><th class="num">Updated</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></header>')


def render_summary(runs: list[RunSeries], metrics: list[str]) -> str:
    """Final raw value of every metric per run, with minimum and maximum for a single run."""
    single = len(runs) == 1
    head = '<th>Metric</th>' + (''.join(f'<th>{swatch(run.run)} {esc(Path(run.run.path).name)}</th>' for run in runs)
                                if not single else '<th>Final</th><th>Min</th><th>Max</th><th>Iterations</th>')
    rows = []
    for metric in metrics:
        cells = []
        for run in runs:
            iterations, values = run.series.get(metric, (np.empty(0), np.empty(0)))
            final = values[-1] if values.size else None
            cells.append(f'<td class="num">{esc(format_value(final))}</td>')
            if single:
                span = f'{format_iteration(iterations[0])} to {format_iteration(iterations[-1])}' if iterations.size else 'n/a'
                cells.append(f'<td class="num">{esc(format_value(values.min() if values.size else None))}</td>'
                             f'<td class="num">{esc(format_value(values.max() if values.size else None))}</td>'
                             f'<td class="num">{esc(span)}</td>')
        rows.append(f'<tr><td>{card_title(metric)}</td>{"".join(cells)}</tr>')
    return (f'<section class="summary"><h2>Summary</h2><p class="note">Final logged value{"" if single else " of each run"}, unsmoothed.</p>'
            f'<table class="values"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></section>')


def render_metrics(runs: list[RunSeries], metrics: list[str], request: ReportRequest) -> str:
    log_metrics = set(request.log_metrics)
    blocks = []
    for group_path, keys in grouped(metrics):
        cards = [render_metric_card(runs, metric, metric in log_metrics, request) for metric in keys]
        blocks.append(f'<div class="group"><h3>{esc(group_path or "ungrouped")}</h3><div class="grid charts">{"".join(cards)}</div></div>')
    note = f'Smoothed with an EMA of weight {request.smoothing:.2f} over the faint raw series. ' if request.smoothing > 0 else ''
    return f'<section class="metrics"><h2>Metrics</h2><p class="note">{note}Legend values are the last drawn value of each run.</p>{"".join(blocks)}</section>'


def render_metric_card(runs: list[RunSeries], metric: str, log_y: bool, request: ReportRequest) -> str:
    present = [run for run in runs if metric in run.series]
    svg = render_chart(present, metric, log_y, request)
    legend = ''.join(f'<span class="leg">{swatch(run.run)}{esc(Path(run.run.path).name)} '
                     f'<b class="num">{esc(format_value(smooth_ema(run.series[metric][1], request.smoothing)[-1]))}</b></span>'
                     for run in present)
    scale = '<span class="tag">log</span>' if log_y and all(run.series[metric][1].min() > 0 for run in present) else ''
    return f'<article class="card"><div class="card-head">{card_title(metric)}{scale}</div><div class="plot">{svg}</div><div class="card-foot">{legend}</div></article>'


def render_chart(runs: list[RunSeries], metric: str, log_y: bool, request: ReportRequest) -> str:
    """Draw one metric of every run as an SVG: faint raw lines under solid smoothed lines."""
    figure = Figure(figsize=CHART_SIZE_IN[request.orientation])
    axes = figure.add_axes((0.14, 0.14, 0.83, 0.82))
    for run in runs:
        iterations, values = run.series[metric]
        dashes = (6, 3) if run.run.dashed else (None, None)
        if request.smoothing > 0:
            axes.plot(iterations, values, color=run.run.color, alpha=0.25, linewidth=0.8, dashes=dashes)
        marker = {'marker': 'o', 'markersize': 3.5, 'markerfacecolor': 'white'} if iterations.size <= MAX_POINTS_WITH_MARKERS else {}
        axes.plot(iterations, smooth_ema(values, request.smoothing), color=run.run.color, linewidth=1.3, dashes=dashes, **marker)
    if log_y and all(run.series[metric][1].min() > 0 for run in runs):
        axes.set_yscale('log')
        low, high = axes.get_ylim()
        axes.yaxis.set_major_locator(FixedLocator(log_splits(low, high)))
        axes.yaxis.set_minor_locator(NullLocator())
        axes.yaxis.set_minor_formatter(NullFormatter())
    if request.x_range:
        axes.set_xlim(*request.x_range)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_iteration(value)))
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_value(value)))
    axes.grid(color=GRID_COLOR, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=AXIS_COLOR, labelsize=6.5, length=0, pad=3)
    for side, spine in axes.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
        spine.set_color(GRID_COLOR)
    buffer = io.StringIO()
    figure.savefig(buffer, format='svg', metadata={'Date': None})
    svg = buffer.getvalue()
    return svg[svg.index('<svg'):]  # drop the XML prolog and doctype, the SVG is inlined


def log_splits(low: float, high: float) -> list[float]:
    """Log-axis ticks like `logSplits` in metrics.js: decades, or 1-2-5 steps within fewer than two decades."""
    powers = [10.0 ** exponent for exponent in range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1)]
    if len(powers) >= 2:
        return powers
    splits = [mantissa * 10.0 ** exponent for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
              for mantissa in (1, 2, 5) if low <= mantissa * 10.0 ** exponent <= high]
    return splits or [low, high]


def render_media(runs: list[RunSeries], request: ReportRequest) -> str:
    """One card per tag with a tile per run, each at its latest step ≤ `request.media_step`."""
    visuals = {run.run.path: reader.list_visuals(run.run_dir) for run in runs}
    tags = []
    for tag in dict.fromkeys(request.media_tags):
        entries = [(run, resolve_visual(visuals[run.run.path], tag, request.media_step)) for run in runs]
        if any(entry is not None for _, entry in entries):
            tags.append((tag, entries))
    if not tags:
        return ''

    entries_by_tag = dict(tags)
    blocks = []
    for group_path, keys in grouped(list(entries_by_tag)):
        cards = [render_media_card(tag, entries_by_tag[tag]) for tag in keys]
        blocks.append(f'<div class="group"><h3>{esc(group_path or "ungrouped")}</h3>{"".join(cards)}</div>')
    step = 'latest step' if request.media_step is None else f'step {request.media_step:,}'
    return f'<section class="media"><h2>Media</h2><p class="note">Each run at its latest visual up to the {esc(step)}.</p>{"".join(blocks)}</section>'


def render_media_card(tag: str, entries: list[tuple[RunSeries, tuple[int, reader.VisualFile] | None]]) -> str:
    tiles = []
    for run, entry in entries:
        caption = f'{swatch(run.run)}<span class="name">{esc(Path(run.run.path).name)}</span>'
        if entry is None:
            tiles.append(f'<figure class="tile missing"><div class="empty">No visual</div><figcaption>{caption}</figcaption></figure>')
            continue
        step, visual_file = entry
        file_path = run.run_dir / visual_file.path
        if visual_file.kind == 'video':
            media, detail = render_video_grid(file_path)
        else:
            media, detail = f'<img src="{esc(file_path.as_uri())}" alt="{esc(tag)}">', ''
        tiles.append(f'<figure class="tile">{media}<figcaption>{caption}<span class="stp num">step {step:,}</span>'
                     f'{f"<span class=detail>{esc(detail)}</span>" if detail else ""}</figcaption></figure>')
    kind = next((entry[1].kind for _, entry in entries if entry is not None), 'image')
    return f'<article class="card media-card {kind}"><div class="card-head">{card_title(tag)}<span class="tag">{kind}</span></div><div class="tiles">{"".join(tiles)}</div></article>'


def render_video_grid(video_path: Path) -> tuple[str, str]:
    """Return an `<img>` of the video's frames laid out row-major in a grid, and a caption detail."""
    try:
        frames = load_video(video_path)  # (T, H, W, 3)
    except OSError as error:
        return f'<div class="empty">{esc(error)}</div>', ''
    num_frames = len(frames)
    if num_frames > MAX_GRID_FRAMES:
        frames = frames[np.linspace(0, num_frames - 1, MAX_GRID_FRAMES).round().astype(int)]  # (MAX_GRID_FRAMES, H, W, 3)
    grid = tile_frames(frames)  # (grid_H, grid_W, 3)
    success, png = cv2.imencode('.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if not success:
        return '<div class="empty">Could not encode the frame grid</div>', ''
    shown = f'{len(frames)} of {num_frames} frames, evenly spaced' if num_frames > len(frames) else f'{num_frames} frames'
    return f'<img src="data:image/png;base64,{base64.b64encode(png.tobytes()).decode()}" alt="frames of {esc(video_path.name)}">', f'{shown}, row by row'


def tile_frames(frames: np.ndarray) -> np.ndarray:
    """Lay out frames row-major in a near-square grid with white gaps.

    Args:
        frames: Video frames, shape (T, H, W, 3), `uint8`.

    Returns:
        Grid image, shape (rows·(H'+gap)−gap, cols·(W'+gap)−gap, 3), where frames are
        downscaled to (H', W') when the grid would exceed `MAX_GRID_WIDTH_PX`.
    """
    num_frames, height, width = frames.shape[:3]
    columns = max(1, min(num_frames, round(math.sqrt(num_frames * height / width * 1.5))))  # grid aspect ≈ 3:2
    rows = math.ceil(num_frames / columns)
    scale = min(1.0, (MAX_GRID_WIDTH_PX - (columns - 1) * GRID_GAP_PX) / (columns * width))
    if scale < 1:
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        frames = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames])  # (T, H', W', 3)
    grid = np.full((rows * (height + GRID_GAP_PX) - GRID_GAP_PX, columns * (width + GRID_GAP_PX) - GRID_GAP_PX, 3), 255, dtype=np.uint8)
    for index, frame in enumerate(frames):
        top = (index // columns) * (height + GRID_GAP_PX)
        left = (index % columns) * (width + GRID_GAP_PX)
        grid[top:top + height, left:left + width] = frame
    return grid


def render_configs(runs: list[RunSeries]) -> str:
    blocks = []
    for run in runs:
        info = reader.read_run_info(run.run_dir, run.run.path)
        for config_file in info.config_files:
            try:
                text = (run.run_dir / config_file).read_text(errors='replace')
            except OSError as error:
                text = f'unreadable: {error}'
            blocks.append(f'<div class="config"><h3>{swatch(run.run)}{esc(run.run.path)} <span class="pre">{esc(config_file)}</span></h3><pre>{esc(text)}</pre></div>')
    if not blocks:
        return ''
    return f'<section class="configs"><h2>Configs</h2>{"".join(blocks)}</section>'

#---------------------------------------------------------------------
# page style
#---------------------------------------------------------------------

def page_css(orientation: str) -> str:
    return f'''
@page {{ size: A4 {orientation}; margin: 14mm 13mm 16mm;
  @bottom-left {{ content: string(doctitle); font: 7.5pt "DejaVu Sans", sans-serif; color: #656d78; }}
  @bottom-right {{ content: counter(page) " / " counter(pages); font: 7.5pt "DejaVu Sans", sans-serif; color: #656d78; }} }}
* {{ box-sizing: border-box; }}
body {{ font: 8.5pt/1.4 "IBM Plex Sans", "DejaVu Sans", sans-serif; color: #1a1d21; margin: 0; }}
.num, pre, .path {{ font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; }}
h1 {{ font-size: 16pt; font-weight: 600; margin: 0 0 1mm; string-set: doctitle content(); }}
h2 {{ font-size: 12pt; font-weight: 600; margin: 7mm 0 2mm; padding-bottom: 1mm; border-bottom: 1px solid #d9dde2; break-after: avoid; }}
h3 {{ font-size: 9pt; font-weight: 600; margin: 4mm 0 2mm; color: #2a3a5c; break-after: avoid; }}
section.summary h2 {{ margin-top: 5mm; }}
.meta, .note {{ color: #656d78; margin: 0 0 3mm; }}
.note {{ break-after: avoid; }}
.pre {{ color: #656d78; font-weight: 400; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ text-align: left; font-weight: 600; color: #656d78; border-bottom: 1px solid #b3bac4; padding: 1.2mm 2mm; }}
td {{ border-bottom: 1px solid #eef0f2; padding: 1mm 2mm; vertical-align: top; }}
td.num, th.num {{ text-align: right; }}
th.num {{ font-family: inherit; }}
table.values th:not(:first-child) {{ text-align: right; }}
table.runs td:first-child, table.runs th:first-child {{ width: 5mm; padding-right: 0; }}
tr {{ break-inside: avoid; }}
.sw {{ display: inline-block; width: 3.2mm; height: 1.2mm; margin: 0 1.2mm 0.4mm 0; vertical-align: middle; background: var(--c); border-radius: 0.3mm; }}
.sw.dash {{ background: linear-gradient(90deg, var(--c) 60%, transparent 60%); background-size: 1.6mm 100%; }}
.grid.charts {{ display: grid; grid-template-columns: repeat({CHART_COLUMNS[orientation]}, 1fr); gap: 3mm; }}
.card {{ border: 1px solid #d9dde2; border-radius: 1.5mm; padding: 2mm 2.5mm; break-inside: avoid; }}
.card-head {{ display: flex; align-items: baseline; gap: 2mm; margin-bottom: 1mm; }}
.ct {{ font-weight: 600; flex: 1; }}
.tag {{ font-size: 6.5pt; color: #656d78; border: 1px solid #d9dde2; border-radius: 0.8mm; padding: 0 1mm; }}
.plot svg {{ width: 100%; height: auto; display: block; }}
.card-foot {{ display: flex; flex-wrap: wrap; gap: 0.5mm 3mm; font-size: 7pt; color: #656d78; margin-top: 1mm; }}
.card-foot b {{ color: #1a1d21; font-weight: 500; }}
.media-card {{ margin-bottom: 3mm; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: 3mm; }}
.tile {{ margin: 0; break-inside: avoid; }}
.tile img {{ display: block; max-width: 100%; height: 42mm; width: auto; object-fit: contain; image-rendering: pixelated; border: 1px solid #eef0f2; }}
.video .tiles {{ display: grid; grid-template-columns: 1fr 1fr; }}
.video .tile img {{ width: 100%; height: auto; }}
.tile figcaption {{ font-size: 7pt; color: #656d78; margin-top: 1mm; display: flex; align-items: center; gap: 2mm; flex-wrap: wrap; }}
.tile .name {{ color: #1a1d21; }}
.tile .empty {{ height: 42mm; width: 42mm; display: flex; align-items: center; justify-content: center; background: #f5f6f8; color: #656d78; }}
.config {{ break-inside: avoid-page; }}
pre {{ font-size: 7pt; line-height: 1.35; background: #f5f6f8; border: 1px solid #eef0f2; border-radius: 1mm; padding: 2mm 3mm;
  white-space: pre-wrap; word-break: break-all; margin: 0 0 3mm; }}
'''
