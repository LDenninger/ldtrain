"""Read-only access to a tree of ldtrain run directories, for the viewer.

A run is a directory holding at least one of the `RunConfig` subdirectories `metrics/`,
`logs/` or `visuals/`. `scan_tree` finds them below a root. `read_metrics` and `read_log`
read incrementally from a byte offset and only return complete lines, so a file being
appended to is never read half-written. `list_visuals` lists the per-iteration media
folders. Every path taken from a request goes through `resolve_inside`, which rejects
anything that resolves outside the root.
"""
from __future__ import annotations

import csv
import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from ldtrain.core.run import RunConfig

METRICS_FILE_NAME = 'metrics.csv'
LOG_CHUNK_BYTES = 256 * 1024
TAIL_BYTES = 4096
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
VIDEO_SUFFIXES = {'.mp4', '.webm', '.mov'}
CONFIG_SUFFIXES = {'.yaml', '.yml', '.json', '.txt'}

_LAYOUT = RunConfig()
RUN_SUBDIRS = (_LAYOUT.PREFERRED_METRICS_DIR, _LAYOUT.PREFERRED_LOG_DIR, _LAYOUT.PREFERRED_VISUAL_DIR)


class PathOutsideRoot(ValueError):
    """Raised when a requested path resolves outside the viewer root."""

#---------------------------------------------------------------------
# paths
#---------------------------------------------------------------------

def resolve_inside(base_dir: Path, relative_path: str) -> Path:
    """Resolve `relative_path` below `base_dir`, following symlinks.

    Args:
        base_dir: Directory the result must stay inside, already resolved.
        relative_path: Path from a request, `/` separated. Empty means `base_dir`.

    Returns:
        The resolved absolute path.

    Raises:
        PathOutsideRoot: If the path is absolute or resolves outside `base_dir`, e.g.
            through `..` or a symlink.
    """
    if Path(relative_path).is_absolute():
        raise PathOutsideRoot(relative_path)
    resolved = (base_dir / relative_path).resolve()
    if resolved != base_dir and base_dir not in resolved.parents:
        raise PathOutsideRoot(relative_path)
    return resolved


def is_run_dir(directory: Path) -> bool:
    return any((directory / name).is_dir() for name in RUN_SUBDIRS)

#---------------------------------------------------------------------
# tree
#---------------------------------------------------------------------

@dataclass
class TreeNode:
    """One folder or run below the root.

    Attributes:
        name: Directory name, the root's own name for the root.
        path: Path relative to the root, `/` separated, empty for the root.
        is_run: Whether the directory is a run. Runs have no children.
        children: Subfolders and runs holding at least one run, sorted by name.
        mtime: For runs, the latest modification time of `metrics.csv` and the rank-0
            log in seconds since the epoch, None if neither exists.
        last_iteration: For runs, the iteration of the last complete metrics row.
        num_runs: Runs at or below this node.
    """
    name: str
    path: str
    is_run: bool
    children: list[TreeNode] = field(default_factory=list)
    mtime: float | None = None
    last_iteration: float | None = None
    num_runs: int = 0


def scan_tree(root_dir: Path) -> TreeNode:
    """Find every run below `root_dir`, keeping only folders that lead to a run.

    Symlinked directories and hidden directories are not entered, so a link cycle
    cannot recurse forever.
    """
    root_dir = root_dir.resolve()
    return scan_directory(root_dir, root_dir) or TreeNode(name=root_dir.name, path='', is_run=False)


def scan_directory(directory: Path, root_dir: Path) -> TreeNode | None:
    relative = directory.relative_to(root_dir).as_posix()
    path = '' if relative == '.' else relative
    if is_run_dir(directory):
        metrics_path = directory / _LAYOUT.PREFERRED_METRICS_DIR / METRICS_FILE_NAME
        mtimes = [p.stat().st_mtime for p in (metrics_path, directory / _LAYOUT.PREFERRED_LOG_DIR / 'log.txt') if p.is_file()]
        return TreeNode(name=directory.name, path=path, is_run=True, mtime=max(mtimes, default=None),
                        last_iteration=read_last_iteration(metrics_path), num_runs=1)

    children: list[TreeNode] = []
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return None
    for entry in entries:
        if entry.name.startswith('.') or not entry.is_dir(follow_symlinks=False):
            continue
        child = scan_directory(Path(entry.path), root_dir)
        if child is not None:
            children.append(child)
    if not children:
        return None
    return TreeNode(name=directory.name, path=path, is_run=False, children=children,
                    num_runs=sum(child.num_runs for child in children))


def read_last_iteration(metrics_path: Path) -> float | None:
    """Return the first cell of the last complete row of `metrics_path`, None if absent."""
    try:
        with metrics_path.open('rb') as src:
            size = src.seek(0, os.SEEK_END)
            src.seek(max(0, size - TAIL_BYTES))
            lines = src.read().split(b'\n')[:-1]  # drop the incomplete tail after the last newline
    except OSError:
        return None
    if len(lines) < 2 and size <= TAIL_BYTES:  # only a header
        return None
    return parse_value(lines[-1].split(b',', 1)[0].decode('utf-8', errors='replace')) if lines else None

#---------------------------------------------------------------------
# metrics
#---------------------------------------------------------------------

@dataclass
class MetricsChunk:
    """Metric rows appended since a byte offset.

    Attributes:
        columns: Header of the CSV, `iteration` first.
        data: Values per column for the returned rows, None for empty or non-finite cells.
        offset: Byte offset after the last complete row, to pass to the next call.
        file_id: Identity of the file (`device:inode`). It changes when `MetricTracker`
            rewrites the file to add a column.
        reset: Whether `data` holds the whole file, replacing what the caller has.
        warning: Why no data could be read, None on success.
    """
    columns: list[str]
    data: dict[str, list[float | None]]
    offset: int
    file_id: str
    reset: bool
    warning: str | None = None


def read_metrics(run_dir: Path, offset: int = 0, file_id: str = '') -> MetricsChunk:
    """Read the rows of `<run_dir>/metrics/metrics.csv` written after `offset`.

    The whole file is returned, with `reset` set, when `offset` is 0, when `file_id`
    no longer matches the file, or when the file shrank below `offset`.

    Args:
        run_dir: Run directory.
        offset: Byte offset returned by the previous call, 0 for a full read.
        file_id: `file_id` returned by the previous call.
    """
    metrics_path = run_dir / _LAYOUT.PREFERRED_METRICS_DIR / METRICS_FILE_NAME
    try:
        with metrics_path.open('rb') as src:
            stat = os.fstat(src.fileno())
            current_id = f'{stat.st_dev}:{stat.st_ino}'
            header_line = src.readline()
            if not header_line.endswith(b'\n'):
                return MetricsChunk(columns=[], data={}, offset=0, file_id=current_id, reset=True)
            reset = offset <= 0 or file_id != current_id or offset > stat.st_size
            start = len(header_line) if reset else offset
            src.seek(start)
            body = src.read(stat.st_size - start)
    except OSError as error:
        return MetricsChunk(columns=[], data={}, offset=0, file_id='', reset=True, warning=f'{metrics_path.name} unreadable: {error}')

    complete = body[:body.rfind(b'\n') + 1]  # hold back a row still being written
    columns = next(csv.reader([header_line.decode('utf-8', errors='replace')]))
    data: dict[str, list[float | None]] = {column: [] for column in columns}
    for row in csv.reader(io.StringIO(complete.decode('utf-8', errors='replace'))):
        if not row:
            continue
        for ii, column in enumerate(columns):
            data[column].append(parse_value(row[ii]) if ii < len(row) else None)
    return MetricsChunk(columns=columns, data=data, offset=start + len(complete), file_id=current_id, reset=reset)


def parse_value(cell: str) -> float | None:
    """Parse one CSV cell, None for empty, unparseable, NaN or infinite values.

    Non-finite values become None because JSON cannot carry them.
    """
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if math.isfinite(value) else None

#---------------------------------------------------------------------
# visuals
#---------------------------------------------------------------------

@dataclass
class VisualFile:
    tag: str
    kind: str
    path: str


@dataclass
class VisualStep:
    step: int
    files: list[VisualFile]


def list_visuals(run_dir: Path, since_step: int = 0) -> list[VisualStep]:
    """List the media of every `visuals/<step>/` folder with a step ≥ `since_step`.

    The bound is inclusive so a caller re-reads the newest step, whose files may still
    have been written during its previous call.

    Returns:
        Steps in ascending order. Each file carries its tag (path below the step
        folder without suffix), its kind (`image` or `video`) and its path relative to
        `run_dir`.
    """
    visual_dir = run_dir / _LAYOUT.PREFERRED_VISUAL_DIR
    try:
        step_dirs = [entry for entry in os.scandir(visual_dir) if entry.name.isdigit() and entry.is_dir(follow_symlinks=False)]
    except OSError:
        return []

    steps: list[VisualStep] = []
    for step_dir in sorted(step_dirs, key=lambda entry: int(entry.name)):
        step = int(step_dir.name)
        if step < since_step:
            continue
        files: list[VisualFile] = []
        for file_path in sorted(Path(step_dir.path).rglob('*')):
            suffix = file_path.suffix.lower()
            kind = 'image' if suffix in IMAGE_SUFFIXES else 'video' if suffix in VIDEO_SUFFIXES else None
            if kind is None or not file_path.is_file():
                continue
            tag = file_path.relative_to(step_dir.path).with_suffix('').as_posix()
            files.append(VisualFile(tag=tag, kind=kind, path=file_path.relative_to(run_dir).as_posix()))
        steps.append(VisualStep(step=step, files=files))
    return steps

#---------------------------------------------------------------------
# logs and run info
#---------------------------------------------------------------------

@dataclass
class LogChunk:
    """Log text after a byte offset.

    Attributes:
        text: Complete lines only, decoded as UTF-8 with replacement.
        offset: Byte offset after `text`, to pass to the next call.
        truncated: Whether the first read skipped the start of a long log.
    """
    text: str
    offset: int
    truncated: bool = False


def log_path(run_dir: Path, rank: int) -> Path:
    file_name = 'log.txt' if rank == 0 else f'log.rank{rank}.txt'
    return run_dir / _LAYOUT.PREFERRED_LOG_DIR / file_name


def read_log(run_dir: Path, rank: int = 0, offset: int | None = None) -> LogChunk:
    """Read at most `LOG_CHUNK_BYTES` of complete log lines of `rank` after `offset`.

    Args:
        run_dir: Run directory.
        rank: Rank whose log to read, 0 for `log.txt`.
        offset: Byte offset returned by the previous call. None starts at the last
            `LOG_CHUNK_BYTES` of the file, at a line boundary.
    """
    try:
        with log_path(run_dir, rank).open('rb') as src:
            size = src.seek(0, os.SEEK_END)
            truncated = False
            if offset is None or offset > size:
                offset = max(0, size - LOG_CHUNK_BYTES)
                truncated = offset > 0
            src.seek(offset)
            chunk = src.read(LOG_CHUNK_BYTES)
    except OSError:
        return LogChunk(text='', offset=0)

    if truncated:  # start at the first full line
        first_newline = chunk.find(b'\n')
        offset += first_newline + 1
        chunk = chunk[first_newline + 1:]
    complete = chunk[:chunk.rfind(b'\n') + 1]
    return LogChunk(text=complete.decode('utf-8', errors='replace'), offset=offset + len(complete), truncated=truncated)


@dataclass
class RunInfo:
    """What a run holds, for the viewer to decide which cards to show.

    Attributes:
        path: Run path relative to the root.
        ranks: Ranks with a log file, ascending.
        config_files: Paths of config files relative to the run directory.
        has_metrics: Whether `metrics/metrics.csv` exists.
        has_visuals: Whether `visuals/` exists.
    """
    path: str
    ranks: list[int]
    config_files: list[str]
    has_metrics: bool
    has_visuals: bool


def read_run_info(run_dir: Path, path: str) -> RunInfo:
    log_dir = run_dir / _LAYOUT.PREFERRED_LOG_DIR
    ranks = [0] if (log_dir / 'log.txt').is_file() else []
    for log_file in log_dir.glob('log.rank*.txt'):
        rank_str = log_file.name[len('log.rank'):-len('.txt')]
        if rank_str.isdigit():
            ranks.append(int(rank_str))

    config_dir = run_dir / _LAYOUT.PREFERRED_CONFIG_DIR
    config_files = sorted(p.relative_to(run_dir).as_posix() for p in config_dir.rglob('*')
                          if p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES)
    return RunInfo(path=path, ranks=sorted(ranks), config_files=config_files,
                   has_metrics=(run_dir / _LAYOUT.PREFERRED_METRICS_DIR / METRICS_FILE_NAME).is_file(),
                   has_visuals=(run_dir / _LAYOUT.PREFERRED_VISUAL_DIR).is_dir())
