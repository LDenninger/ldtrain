"""Tests for `ldtrain.viewer.reader` against run directories on disk."""
import os
from pathlib import Path

import pytest

from ldtrain.log.metric_tracker import MetricTracker
from ldtrain.viewer import reader
from ldtrain.viewer.reader import PathOutsideRoot

#---------------------------------------------------------------------
# helpers
#---------------------------------------------------------------------

def make_run(run_dir: Path, csv_text: str | None = None, log_text: str | None = None) -> Path:
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    if csv_text is not None:
        (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (run_dir / 'metrics' / 'metrics.csv').write_text(csv_text)
    if log_text is not None:
        (run_dir / 'logs' / 'log.txt').write_text(log_text)
    return run_dir


def append(path: Path, text: str) -> None:
    with path.open('a') as dst:
        dst.write(text)

#---------------------------------------------------------------------
# paths and tree
#---------------------------------------------------------------------

def test_resolve_inside_rejects_escapes(tmp_path: Path) -> None:
    root = (tmp_path / 'root').resolve()
    make_run(root / 'a')
    (tmp_path / 'secret.txt').write_text('x')
    os.symlink(tmp_path, root / 'link_out')

    assert reader.resolve_inside(root, 'a') == root / 'a'
    for bad in ['../secret.txt', 'a/../../secret.txt', '/etc/passwd', 'link_out/secret.txt']:
        with pytest.raises(PathOutsideRoot):
            reader.resolve_inside(root, bad)


def test_scan_tree_finds_nested_runs_and_prunes_empty_folders(tmp_path: Path) -> None:
    make_run(tmp_path / 'proj' / 'sweep' / 'run_b', csv_text='iteration,loss\n0,1.0\n7,0.5\n')
    make_run(tmp_path / 'proj' / 'sweep' / 'run_a')
    make_run(tmp_path / 'proj' / 'single')
    (tmp_path / 'empty' / 'nothing').mkdir(parents=True)
    (tmp_path / '.hidden' / 'logs').mkdir(parents=True)

    tree = reader.scan_tree(tmp_path)

    assert tree.path == '' and tree.num_runs == 3
    assert [child.name for child in tree.children] == ['proj']
    proj = tree.children[0]
    assert [child.name for child in proj.children] == ['single', 'sweep']
    sweep = proj.children[1]
    assert [(run.name, run.path, run.is_run) for run in sweep.children] == [
        ('run_a', 'proj/sweep/run_a', True), ('run_b', 'proj/sweep/run_b', True)]
    assert sweep.children[1].last_iteration == 7.0
    assert sweep.children[0].last_iteration is None

#---------------------------------------------------------------------
# metrics
#---------------------------------------------------------------------

def test_read_metrics_full_and_incremental(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run', csv_text='iteration,loss,lr\n0,1.0,0.1\n1,0.5,\n')
    csv_path = run / 'metrics' / 'metrics.csv'

    full = reader.read_metrics(run)
    assert full.reset and full.columns == ['iteration', 'loss', 'lr']
    assert full.data == {'iteration': [0.0, 1.0], 'loss': [1.0, 0.5], 'lr': [0.1, None]}

    append(csv_path, '2,0.25,0.1\n3,nan,inf\n4,0.1')  # last row still being written
    chunk = reader.read_metrics(run, full.offset, full.file_id)
    assert not chunk.reset
    assert chunk.data == {'iteration': [2.0, 3.0], 'loss': [0.25, None], 'lr': [0.1, None]}

    append(csv_path, ',0.1\n')
    tail = reader.read_metrics(run, chunk.offset, chunk.file_id)
    assert tail.data == {'iteration': [4.0], 'loss': [0.1], 'lr': [0.1]}
    assert tail.offset == csv_path.stat().st_size


def test_read_metrics_resets_when_tracker_adds_a_column(tmp_path: Path) -> None:
    csv_path = tmp_path / 'run' / 'metrics' / 'metrics.csv'
    (tmp_path / 'run' / 'logs').mkdir(parents=True)
    tracker = MetricTracker(csv_path)
    tracker.track('loss', 1.0, step=0)
    tracker.flush(wait=True)
    first = reader.read_metrics(tmp_path / 'run')

    tracker.track('loss', 0.5, step=1)
    tracker.track('val/psnr', 20.0, step=1)  # new column: the tracker rewrites the file
    tracker.close()
    second = reader.read_metrics(tmp_path / 'run', first.offset, first.file_id)

    assert second.reset and second.file_id != first.file_id
    assert second.columns == ['iteration', 'loss', 'val/psnr']
    assert second.data['val/psnr'] == [None, 20.0]


def test_read_metrics_missing_file_warns(tmp_path: Path) -> None:
    chunk = reader.read_metrics(make_run(tmp_path / 'run'))
    assert chunk.warning is not None and chunk.data == {}

#---------------------------------------------------------------------
# visuals and logs
#---------------------------------------------------------------------

def test_list_visuals_keeps_tag_subfolders_and_includes_since_step(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run')
    for step, files in {5: ['val/pred.png', 'val/rollout.webm'], 10: ['train/in.jpg', 'notes.txt'], 20: []}.items():
        step_dir = run / 'visuals' / f'{step:08d}'
        step_dir.mkdir(parents=True)
        for file_name in files:
            (step_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
            (step_dir / file_name).write_bytes(b'x')

    steps = reader.list_visuals(run)
    assert [step.step for step in steps] == [5, 10, 20]
    assert [(f.tag, f.kind, f.path) for f in steps[0].files] == [
        ('val/pred', 'image', 'visuals/00000005/val/pred.png'), ('val/rollout', 'video', 'visuals/00000005/val/rollout.webm')]
    assert [f.tag for f in steps[1].files] == ['train/in']
    assert [step.step for step in reader.list_visuals(run, since_step=10)] == [10, 20]


def test_read_log_tail_and_follow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader, 'LOG_CHUNK_BYTES', 64)
    lines = ''.join(f'line {ii:03d}\n' for ii in range(20))  # 9 bytes per line
    run = make_run(tmp_path / 'run', log_text=lines)
    log_path = run / 'logs' / 'log.txt'

    tail = reader.read_log(run)
    assert tail.truncated and tail.text.startswith('line ') and tail.text.endswith('line 019\n')
    assert tail.offset == log_path.stat().st_size

    append(log_path, 'line 020\npartial')
    follow = reader.read_log(run, offset=tail.offset)
    assert follow.text == 'line 020\n' and not follow.truncated


def test_read_run_info_lists_ranks_and_configs(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run', csv_text='iteration\n', log_text='x\n')
    (run / 'logs' / 'log.rank2.txt').write_text('y\n')
    (run / 'configs').mkdir()
    (run / 'configs' / 'config.yaml').write_text('lr: 0.1\n')

    info = reader.read_run_info(run, 'run')
    assert info.ranks == [0, 2] and info.config_files == ['configs/config.yaml']
    assert info.has_metrics and not info.has_visuals

#---------------------------------------------------------------------
# edge cases
#---------------------------------------------------------------------

def test_scan_tree_skips_hidden_dirs_and_symlink_loops(tmp_path: Path) -> None:
    make_run(tmp_path / 'proj' / 'run', log_text='x\n')
    make_run(tmp_path / '.hidden' / 'run', log_text='x\n')
    (tmp_path / 'proj' / 'loop').symlink_to(tmp_path, target_is_directory=True)
    tree = reader.scan_tree(tmp_path)
    assert tree.num_runs == 1
    assert [child.name for child in tree.children] == ['proj']
    assert [child.name for child in tree.children[0].children] == ['run']


def test_read_last_iteration_on_thin_files(tmp_path: Path) -> None:
    empty = tmp_path / 'empty.csv'
    empty.write_text('')
    header_only = tmp_path / 'header.csv'
    header_only.write_text('iteration,a\n')
    rows = tmp_path / 'rows.csv'
    rows.write_text('iteration,a\n0,1.0\n12,2.0\n')
    assert reader.read_last_iteration(empty) is None
    assert reader.read_last_iteration(header_only) is None
    assert reader.read_last_iteration(rows) == 12.0
    assert reader.read_last_iteration(tmp_path / 'missing.csv') is None


@pytest.mark.parametrize('cell, expected', [('1.5', 1.5), ('', None), ('nan', None), ('inf', None), ('-inf', None), ('abc', None), ('1e3', 1000.0)])
def test_parse_value(cell: str, expected: float | None) -> None:
    assert reader.parse_value(cell) == expected


def test_read_metrics_turns_non_finite_cells_into_none(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run', csv_text='iteration,a\n0,nan\n1,inf\n2,\n3,4.0\n')
    chunk = reader.read_metrics(run)
    assert chunk.columns == ['iteration', 'a'] and chunk.data['a'] == [None, None, None, 4.0]


def test_list_visuals_ignores_unknown_files_and_bad_folders(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run')
    (run / 'visuals' / '00000001').mkdir(parents=True)
    (run / 'visuals' / '00000001' / 'notes.txt').write_text('x')
    (run / 'visuals' / '00000001' / 'a.png').write_bytes(b'x')
    (run / 'visuals' / 'not_a_step').mkdir()
    (run / 'visuals' / 'not_a_step' / 'b.png').write_bytes(b'x')
    steps = reader.list_visuals(run)
    assert [(step.step, [f.tag for f in step.files]) for step in steps] == [(1, ['a'])]
    assert reader.list_visuals(make_run(tmp_path / 'bare')) == []


def test_read_log_for_missing_rank_and_offset_past_end(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run', log_text='one\ntwo\n')
    assert reader.read_log(run, rank=3) == reader.LogChunk(text='', offset=0)
    chunk = reader.read_log(run, offset=999)  # a truncated file restarts from the tail
    assert chunk.text == 'one\ntwo\n' and chunk.offset == 8 and not chunk.truncated


def test_read_run_info_without_configs_or_logs(tmp_path: Path) -> None:
    run = make_run(tmp_path / 'run', csv_text='iteration\n')
    info = reader.read_run_info(run, 'run')
    assert info == reader.RunInfo(path='run', ranks=[], config_files=[], has_metrics=True, has_visuals=False)
