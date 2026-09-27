"""Tests for `ldtrain.log.metric_tracker.MetricTracker`: the CSV contract and its threading."""
import threading
from pathlib import Path

import pytest

from ldtrain.log.metric_tracker import MetricTracker


def test_csv_layout_and_empty_cells(tmp_path: Path) -> None:
    tracker = MetricTracker(tmp_path / 'm' / 'metrics.csv')
    tracker.track('train/loss', 1, 0)
    tracker.track('val/psnr', 2.5, 0)
    tracker.track('train/loss', 0.5, 1)
    tracker.close()
    assert (tmp_path / 'm' / 'metrics.csv').read_text() == 'iteration,train/loss,val/psnr\n0,1.0,2.5\n1,0.5,\n'


def test_append_keeps_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.csv'
    first = MetricTracker(path)
    first.track('a', 1.0, 0)
    first.close()
    with pytest.raises(FileExistsError):
        MetricTracker(path)
    second = MetricTracker(path, exist_ok=True)
    second.track('a', 2.0, 1)
    second.close()
    assert path.read_text() == 'iteration,a\n0,1.0\n1,2.0\n'


def test_new_column_after_rows_rewrites_header(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.csv'
    tracker = MetricTracker(path, buffer_size=1)
    tracker.track('a', 1.0, 0)
    tracker.track('a', 2.0, 1)  # hands off step 0 to the writer
    tracker.flush(wait=True)
    assert path.read_text() == 'iteration,a\n0,1.0\n1,2.0\n'
    tracker.track('b', 3.0, 2)
    tracker.close()
    assert path.read_text() == 'iteration,a,b\n0,1.0,\n1,2.0,\n2,,3.0\n'
    assert not path.with_name('metrics.csv.tmp').exists()


def test_flush_and_close_are_safe_to_repeat(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.csv'
    tracker = MetricTracker(path)
    tracker.flush(wait=True)  # nothing queued
    tracker.track('a', 1.0, 0)
    tracker.flush(wait=True)
    assert path.read_text().splitlines() == ['iteration,a', '0,1.0']
    tracker.close()
    tracker.close()
    assert path.read_text().splitlines() == ['iteration,a', '0,1.0']


def test_writer_error_is_raised_on_next_hand_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = MetricTracker(tmp_path / 'metrics.csv', buffer_size=1)
    tracker.track('a', 1.0, 0)

    def fail(rows: list) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(tracker, '_MetricTracker__rewrite_file', fail)  # the first column always rewrites
    tracker.track('a', 2.0, 1)  # hands off step 0, whose write fails on the thread
    with pytest.raises(RuntimeError, match='writing metrics'):
        tracker.flush(wait=True)


def test_concurrent_tracking_writes_every_row(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.csv'
    tracker = MetricTracker(path, buffer_size=10)

    def worker(offset: int) -> None:
        for ii in range(50):
            tracker.track('a', float(offset * 50 + ii), offset * 50 + ii)

    threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    tracker.close()
    rows = path.read_text().splitlines()[1:]
    assert len(rows) == 200
    assert sorted(int(row.split(',')[0]) for row in rows) == list(range(200))
