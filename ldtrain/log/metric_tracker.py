"""Scalar metric tracking for a training run, stored as a wide CSV file with one row per iteration.

`tracker` is the global `MetricTracker` of the run, writing `<run_dir>/metrics/metrics.csv`.
`initialize()` replaces it, and it is None before that, without a run directory and on
ranks other than 0, since a tracker must be the single writer of its file.
"""
from __future__ import annotations

import atexit
import csv
import os
import queue
import threading
from pathlib import Path

from ldtrain.distributed import dist
from ldtrain.run_config import run

METRICS_FILE_NAME = 'metrics.csv'


#---------------------------------------------------------------------
# metric tracking
#---------------------------------------------------------------------

class MetricTracker:
    """Scalar metric store backed by a wide CSV file with one row per iteration.

    The first column is `iteration` and every further column is one metric, in the order
    the metrics were first written. Values tracked for the same iteration share one row,
    and a metric not tracked at an iteration leaves its cell empty. `track` only puts the
    value on a thread-safe queue. Once `buffer_size` iterations are queued, the first value
    of the next iteration hands the queued values to a background thread that writes them.
    The next hand-off, or `close`, first joins that thread and re-raises any error it hit.
    A metric first seen after rows were written extends the header, which rewrites the
    file once. Read the file back with ``pd.read_csv(path, index_col='iteration')``.

    Args:
        file_path: CSV file to write. Missing parent directories are created.
        exist_ok: Append to an existing file, keeping its rows and metric columns. Rows
            already in the file are not checked against new ones, so a run that restarts
            from an earlier checkpoint writes those iterations a second time. Defaults to
            False.
        buffer_size: Number of iterations queued before they are handed to the writer
            thread. Defaults to 100.

    Raises:
        FileExistsError: The file exists and `exist_ok` is False.
    """

    ITERATION_COLUMN = 'iteration'

    def __init__(self, file_path: Path, exist_ok: bool = False, buffer_size: int = 100) -> None:
        self.file_path = Path(file_path)
        if self.file_path.exists() and not exist_ok:
            raise FileExistsError(f'{self.file_path} exists, pass exist_ok=True')
        self.buffer_size = buffer_size
        self._queue: queue.Queue[tuple[int, str, float]] = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._writer_error: Exception | None = None
        self._last_step: int | None = None
        self._queued_steps = 0
        self._hand_off_lock = threading.Lock()
        self._closed = False

        #--- file state, owned by the writer thread once the first hand-off happened ---
        self._columns: list[str] = []
        if self.file_path.exists():
            with self.file_path.open(newline='') as src:
                fieldnames = csv.DictReader(src).fieldnames
            if fieldnames:
                self._columns = list(fieldnames)[1:]
                self._file = self.file_path.open('a', newline='')
                self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, restval='')
                return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.__rewrite_file([])

    #---------------------------------------------------------------------
    # interface
    #---------------------------------------------------------------------

    def track(self, name: str, value: int | float, step: int) -> None:
        """Queue one scalar under `name` at iteration `step`, handing off full buffers.

        Args:
            name: Metric name and CSV column, e.g. ``'train/psnr'``.
            value: Scalar value, stored as float.
            step: Training iteration the value belongs to. An iteration whose row is
                already on disk gets a second row.

        Raises:
            RuntimeError: The previous background write failed.
        """
        with self._hand_off_lock:
            if step != self._last_step:
                if self._queued_steps >= self.buffer_size:
                    self.__hand_off(wait=False)
                self._queued_steps += 1
                self._last_step = step
            self._queue.put((step, name, float(value)))

    def flush(self, wait: bool = False) -> None:
        """Join the previous writer thread and hand every queued value to a new one.

        Args:
            wait: Block until the new thread has written the values. Defaults to False.

        Raises:
            RuntimeError: The previous background write failed.
        """
        with self._hand_off_lock:
            self.__hand_off(wait)

    def close(self) -> None:
        """Write every queued value and close the file. Calling it again does nothing.

        Raises:
            RuntimeError: A background write failed.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.flush(wait=True)
        finally:
            self._file.close()

    #---------------------------------------------------------------------
    # helpers
    #---------------------------------------------------------------------

    def __hand_off(self, wait: bool) -> None:
        """Join the previous writer thread and start one on every queued value. Caller holds the lock."""
        self.__join_writer()
        self._queued_steps = 0
        num_items = self._queue.qsize()
        if num_items == 0:
            return
        self._writer_thread = threading.Thread(target=self.__write_items, args=(num_items,), name='metric-tracker-writer')
        self._writer_thread.start()
        if wait:
            self.__join_writer()

    def __join_writer(self) -> None:
        """Wait for the running writer thread and re-raise the error it hit, if any."""
        if self._writer_thread is not None:
            self._writer_thread.join()
            self._writer_thread = None
        if self._writer_error is not None:
            error, self._writer_error = self._writer_error, None
            raise RuntimeError(f'writing metrics to {self.file_path} failed') from error

    def __write_items(self, num_items: int) -> None:
        """Take `num_items` values off the queue and write them as one row per iteration.

        Runs on the writer thread. An exception is stored for `__join_writer` to re-raise.
        """
        try:
            rows: dict[int, dict[str, float | int]] = {}
            for _ in range(num_items):
                step, name, value = self._queue.get_nowait()
                rows.setdefault(step, {self.ITERATION_COLUMN: step})[name] = value

            names = dict.fromkeys(name for row in rows.values() for name in row if name != self.ITERATION_COLUMN)
            new_columns = [name for name in names if name not in self._columns]
            if not new_columns:
                self._writer.writerows(rows.values())
                self._file.flush()
                return
            self._columns.extend(new_columns)
            self._file.close()
            with self.file_path.open(newline='') as src:
                old_rows = list(csv.DictReader(src))
            self.__rewrite_file([*old_rows, *rows.values()])
        except Exception as error:
            self._writer_error = error

    def __rewrite_file(self, rows: list[dict]) -> None:
        """Replace the file with the current header and `rows`, then reopen it for appending."""
        fieldnames = [self.ITERATION_COLUMN, *self._columns]
        tmp_path = self.file_path.with_name(f'{self.file_path.name}.tmp')
        with tmp_path.open('w', newline='') as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, self.file_path)

        self._file = self.file_path.open('a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, restval='')


#---------------------------------------------------------------------
# global tracker
#---------------------------------------------------------------------

tracker: MetricTracker | None = None


def initialize() -> None:
    """Open the metrics CSV of `run.root_directory`, appending to an existing one.

    Closes the tracker of a previous call first. Does nothing without a run directory or
    on ranks other than 0.
    """
    global tracker
    close()
    if run.root_directory and dist.is_master:
        tracker = MetricTracker(Path(run.metrics_dir) / METRICS_FILE_NAME, exist_ok=True)


def close() -> None:
    """Write every buffered metric and close the CSV. Registered to run at exit."""
    global tracker
    if tracker is not None:
        closing, tracker = tracker, None
        closing.close()


atexit.register(close)
