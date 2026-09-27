"""Tests for `ldtrain.catch_exceptions` over a run directory on disk."""
from pathlib import Path

import pytest

import ldtrain


@pytest.fixture
def log_path(tmp_path: Path):
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    yield tmp_path / 'run' / 'logs' / 'log.txt'
    ldtrain.initialize()  # detach logging from the temporary run


def read_log(log_path: Path) -> str:
    ldtrain.finish()
    return log_path.read_text()


def test_non_critical_logs_and_continues(log_path: Path) -> None:
    @ldtrain.catch_exceptions(critical=False)
    def fail() -> int:
        raise ValueError('bad batch')

    assert fail() is None
    text = read_log(log_path)
    assert 'ERROR' in text and 'ValueError: bad batch' in text and 'Continuing.' in text
    assert 'Traceback' in text and 'in fail' in text


def test_critical_logs_finishes_and_exits(log_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        with ldtrain.catch_exceptions():
            ldtrain.log_metrics({'loss': 1.0}, step=0)
            raise RuntimeError('nan loss')
    assert info.value.code == 1
    text = log_path.read_text()  # already flushed by the guard's finish()
    assert 'CRITICAL' in text and 'RuntimeError: nan loss' in text and 'Stopping the run.' in text
    assert (log_path.parent.parent / 'metrics' / 'metrics.csv').read_text().splitlines()[1] == '0,1.0'


def test_innermost_guard_decides(log_path: Path) -> None:
    seen = []
    with ldtrain.catch_exceptions(critical=False):
        with ldtrain.catch_exceptions(critical=False):
            raise ValueError('inner')
        seen.append('after inner')
    assert seen == ['after inner']
    assert read_log(log_path).count('Continuing.') == 1

    with pytest.raises(SystemExit):
        with ldtrain.catch_exceptions(critical=False):
            with ldtrain.catch_exceptions(critical=True):
                raise ValueError('inner critical')


def test_interrupts_pass_through(log_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        with ldtrain.catch_exceptions(critical=False):
            raise KeyboardInterrupt
    assert 'Continuing.' not in read_log(log_path)
