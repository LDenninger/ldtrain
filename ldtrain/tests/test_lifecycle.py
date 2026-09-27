"""Tests for `initialize`, `finish`, the package surface and `catch_exceptions`, over run directories on disk."""
from pathlib import Path

import pytest

import ldtrain
from ldtrain.tests.conftest import flushed_text


def test_package_surface() -> None:
    assert isinstance(ldtrain.__version__, str) and ldtrain.__version__
    namespace: dict = {}
    exec('from ldtrain import *', namespace)
    assert set(ldtrain.__all__) <= set(namespace)
    assert ldtrain.run is ldtrain.run_config.run and ldtrain.dist is ldtrain.distributed.dist


def test_initialize_writes_the_run_layout_and_reuses_it(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    ldtrain.initialize(run_dir=run_dir, color=False)
    ldtrain.info('first session')
    ldtrain.log_metrics({'loss': 1.0}, step=0)
    ldtrain.finish()
    assert ldtrain.run.root_directory == str(run_dir)
    assert sorted(p.name for p in run_dir.iterdir()) == ['logs', 'metrics']
    ldtrain.initialize(run_dir=run_dir, color=False)  # resume: append, warn
    ldtrain.info('second session')
    ldtrain.log_metrics({'loss': 0.5}, step=1)
    ldtrain.finish()
    text = flushed_text(run_dir / 'logs' / 'log.txt')
    assert 'first session' in text and 'second session' in text and 'already exists' in text
    assert (run_dir / 'metrics' / 'metrics.csv').read_text() == 'iteration,loss\n0,1.0\n1,0.5\n'


def test_initialize_without_run_dir_detaches_the_run(tmp_path: Path) -> None:
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    ldtrain.initialize(color=False)
    ldtrain.info('console only')
    ldtrain.log_metrics({'loss': 1.0}, step=0)
    ldtrain.finish()
    assert ldtrain.run.root_directory == ''
    assert 'console only' not in flushed_text(tmp_path / 'run' / 'logs' / 'log.txt')
    assert (tmp_path / 'run' / 'metrics' / 'metrics.csv').read_text() == 'iteration\n'  # header only


def test_finish_is_idempotent(tmp_path: Path) -> None:
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    ldtrain.log_metrics({'loss': 1.0}, step=0)
    ldtrain.finish()
    ldtrain.finish()
    assert (tmp_path / 'run' / 'metrics' / 'metrics.csv').read_text().count('\n') == 2


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
