"""Tests for `ldtrain.log.console`: handlers, filters, formatting, pausing and the async streams."""
import logging
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ldtrain
from ldtrain.log import console
from ldtrain.tests.conftest import flushed_text


def test_levels_route_to_console_and_file(tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False, log_level='warning', log_level_file='debug')
    ldtrain.debug('to file only')
    ldtrain.info('info line')
    ldtrain.warning('warn line')
    ldtrain.error('error line')
    console.flush_async_streams()
    out = capfd.readouterr().out
    assert 'warn line' in out and 'error line' in out
    assert 'info line' not in out and 'to file only' not in out
    text = log_path.read_text()
    assert 'to file only' in text and 'info line' in text and 'warn line' in text
    assert text.count('\n') == 4


def test_line_format_has_level_time_source_and_rank(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    ldtrain.warning('formatted')
    line = flushed_text(log_path).splitlines()[0]
    assert line.startswith('WARNING [') and '][test_console:' in line and '][rank:0] formatted' in line


def test_rank_suffix_and_master_only_console(tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
    ldtrain.dist.rank = 1
    console.initialize(output_file=str(tmp_path / 'log.txt'), color=False, only_master_to_console=True)
    ldtrain.info('rank one')
    console.flush_async_streams()
    assert 'rank one' not in capfd.readouterr().out
    assert 'rank one' in (tmp_path / 'log.rank1.txt').read_text()
    assert not (tmp_path / 'log.txt').exists()


def test_to_stdout_and_to_file_routing(tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    ldtrain.info('file only', to_stdout=False)
    ldtrain.info('console only', to_file=False)
    console.flush_async_streams()
    out = capfd.readouterr().out
    assert 'console only' in out and 'file only' not in out
    text = log_path.read_text()
    assert 'file only' in text and 'console only' not in text


def test_third_party_warning_like_errors_are_downgraded(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    logging.getLogger('some.library').error('UserWarning: this API is deprecated')
    logging.getLogger('some.library').error('connection refused')
    ldtrain.error('ours stays an error')
    lines = flushed_text(log_path).splitlines()
    assert lines[0].startswith('WARNING ') and 'deprecated' in lines[0]
    assert lines[1].startswith('ERROR ') and 'connection refused' in lines[1]
    assert lines[2].startswith('ERROR ') and 'ours stays' in lines[2]


def test_traceback_follows_the_message(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    try:
        raise ValueError('boom')
    except ValueError as error:
        ldtrain.error('failed', exc_info=error)
    lines = flushed_text(log_path).splitlines()
    assert lines[0].startswith('ERROR [') and lines[0].endswith('failed')
    assert lines[1] == 'Traceback (most recent call last):'
    assert lines[-1] == 'ValueError: boom'


def test_log_config_dumps_yaml(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    ldtrain.log_config({'model': {'lr': 0.1}}, msg='config')
    ldtrain.log_config(OmegaConf.create({'steps': 3}))
    text = flushed_text(log_path)
    assert 'config\nmodel:\n  lr: 0.1\n' in text and '] steps: 3\n' in text


def test_pause_and_resume_nest(tmp_path: Path) -> None:
    console.initialize(output_file=str(tmp_path / 'log.txt'), color=True)
    handler = logging.getLogger().handlers[0]
    assert handler.formatter._color is True
    console.pause_logging()
    console.pause_logging()
    assert handler.formatter._color is False
    console.resume_logging()
    assert handler.formatter._color is False  # still paused by the outer call
    console.resume_logging()
    assert handler.formatter._color is True
    console.resume_logging()  # extra resume is a no-op
    assert handler.formatter._color is True


def test_reinitialize_never_duplicates_handlers(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    for _ in range(3):
        console.initialize(output_file=str(log_path), color=False)
    ldtrain.info('once')
    assert flushed_text(log_path).count('once') == 1
    assert len(logging.getLogger().handlers) == 2


def test_forked_child_writes_synchronously(tmp_path: Path) -> None:
    log_path = tmp_path / 'log.txt'
    console.initialize(output_file=str(log_path), color=False)
    ldtrain.info('parent before fork')
    pid = os.fork()
    if pid == 0:  # child: the drain thread does not exist here, writes must land before _exit
        ldtrain.info('from child')
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    ldtrain.info('parent after fork')
    text = flushed_text(log_path)
    assert 'from child' in text and 'parent before fork' in text and 'parent after fork' in text
