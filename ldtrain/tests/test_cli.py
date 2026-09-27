"""Tests for the `ldtrain-viewer` command: startup report, Ctrl-C, and the two failure paths."""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ldtrain.tests.conftest import RunTree

STARTUP_TIMEOUT_S = 30


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def run_viewer(*args: str, timeout_s: float = STARTUP_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-m', 'ldtrain.viewer', *args], capture_output=True, text=True, timeout=timeout_s)


def test_missing_directory_fails_with_one_line(tmp_path: Path) -> None:
    result = run_viewer(str(tmp_path / 'nope'))
    assert result.returncode == 1 and result.stderr.strip() == f'error: {tmp_path / "nope"} is not a directory'


def test_busy_port_fails_with_one_line(tmp_path: Path) -> None:
    with socket.socket() as holder:
        holder.bind(('127.0.0.1', 0))
        holder.listen()
        port = holder.getsockname()[1]
        result = run_viewer(str(tmp_path), '--port', str(port))
    assert result.returncode == 1 and result.stderr.strip() == f'error: cannot listen on 127.0.0.1:{port} (Address already in use), pick another port with --port'


def test_reports_runs_serves_and_stops_on_sigint(run_root: RunTree) -> None:
    port = free_port()
    process = subprocess.Popen([sys.executable, '-m', 'ldtrain.viewer', str(run_root.root), '--port', str(port)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail('viewer did not start listening')
        process.send_signal(signal.SIGINT)
        out, _ = process.communicate(timeout=STARTUP_TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 0
    lines = out.splitlines()
    assert lines[0] == 'ldtrain viewer'
    assert lines[1].startswith('  runs   ') and '(3 found' in lines[1]
    assert lines[2] == f'  open   http://localhost:{port}'
    assert lines[-1] == 'ldtrain viewer stopped'
    assert os.linesep == '\n'
