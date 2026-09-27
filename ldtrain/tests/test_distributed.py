"""Tests for `ldtrain.distributed`: environment parsing, the no-op barrier and a real two-process group."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import ldtrain
from ldtrain import distributed
from ldtrain.distributed import DistributedConfig
from ldtrain.exceptions import DistributedMisconfigured

WORKER = Path(__file__).parent / 'distributed_worker.py'
TIMEOUT_S = 120


def test_environment_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK', 'DIST_URL'):
        monkeypatch.delenv(name, raising=False)
    assert DistributedConfig.from_environment() == DistributedConfig()
    monkeypatch.setenv('MASTER_ADDR', '10.0.0.1')
    monkeypatch.setenv('MASTER_PORT', '12345')
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setenv('RANK', '3')
    expected = DistributedConfig(master_addr='10.0.0.1', master_port=12345, world_size=4, local_rank=2, rank=3)
    assert DistributedConfig.from_environment() == expected
    config = DistributedConfig()
    config.update_from_environment()
    assert config == expected and not config.is_master and not config.initialized


def test_barriers_are_no_ops_without_a_group() -> None:
    assert not ldtrain.dist.initialized
    ldtrain.barrier()
    ldtrain.barrier(only_master=False)
    with ldtrain.distributed_barrier():
        pass
    distributed.finish()  # nothing to destroy


def test_initialize_needs_a_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MASTER_ADDR', raising=False)
    monkeypatch.delenv('MASTER_PORT', raising=False)
    with pytest.raises(DistributedMisconfigured):
        distributed.initialize(world_size=2, local_rank=0, global_rank=0)
    assert not ldtrain.dist.initialized


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def launch_ranks(run_dir: Path, world_size: int, fail_rank: int = -1) -> list[subprocess.CompletedProcess]:
    """Start `world_size` workers as torchrun would, and wait for all of them."""
    port = free_port()
    processes = []
    for rank in range(world_size):
        env = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port), 'WORLD_SIZE': str(world_size),
               'RANK': str(rank), 'LOCAL_RANK': str(rank), 'CUDA_VISIBLE_DEVICES': ''}
        processes.append(subprocess.Popen([sys.executable, str(WORKER), str(run_dir), '--fail_rank', str(fail_rank)],
                                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    results = []
    for process in processes:
        try:
            out, _ = process.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            out, _ = process.communicate()
            pytest.fail(f'rank hung for {TIMEOUT_S} s:\n{out}')
        results.append(subprocess.CompletedProcess(process.args, process.returncode, out))
    return results


@pytest.mark.multiprocess
def test_two_ranks_share_a_gloo_group_and_write_their_files(tmp_path: Path) -> None:
    results = launch_ranks(tmp_path / 'run', world_size=2)
    assert [result.returncode for result in results] == [0, 0], [result.stdout for result in results]
    logs_dir = tmp_path / 'run' / 'logs'
    rank0 = (logs_dir / 'log.txt').read_text()
    rank1 = (logs_dir / 'log.rank1.txt').read_text()
    assert 'rank 0 of 2 on gloo' in rank0 and 'past the barrier' in rank0
    assert 'rank 1 of 2 on gloo' in rank1 and 'past the barrier' in rank1
    assert (tmp_path / 'run' / 'metrics' / 'metrics.csv').read_text() == 'iteration,loss\n0,0.0\n'
    assert (tmp_path / 'run' / 'visuals' / '00000000' / 'x.png').exists()
    assert 'rank 1 of 2' not in results[0].stdout  # only the master prints to the console


@pytest.mark.multiprocess
def test_failing_rank_exits_one_and_the_other_does_not_hang(tmp_path: Path) -> None:
    results = launch_ranks(tmp_path / 'run', world_size=2, fail_rank=1)
    assert results[1].returncode == 1, results[1].stdout
    assert 'RuntimeError: simulated failure' in (tmp_path / 'run' / 'logs' / 'log.rank1.txt').read_text()
    assert results[0].returncode != 0  # its barrier partner is gone, so rank 0 fails instead of waiting forever
