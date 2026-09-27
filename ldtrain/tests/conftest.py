"""Shared fixtures: isolated logging state per test, and a run tree written through the public API."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import ldtrain
from ldtrain.log import console


@dataclass
class RunTree:
    """A root of runs written by `run_root`, with the number of iterations per run path."""
    root: Path
    steps: dict[str, int]


@pytest.fixture(autouse=True)
def isolated_logging():
    """Start every test with console-only logging, rank 0 and no tracker, and leave it so."""
    ldtrain.dist.reset()
    ldtrain.initialize(color=False)
    yield
    ldtrain.finish()
    ldtrain.dist.reset()
    ldtrain.initialize(color=False)


def write_run(run_dir: Path, steps: int, with_media: bool, with_config: str | None, seed: int) -> None:
    """Write one run through the public API: metrics every step, val every 10, media at 0 and 30."""
    rng = np.random.default_rng(seed)
    ldtrain.initialize(run_dir=run_dir, color=False, log_level='warning')
    if with_config:
        (run_dir / 'configs').mkdir(exist_ok=True)
        (run_dir / 'configs' / f'config.{with_config}').write_text(
            'model:\n  name: mlp\n  lr: 0.001\n' if with_config == 'yaml' else '{"model": {"name": "mlp"}}\n')
    ldtrain.info(f'starting {run_dir.name}')
    for step in range(steps):
        metrics = {'train/loss': float(np.exp(-step / 20) + 0.01 * rng.standard_normal()), 'train/lr': 1e-3}
        if step % 10 == 0:
            metrics['val/psnr'] = float(20 + step / 4)
        ldtrain.log_metrics(metrics, step=step)
        if with_media and step in (0, 30):
            ldtrain.log_images({'val/pred': rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), 'depth': rng.random((16, 16), dtype=np.float32)}, step=step)
        if with_media and step == 30:
            ldtrain.log_videos({'val/rollout': rng.integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)}, step=step, fps=5)
        if step == 5:
            ldtrain.warning('learning rate warm-up finished')
        if step == 7:
            ldtrain.error('one bad batch skipped')
    ldtrain.info('finished')
    ldtrain.finish()


@pytest.fixture(scope='session')
def run_root(tmp_path_factory: pytest.TempPathFactory) -> RunTree:
    """Three runs in two projects: one with media and a YAML config, one with a rank-1 log, one tiny."""
    root = tmp_path_factory.mktemp('runs')
    steps = {'proj_a/run_1': 40, 'proj_a/run_2': 20, 'proj_b/run_3': 5}
    write_run(root / 'proj_a/run_1', 40, with_media=True, with_config='yaml', seed=0)
    write_run(root / 'proj_a/run_2', 20, with_media=False, with_config=None, seed=1)
    ldtrain.dist.rank = 1  # a second rank appends its own log file to run_2
    ldtrain.initialize(run_dir=root / 'proj_a/run_2', color=False, log_level='warning')
    ldtrain.warning('hello from rank 1')
    ldtrain.finish()
    ldtrain.dist.reset()
    write_run(root / 'proj_b/run_3', 5, with_media=False, with_config='json', seed=2)
    ldtrain.initialize(color=False)
    return RunTree(root=root, steps=steps)


def flushed_text(path: Path) -> str:
    """Read a log file after every async stream has drained."""
    console.flush_async_streams()
    return path.read_text()
