"""Tests for the data logging API: the run layout that `log_metrics`, `log_images` and `log_videos` write."""
from pathlib import Path

import numpy as np
import torch

import ldtrain
from ldtrain.utils import files


def test_metrics_accept_scalars_tensors_and_arrays(tmp_path: Path) -> None:
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    ldtrain.log_metrics({'loss': 1, 'psnr': torch.tensor(2.5), 'lr': np.float32(0.25)}, step=0)
    ldtrain.log_metrics({'loss': torch.tensor([0.5])}, step=1)
    ldtrain.finish()
    assert (tmp_path / 'run' / 'metrics' / 'metrics.csv').read_text() == 'iteration,loss,psnr,lr\n0,1.0,2.5,0.25\n1,0.5,,\n'


def test_images_and_videos_land_in_step_folders(tmp_path: Path) -> None:
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    image = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    ldtrain.log_images({'val/pred': image, 'depth': torch.full((8, 8), 0.5)}, step=3)
    ldtrain.log_videos({'val/rollout': np.zeros((4, 16, 16, 3), np.uint8)}, step=3, fps=5)
    step_dir = tmp_path / 'run' / 'visuals' / '00000003'
    assert sorted(p.relative_to(step_dir).as_posix() for p in step_dir.rglob('*') if p.is_file()) == ['depth.png', 'val/pred.png', 'val/rollout.webm']
    assert np.array_equal(files.load_image(step_dir / 'val' / 'pred.png'), image)
    assert files.load_image(step_dir / 'depth.png')[0, 0].tolist() == [128, 128, 128]
    assert files.load_video(step_dir / 'val' / 'rollout.webm').shape == (4, 16, 16, 3)


def test_nothing_is_written_without_a_run_or_on_other_ranks(tmp_path: Path) -> None:
    ldtrain.log_metrics({'loss': 1.0}, step=0)  # console-only logging: no tracker
    ldtrain.log_images({'x': np.zeros((2, 2, 3), np.uint8)}, step=0)
    ldtrain.dist.rank = 1
    ldtrain.initialize(run_dir=tmp_path / 'run', color=False)
    ldtrain.log_metrics({'loss': 1.0}, step=0)
    ldtrain.log_images({'x': np.zeros((2, 2, 3), np.uint8)}, step=0)
    ldtrain.log_videos({'v': np.zeros((2, 2, 2, 3), np.uint8)}, step=0)
    ldtrain.finish()
    assert not (tmp_path / 'run' / 'metrics').exists() and not (tmp_path / 'run' / 'visuals').exists()
    assert (tmp_path / 'run' / 'logs' / 'log.rank1.txt').exists()
