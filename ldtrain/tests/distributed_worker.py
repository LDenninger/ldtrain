"""Worker for the multi-process tests: one training rank driven by the torchrun environment."""
import argparse
from pathlib import Path

import numpy as np

import ldtrain


def distributed_worker(run_dir: str, fail_rank: int) -> None:
    with ldtrain.catch_exceptions():
        ldtrain.initialize(run_dir=Path(run_dir), use_distributed=True, color=False)
        ldtrain.info(f'rank {ldtrain.dist.rank} of {ldtrain.dist.world_size} on {ldtrain.dist.backend}')
        ldtrain.barrier(only_master=False)
        ldtrain.log_metrics({'loss': float(ldtrain.dist.rank)}, step=0)
        ldtrain.log_images({'x': np.zeros((2, 2, 3), np.uint8)}, step=0)
        if ldtrain.dist.rank == fail_rank:
            raise RuntimeError('simulated failure')
        with ldtrain.distributed_barrier(only_master=False):
            ldtrain.info('past the barrier')
        ldtrain.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--fail_rank', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    distributed_worker(**vars(args))
