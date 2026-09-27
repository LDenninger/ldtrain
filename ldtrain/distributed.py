from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from contextlib import contextmanager

import torch

from ldtrain.config import Config
from ldtrain.exceptions import DistributedMisconfigured, DistributedInitializationFailure

logger = logging.getLogger(__name__)

#---------------------------------------------------------------------
# configuration
#---------------------------------------------------------------------

@dataclass
class DistributedConfig(Config):
    master_addr: str = 'localhost'
    master_port: int = 29500
    world_size: int = 1
    local_rank: int = 0
    dist_url: str = 'env://'
    rank: int = 0
    backend: str = ''  # 'nccl' or 'gloo' once initialized

    def __post_init__(self) -> None:
        self._initialized: bool = False

    def reset(self) -> None:
        super().reset()
        self._initialized = False

    def update_from_environment(self) -> None:
        if os.environ.get('MASTER_ADDR'):
            self.master_addr = os.environ['MASTER_ADDR']
        if os.environ.get('MASTER_PORT'):
            self.master_port = int(os.environ['MASTER_PORT'])
        if os.environ.get('WORLD_SIZE'):
            self.world_size = int(os.environ['WORLD_SIZE'])
        if os.environ.get('LOCAL_RANK'):
            self.local_rank = int(os.environ['LOCAL_RANK'])
        if os.environ.get('RANK'):
            self.rank = int(os.environ['RANK'])
        if os.environ.get('DIST_URL'):
            self.dist_url = os.environ['DIST_URL']

    @classmethod
    def from_environment(cls) -> DistributedConfig:
        return cls(
            master_addr=os.environ.get('MASTER_ADDR', 'localhost'),
            master_port=int(os.environ['MASTER_PORT']) if os.environ.get('MASTER_PORT') else 29500,
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dist_url=os.environ.get('DIST_URL', 'env://'),
            rank=int(os.environ.get('RANK', 0)),
        )

    @property
    def configured(self) -> bool:
        return (self.master_addr is not None and self.master_port is not None and self.world_size is not None
                and self.local_rank is not None and self.dist_url is not None and self.rank is not None)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_master(self) -> bool:
        return self.rank == 0

dist = DistributedConfig()

#---------------------------------------------------------------------
# initialization
#---------------------------------------------------------------------

def initialize(
    world_size: int = dist.world_size,
    local_rank: int = dist.local_rank,
    global_rank: int = dist.rank,
    dist_url: str = dist.dist_url,
) -> bool:
    """Initialize the process group: NCCL on a CUDA machine, gloo on a CPU-only one.

    Args:
        world_size: Number of processes in the group.
        local_rank: Index of this process on its machine, also its CUDA device.
        global_rank: Index of this process in the group.
        dist_url: Rendezvous, `env://` reads `MASTER_ADDR` and `MASTER_PORT`.

    Returns:
        True once the group is up.

    Raises:
        DistributedMisconfigured: `dist_url` is `env://` and `MASTER_ADDR` or `MASTER_PORT`
            is not set.
        DistributedInitializationFailure: The CUDA device or the process group cannot be
            set up.
    """
    dist.update(
        world_size=world_size,
        local_rank=local_rank,
        rank=global_rank,
        dist_url=dist_url,
    )
    if dist_url == 'env://' and not (os.environ.get('MASTER_ADDR') and os.environ.get('MASTER_PORT')):
        raise DistributedMisconfigured('dist_url is env:// but MASTER_ADDR or MASTER_PORT is not set')

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    device_id = None
    if backend == 'nccl':
        try:
            torch.cuda.set_device(local_rank)
        except Exception as e:
            raise DistributedInitializationFailure(f"Failed to initialize cuda device '{local_rank}'\n{e}")
        device_id = torch.device('cuda', local_rank)

    logger.debug(f'Initializing {backend} process group: rank {global_rank}/{world_size}, local_rank {local_rank}, '
                 f'dist_url {dist_url}')
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            device_id=device_id,
        )
        torch.distributed.barrier()
    except Exception as e:
        raise DistributedInitializationFailure(f"Failed to initialize distributed process group\n{e}")

    dist._initialized = True
    dist.backend = backend
    logger.info(f'Initialized {backend} process group: rank {global_rank}/{world_size}, local_rank {local_rank}')
    return True


def finish() -> None:
    """Destroy the process group opened by `initialize()`, a no-op when there is none."""
    if dist.initialized:
        torch.distributed.destroy_process_group()
        dist._initialized = False
        dist.backend = ''


#---------------------------------------------------------------------
# synchronization
#---------------------------------------------------------------------

def barrier(only_master: bool = True, only_non_master: bool = False) -> None:
    if dist.initialized:
        if only_master and dist.is_master:
            torch.distributed.barrier()
        elif only_non_master and not dist.is_master:
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    return

@contextmanager
def distributed_barrier(only_master: bool = True):
    """
    Context manager for distributed barrier.
    If only_master is True:
        - The barrier will only be executed for non-master ranks (i.e., non-masters wait for the master).
        - The master skips the barrier, allowing others to wait for it.
    If only_master is False:
        - All processes execute the barrier normally.
    """
    try:
        if dist.initialized:
            if only_master:
                if not dist.is_master:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()
        yield
    finally:
        pass