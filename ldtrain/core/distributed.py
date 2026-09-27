from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from contextlib import contextmanager

import torch

from ldtrain.core.config import Config
from ldtrain.core.exceptions import DistributedMisconfigured, DistributedInitializationFailure

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

distributed = DistributedConfig()

#---------------------------------------------------------------------
# initialization
#---------------------------------------------------------------------

def initialize(
    world_size: int = distributed.world_size,
    local_rank: int = distributed.local_rank,
    global_rank: int = distributed.rank,
    dist_url: str = distributed.dist_url,
) -> bool:
    """Initialize distributed training mode."""

    distributed.update(
        world_size=world_size,
        local_rank=local_rank,
        rank=global_rank,
        dist_url=dist_url,
    )

    if not torch.cuda.is_available():
        raise DistributedMisconfigured("CUDA is not available. Please check your CUDA installation and setup.")

    try:
        torch.cuda.set_device(local_rank)
    except Exception as e:
        raise DistributedInitializationFailure(f"Failed to initialize cuda device '{local_rank}'\n{e}")

    logger.debug(f'Initializing process group: rank {global_rank}/{world_size}, local_rank {local_rank}, '
                 f'dist_url {dist_url}')
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            device_id=torch.device("cuda", local_rank),
        )
        torch.distributed.barrier()
    except Exception as e:
        raise DistributedInitializationFailure(f"Failed to initialize distributed process group\n{e}")

    distributed._initialized = True
    logger.info(f'Initialized distributed process group: rank {global_rank}/{world_size}, local_rank {local_rank}')
    return True


#---------------------------------------------------------------------
# synchronization
#---------------------------------------------------------------------

def barrier(only_master: bool = True, only_non_master: bool = False) -> None:
    if distributed.initialized:
        if only_master and distributed.is_master:
            torch.distributed.barrier()
        elif only_non_master and not distributed.is_master:
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
        if distributed.initialized:
            if only_master:
                if not distributed.is_master:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()
        yield
    finally:
        pass