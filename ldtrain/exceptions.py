import logging

logger = logging.getLogger(__name__)


class DistributedError(Exception):
    """Base class for distributed errors, logged once as CRITICAL on construction."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
        logger.critical(str(self), stacklevel=2)

    def __str__(self) -> str:
        return f'{type(self).__name__}: {self.message}\n{self.config_str()}'

    def config_str(self) -> str:
        from ldtrain.distributed import dist
        return (f' - master_addr: {dist.master_addr}\n'
                f' - master_port: {dist.master_port}\n'
                f' - world_size: {dist.world_size}\n'
                f' - local_rank: {dist.local_rank}\n'
                f' - dist_url: {dist.dist_url}\n'
                f' - rank: {dist.rank}\n'
                f' - is_master: {dist.is_master}\n'
                f' - is_initialized: {dist.initialized}\n')


class DistributedMisconfigured(DistributedError):
    """Error raised when distributed configuration is misconfigured."""


class DistributedInitializationFailure(DistributedError):
    """Error raised when distributed initialization fails due to an exception."""
