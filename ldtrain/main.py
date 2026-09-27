"""Framework entry point: `initialize()` sets up logging and the distributed backend."""
import os
from pathlib import Path

from ldtrain.core import distributed, run_config
from ldtrain.core.distributed import initialize as initialize_distributed
from ldtrain.logging import api
from ldtrain.logging import console
from ldtrain.logging import metric_tracker

#---------------------------------------------------------------------
# life cycle
#---------------------------------------------------------------------

def initialize(
    run_dir: str | Path | None = None,
    use_distributed: bool | None = None,
    log_level: console.LogLevelName = 'info',
    log_level_file: console.LogLevelName = 'info',
    only_master_to_console: bool = True,
    color: bool = True,
) -> None:
    """Initialize logging and, if requested, the distributed backend.

    Safe to call again: logging is torn down and rebuilt with the new arguments, and an
    existing process group is kept, since rebuilding it would need every rank at once.
    The rank is read from the environment first, so logging already knows it: rank
    `r > 0` writes to `log.rank<r>.txt`. Logging starts before the process group, so a
    failing NCCL setup is recorded in the log file too.

    Args:
        run_dir: Run directory. The text log goes to `<run_dir>/logs/log.txt`, scalars to
            `<run_dir>/metrics/metrics.csv`, images and videos to
            `<run_dir>/visuals/<step>/`. An existing directory is reused with a warning,
            and its files are appended to. None logs to the console only.
        use_distributed: Initialize an NCCL process group from the torchrun environment
            (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`). None
            enables it when `WORLD_SIZE` > 1.
        log_level: Minimum level written to the console.
        log_level_file: Minimum level written to the log file.
        only_master_to_console: Silence the console on every rank but 0. Defaults to
            True.
        color: Emit ANSI colors on the console.

    Raises:
        DistributedMisconfigured: If distributed mode is requested without CUDA.
        DistributedInitializationFailure: If the CUDA device or process group cannot be
            set up.
    """
    if use_distributed is None:
        use_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if use_distributed and not distributed.initialized:
        distributed.update_from_environment()

    run_config.update(root_directory=str(run_dir) if run_dir is not None else '')
    run_dir_existed = run_dir is not None and Path(run_dir).exists()
    console.initialize(
        output_file=str(Path(run_config.log_dir) / 'log.txt') if run_dir is not None else None,
        only_master_to_console=only_master_to_console,
        color=color,
        log_level=log_level,
        log_level_file=log_level_file,
    )
    if run_dir_existed:
        api.log_warning(f'Run directory {run_dir} already exists, continuing in it')

    if use_distributed and not distributed.initialized:
        initialize_distributed(
            world_size=distributed.world_size,
            local_rank=distributed.local_rank,
            global_rank=distributed.rank,
            dist_url=distributed.dist_url,
        )

    metric_tracker.initialize()
    api.log_info('ldtrain initialized', color='light_green')


def main():
    pass

def parse_args():
    pass

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
