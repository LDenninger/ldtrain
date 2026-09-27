"""Shallow logging for deep-learning runs into human-readable files, with a browser viewer.

`initialize()` turns a run directory into a log file, a metrics CSV and a visuals folder, and
`finish()` closes them, and `catch_exceptions()` guards a block so a failure is logged before the
run stops or continues. Console logging goes through `info`, `warning`, `error`, `critical`,
`debug`, `dev` and `log_config`, and run data through `log_metrics`, `log_images` and
`log_videos`. `run` holds the run's directories, `dist` the process rank and world size, and
`barrier()` synchronizes ranks. The viewer lives in `ldtrain.viewer` and is started with
`python -m ldtrain.viewer <root_dir>`.
"""
from importlib.metadata import PackageNotFoundError, version

from ldtrain.config import Config
from ldtrain.distributed import DistributedConfig, barrier, dist, distributed_barrier
from ldtrain.exceptions import DistributedError, DistributedInitializationFailure, DistributedMisconfigured
from ldtrain.lifecycle import catch_exceptions, finish, initialize
from ldtrain.log.api import log_config, log_images, log_metrics, log_videos
from ldtrain.log.api import log_critical as critical
from ldtrain.log.api import log_debug as debug
from ldtrain.log.api import log_dev as dev
from ldtrain.log.api import log_error as error
from ldtrain.log.api import log_info as info
from ldtrain.log.api import log_warning as warning
from ldtrain.run_config import RunConfig, run

try:
    __version__ = version('ldtrain')
except PackageNotFoundError:  # running from a checkout that is not installed
    __version__ = '0.0.0'

__all__ = [
    'initialize',
    'finish',
    'catch_exceptions',
    'info',
    'warning',
    'error',
    'critical',
    'debug',
    'dev',
    'log_config',
    'log_metrics',
    'log_images',
    'log_videos',
    'run',
    'dist',
    'barrier',
    'distributed_barrier',
    'Config',
    'RunConfig',
    'DistributedConfig',
    'DistributedError',
    'DistributedMisconfigured',
    'DistributedInitializationFailure',
    '__version__',
]
