import os
from dataclasses import dataclass

from ldtrain.config import Config


@dataclass
class RunConfig(Config):
    """Run directory and the names of its per-modality subdirectories."""

    PREFERRED_LOG_DIR: str = 'logs'
    PREFERRED_VISUAL_DIR: str = 'visuals'
    PREFERRED_CHECKPOINT_DIR: str = 'checkpoints'
    PREFERRED_CONFIG_DIR: str = 'configs'
    PREFERRED_METRICS_DIR: str = 'metrics'

    #--- run description ---
    run_name: str = 'not_defined'
    project_name: str = 'not_defined'

    #--- paths ---
    root_directory: str = ''

    #---------------------------------------------------------------------
    # properties
    #---------------------------------------------------------------------

    @property
    def log_dir(self) -> str:
        return os.path.join(self.root_directory, self.PREFERRED_LOG_DIR)

    @property
    def visual_dir(self) -> str:
        return os.path.join(self.root_directory, self.PREFERRED_VISUAL_DIR)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.root_directory, self.PREFERRED_CHECKPOINT_DIR)

    @property
    def config_dir(self) -> str:
        return os.path.join(self.root_directory, self.PREFERRED_CONFIG_DIR)

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.root_directory, self.PREFERRED_METRICS_DIR)


run = RunConfig()
