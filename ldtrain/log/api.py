from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from ldtrain.distributed import dist
from ldtrain.log import metric_tracker
from ldtrain.log.console import LEVEL_STYLES, LogLevel, LogLevelName, get_logger, is_initialized
from ldtrain.run_config import run
from ldtrain.utils import files

#---------------------------------------------------------------------
# text logging
#---------------------------------------------------------------------

def log_info(msg: str, *args: Any, to_stdout: bool = True, to_file: bool = True, stacklevel: int = 4,
             **kwargs: Any) -> None:
    """ Log info message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.INFO].format(msg))
    else:
        get_logger().info(msg, *args, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel, **kwargs)


def log_warning(msg: str, *args: Any, to_stdout: bool = True, to_file: bool = True, stacklevel: int = 4,
                **kwargs: Any) -> None:
    """ Log warning message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.WARNING].format(msg))
    else:
        get_logger().warning(msg, *args, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel, **kwargs)


def log_error(msg: str, *args: Any, to_stdout: bool = True, to_file: bool = True, stacklevel: int = 4,
              **kwargs: Any) -> None:
    """ Log error message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.ERROR].format(msg))
    else:
        get_logger().error(msg, *args, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel, **kwargs)


def log_critical(msg: str, *args: Any, to_stdout: bool = True, to_file: bool = True, stacklevel: int = 4,
                 **kwargs: Any) -> None:
    """ Log critical message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.CRITICAL].format(msg))
    else:
        get_logger().critical(msg, *args, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel, **kwargs)


def log_debug(msg: str, *args: Any, color: str | None = None, to_stdout: bool = True, to_file: bool = True,
              stacklevel: int = 4, **kwargs: Any) -> None:
    """ Log debug message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.DEBUG].format(msg))
    else:
        get_logger().debug(msg, *args, color=color, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel,
                           **kwargs)


def log_dev(msg: str, *args: Any, to_stdout: bool = True, to_file: bool = True, stacklevel: int = 4,
            **kwargs: Any) -> None:
    """ Log dev message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.DEV].format(msg))
    else:
        get_logger().dev(msg, *args, to_stdout=to_stdout, to_file=to_file, stacklevel=stacklevel, **kwargs)


def log_config(config: dict | list | DictConfig | ListConfig, msg: str | None = None, to_stdout: bool = False,
               to_file: bool = True, log_level: LogLevelName = 'info', stacklevel: int = 4) -> None:
    """ Log configuration message to the console and/or file. """
    if not is_initialized():
        print(LEVEL_STYLES[LogLevel.INFO].format(msg or ''))
    else:
        get_logger().log_config(config, msg, to_stdout=to_stdout, to_file=to_file, log_level=log_level,
                                stacklevel=stacklevel - 1)


#---------------------------------------------------------------------
# data logging
#---------------------------------------------------------------------

def log_metrics(metrics: dict[str, float | int | torch.Tensor | np.ndarray], step: int) -> None:
    """Record scalars at iteration `step`.

    Args:
        metrics: Scalar per metric name, the name becoming the CSV column. Tensors and
            arrays must hold one element. A CUDA tensor synchronizes the device.
        step: Training iteration the values belong to.
    """
    tracker = metric_tracker.tracker
    if tracker is None:
        return
    for name, value in metrics.items():
        tracker.track(name, float(value), step)


def log_images(images: dict[str, np.ndarray | torch.Tensor], step: int) -> None:
    """Write each image to `visuals/<step:08d>/<tag>.png`.

    Args:
        images: Image per tag, shape (H, W), (H, W, 1), (H, W, 3) RGB or (H, W, 4) RGBA,
            `uint8` or floating point in [0, 1].
        step: Training iteration, naming the folder.
    """
    if not run.root_directory or not dist.is_master:
        return
    step_dir = Path(run.visual_dir) / f'{step:08d}'
    for tag, image in images.items():
        files.save_image(image, step_dir / f'{tag}.png')


def log_videos(videos: dict[str, np.ndarray | torch.Tensor], step: int, fps: float = 30.0) -> None:
    """Write each video to `visuals/<step:08d>/<tag>.webm`, encoded as VP8.

    VP8 WebM plays in every current browser, so the viewer can show it. OpenCV's `mp4v`
    (MPEG-4 Part 2) does not, and its H.264 writer is missing from many builds. VP8 is
    chosen over VP9 for speed: encoding blocks the caller, and VP9 took 2.0 s against
    0.29 s for VP8 on 30 noise frames of 256 × 256.

    Args:
        videos: Frames per tag, shape (T, H, W, 3) RGB or (T, H, W) grayscale, `uint8` or
            floating point in [0, 1].
        step: Training iteration, naming the folder.
        fps: Playback frame rate. Defaults to 30.
    """
    if not run.root_directory or not dist.is_master:
        return
    step_dir = Path(run.visual_dir) / f'{step:08d}'
    for tag, video in videos.items():
        files.save_video(video, step_dir / f'{tag}.webm', fps=fps, fourcc='VP80')
