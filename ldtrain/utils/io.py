"""File I/O for images, videos and structured data.

Images and videos are RGB in memory and written with OpenCV, which stores BGR, so
every function converts at the file boundary. Arrays and tensors may be `uint8`, or
floating point in [0, 1], which is clipped and scaled to `uint8`. `save_json` and
`save_yaml` pass their input through `serialize_object`, so tensors, arrays and
dataclasses can be saved directly, and they replace the target atomically.
"""
import json
import os
import tempfile
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ldtrain.utils.image import to_bgr_uint8

#---------------------------------------------------------------------
# image and video
#---------------------------------------------------------------------

def save_image(image: np.ndarray | torch.Tensor, file_path: str | Path) -> None:
    """Write an RGB(A) or grayscale image, creating the parent directory.

    Args:
        image: Image, shape (H, W), (H, W, 1), (H, W, 3) RGB or (H, W, 4) RGBA. `uint8`,
            or floating point in [0, 1].
        file_path: Target file, the format follows its suffix (`.png`, `.jpg`, ...).

    Raises:
        OSError: If OpenCV cannot encode or write the file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        success = cv2.imwrite(str(file_path), to_bgr_uint8(image))
    except cv2.error as err:  # raised for an unknown suffix
        raise OSError(f'Could not write image {file_path}: {err}') from err
    if not success:
        raise OSError(f'Could not write image {file_path}')


def load_image(file_path: str | Path) -> np.ndarray:
    """Read an image as 8-bit RGB.

    Args:
        file_path: Image file readable by OpenCV. Alpha is dropped and grayscale is
            expanded to three channels.

    Returns:
        Image, shape (H, W, 3), `uint8`, RGB.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        OSError: If OpenCV cannot decode the file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'Image {file_path} does not exist')
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)  # (H, W, 3) BGR
    if image is None:
        raise OSError(f'Could not read image {file_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_video(video: np.ndarray | torch.Tensor, file_path: str | Path, fps: float = 30.0, fourcc: str = 'mp4v') -> None:
    """Encode a video, creating the parent directory.

    Args:
        video: Frames, shape (T, H, W, 3) RGB or (T, H, W) grayscale. `uint8`, or
            floating point in [0, 1].
        file_path: Target file, the container follows its suffix (`.mp4`, `.avi`, ...).
        fps: Playback frame rate. Defaults to 30.
        fourcc: Four-character codec code, must be supported by the OpenCV build and the
            container. Defaults to `mp4v`, which every OpenCV build ships.

    Raises:
        ValueError: If `video` has no frames or an unsupported shape.
        OSError: If OpenCV cannot open a writer for the codec and container.
    """
    frames = to_bgr_uint8(video)  # (T, H, W, 3) or (T, H, W)
    if frames.ndim not in (3, 4) or frames.shape[0] == 0:
        raise ValueError(f'Expected video of shape (T, H, W[, 3]) with T > 0, got {tuple(frames.shape)}')

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(file_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height),
                             isColor=frames.ndim == 4)
    if not writer.isOpened():
        raise OSError(f'Could not open video writer for {file_path} with codec {fourcc}')
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def load_video(file_path: str | Path) -> np.ndarray:
    """Decode every frame of a video as 8-bit RGB.

    Args:
        file_path: Video file readable by OpenCV.

    Returns:
        Frames, shape (T, H, W, 3), `uint8`, RGB.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        OSError: If OpenCV cannot open the file or decodes no frame.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'Video {file_path} does not exist')
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        raise OSError(f'Could not open video {file_path}')

    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()  # (H, W, 3) BGR
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise OSError(f'Decoded no frame from video {file_path}')
    return np.stack(frames)  # (T, H, W, 3)




#---------------------------------------------------------------------
# structured data
#---------------------------------------------------------------------

def save_json(obj: object, file_path: str | Path, indent: int = 4) -> None:
    """Serialize `obj` with `serialize_object` and write it as JSON.

    Args:
        obj: Any object `serialize_object` reduces to lists, dicts and scalars.
        file_path: Target file, parent directories are created.
        indent: Spaces per nesting level. Defaults to 4.

    Raises:
        TypeError: If a value remains that JSON cannot encode.
    """
    text = json.dumps(serialize_object(obj), indent=indent)
    write_text_atomic(text, Path(file_path))


def save_yaml(obj: object, file_path: str | Path) -> None:
    """Serialize `obj` with `serialize_object` and write it as YAML, keeping key order.

    Args:
        obj: Any object `serialize_object` reduces to lists, dicts and scalars.
        file_path: Target file, parent directories are created.

    Raises:
        yaml.representer.RepresenterError: If a value remains that plain YAML cannot
            represent.
    """
    text = yaml.safe_dump(serialize_object(obj), sort_keys=False)
    write_text_atomic(text, Path(file_path))


def write_text_atomic(text: str, file_path: Path) -> None:
    """Write `text` to a temporary sibling and rename it over `file_path`.

    A crash mid-write leaves the previous file intact rather than a truncated one.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=file_path.parent, prefix=f'.{file_path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_name, file_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def serialize_object(obj: object) -> object:
    """Reduce `obj` recursively to lists, dicts and Python scalars.

    Tensors and arrays become nested lists, NumPy scalars Python scalars, tuples lists,
    paths strings, enums their value, and other objects the dict of their attributes.
    Everything else is returned unchanged.
    """
    if isinstance(obj, list | tuple):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return serialize_object(obj.value)
    elif hasattr(obj, '__dict__'):
        return {key: serialize_object(value) for key, value in obj.__dict__.items()}
    else:
        return obj


def deserialize_object(obj: object) -> object:
    if isinstance(obj, list):
        return [deserialize_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deserialize_object(value) for key, value in obj.items()}
    else:
        return obj
