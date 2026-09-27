"""Tests for `ldtrain.utils.files` and `ldtrain.utils.image`: media round trips and atomic writes."""
import enum
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ldtrain.utils import files
from ldtrain.utils.image import to_bgr_uint8


def test_to_bgr_uint8_layouts() -> None:
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert to_bgr_uint8(rgb).tolist() == [[[3, 2, 1]]]
    rgba = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    assert to_bgr_uint8(rgba).tolist() == [[[3, 2, 1, 4]]]
    assert to_bgr_uint8(np.ones((2, 2, 1), np.float32)).tolist() == [[255, 255], [255, 255]]
    assert to_bgr_uint8(torch.full((1, 1, 3), 2.0)).tolist() == [[[255, 255, 255]]]  # clipped to 1.0
    with pytest.raises(TypeError):
        to_bgr_uint8(np.zeros((2, 2), np.int32))


def test_image_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (8, 6, 3), dtype=np.uint8)
    files.save_image(image, tmp_path / 'nested' / 'a.png')
    assert np.array_equal(files.load_image(tmp_path / 'nested' / 'a.png'), image)
    files.save_image(np.full((4, 4), 0.5, np.float32), tmp_path / 'gray.png')
    gray = files.load_image(tmp_path / 'gray.png')
    assert gray.shape == (4, 4, 3) and gray[0, 0].tolist() == [128, 128, 128]
    files.save_image(np.concatenate([image, np.full((8, 6, 1), 255, np.uint8)], axis=-1), tmp_path / 'rgba.png')
    assert np.array_equal(files.load_image(tmp_path / 'rgba.png'), image)  # alpha dropped on load


def test_image_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        files.save_image(np.zeros((2, 2, 3), np.uint8), tmp_path / 'a.nosuchformat')
    with pytest.raises(FileNotFoundError):
        files.load_image(tmp_path / 'missing.png')
    (tmp_path / 'junk.png').write_bytes(b'not an image')
    with pytest.raises(OSError):
        files.load_image(tmp_path / 'junk.png')


def test_video_round_trip(tmp_path: Path) -> None:
    video = np.zeros((4, 16, 16, 3), np.uint8)
    video[:, :, :, 0] = 200  # red frames survive lossy encoding recognisably
    files.save_video(video, tmp_path / 'v.webm', fps=5, fourcc='VP80')
    loaded = files.load_video(tmp_path / 'v.webm')
    assert loaded.shape == (4, 16, 16, 3) and loaded[..., 0].mean() > 150 and loaded[..., 1].mean() < 50
    files.save_video(torch.rand(2, 8, 8), tmp_path / 'gray.avi', fourcc='MJPG')
    assert files.load_video(tmp_path / 'gray.avi').shape == (2, 8, 8, 3)


def test_video_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='shape'):
        files.save_video(np.zeros((0, 8, 8, 3), np.uint8), tmp_path / 'empty.mp4')
    with pytest.raises(ValueError, match='shape'):
        files.save_video(np.zeros((8, 8), np.uint8), tmp_path / 'flat.mp4')
    with pytest.raises(FileNotFoundError):
        files.load_video(tmp_path / 'missing.mp4')


class Color(enum.Enum):
    RED = 'red'


class Plain:
    def __init__(self) -> None:
        self.value = np.float32(1.5)
        self.path = Path('/tmp/x')


def test_serialize_object() -> None:
    result = files.serialize_object({'t': torch.tensor([1, 2]), 'a': np.arange(2), 'p': Path('a/b'), 'c': Color.RED, 'o': Plain(), 'tup': (1, 2), 'n': None})
    assert result == {'t': [1, 2], 'a': [0, 1], 'p': 'a/b', 'c': 'red', 'o': {'value': 1.5, 'path': '/tmp/x'}, 'tup': [1, 2], 'n': None}
    assert files.deserialize_object(result) == result


def test_json_and_yaml_writers(tmp_path: Path) -> None:
    files.save_json({'b': np.int64(2), 'a': [Path('p')]}, tmp_path / 'd' / 'x.json')
    assert (tmp_path / 'd' / 'x.json').read_text() == '{\n    "b": 2,\n    "a": [\n        "p"\n    ]\n}'
    files.save_yaml({'b': 2, 'a': 1}, tmp_path / 'x.yaml')
    assert (tmp_path / 'x.yaml').read_text() == 'b: 2\na: 1\n'  # insertion order kept
    assert yaml.safe_load((tmp_path / 'x.yaml').read_text()) == {'b': 2, 'a': 1}


def test_write_text_atomic_keeps_previous_on_failure(tmp_path: Path) -> None:
    target = tmp_path / 'out.txt'
    files.write_text_atomic('first', target)
    with pytest.raises(TypeError):
        files.write_text_atomic(123, target)  # type: ignore[arg-type]
    assert target.read_text() == 'first'
    assert [p.name for p in tmp_path.iterdir()] == ['out.txt']  # no temp file left behind
