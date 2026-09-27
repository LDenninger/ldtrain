"""Tests for `ldtrain.config.Config` loaders, savers and updates on a small subclass."""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ldtrain.config import Config


@dataclass
class Sample(Config):
    name: str = 'mlp'
    lr: float = 1e-3
    steps: int = 10
    flag: bool = False
    tags: list = field(default_factory=list)


def test_yaml_and_json_round_trip(tmp_path: Path) -> None:
    sample = Sample(name='siren', lr=5e-4, steps=3, flag=True, tags=['a', 'b'])
    for suffix in ('yaml', 'json'):
        path = tmp_path / f'config.{suffix}'
        sample.save(path)
        assert Sample.load_from_file(path) == sample
    assert 'name: siren' in sample.to_yaml() and '"steps": 3' in sample.to_json()


def test_unknown_suffix_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Unsupported file extension'):
        Sample().save(tmp_path / 'config.toml')
    with pytest.raises(ValueError, match='Unsupported file extension'):
        Sample.load_from_file(tmp_path / 'config.toml')


def test_from_dict_strict_and_lenient() -> None:
    with pytest.raises(ValueError, match='Invalid key: extra'):
        Sample.from_dict({'name': 'x', 'extra': 1})
    assert Sample.from_dict({'name': 'x', 'extra': 1}, strict=False) == Sample(name='x')


def test_update_and_reset() -> None:
    sample = Sample()
    sample.update(lr=0.1, tags=['t'])
    assert sample.lr == 0.1 and sample.tags == ['t']
    with pytest.raises(ValueError, match='Invalid key'):
        sample.update(nope=1)
    sample.update(strict=False, nope=1, steps=2)
    assert sample.steps == 2 and not hasattr(sample, 'nope')
    sample.reset()
    assert sample == Sample()


def test_omegaconf_round_trip() -> None:
    conf = OmegaConf.create({'name': 'x', 'lr': 0.5, 'steps': 1, 'flag': True, 'tags': ['q']})
    assert Sample.from_omegaconf(conf) == Sample(name='x', lr=0.5, steps=1, flag=True, tags=['q'])
    sample = Sample()
    sample.update_from_omegaconf(OmegaConf.create({'steps': 7}))
    assert sample.steps == 7


def test_add_arguments_and_from_args() -> None:
    parser = argparse.ArgumentParser()
    Sample().add_arguments(parser)
    args = parser.parse_args(['--name', 'cli', '--lr', '0.25', '--steps', '4', '--flag', '--tags', 'x', 'y'])
    assert Sample.from_args(args) == Sample(name='cli', lr=0.25, steps=4, flag=True, tags=['x', 'y'])
    assert Sample.from_args(parser.parse_args([])) == Sample()
