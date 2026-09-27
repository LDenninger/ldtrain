import argparse
import dataclasses
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self


#---------------------------------------------------------------------
# base config
#---------------------------------------------------------------------

@dataclass
class Config:
    def update(self, strict: bool = True, **kwargs: Any) -> None:
        valid_keys = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in valid_keys:
                if strict:
                    raise ValueError(f'Invalid key: {key} ({value})')
                continue
            setattr(self, key, value)

    #---------------------------------------------------------------------
    # management
    #---------------------------------------------------------------------

    def update_from_dict(self, config: dict, strict: bool = True) -> None:
        self.update(strict=strict, **config)

    def update_from_omegaconf(self, config: DictConfig) -> None:
        self.update(**OmegaConf.to_container(config, resolve=True))

    def reset(self) -> None:
        for field in fields(self):
            if field.default is not dataclasses.MISSING:
                setattr(self, field.name, field.default)
            elif field.default_factory is not dataclasses.MISSING:
                setattr(self, field.name, field.default_factory())

    #---------------------------------------------------------------------
    # loader
    #---------------------------------------------------------------------

    @classmethod
    def load_from_file(cls, file_path: str | Path | os.PathLike) -> Self:
        suffix = Path(file_path).suffix
        if suffix == '.json':
            return cls.load_from_json(file_path)
        elif suffix == '.yaml':
            return cls.load_from_yaml(file_path)
        raise ValueError(f'Unsupported file extension: {suffix}')

    @classmethod
    def load_from_json(cls, file_path: str | Path | os.PathLike) -> Self:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def load_from_yaml(cls, file_path: str | Path | os.PathLike) -> Self:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict, strict: bool = True) -> Self:
        valid_keys = {f.name for f in fields(cls)}
        if strict:
            for key in config:
                if key not in valid_keys:
                    raise ValueError(f'Invalid key: {key}')
            return cls(**config)
        config_valid = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**config_valid)

    @classmethod
    def from_omegaconf(cls, config: DictConfig) -> Self:
        return cls(**OmegaConf.to_container(config, resolve=True))

    @classmethod
    def from_args(cls, args: argparse.Namespace, strict: bool = False) -> Self:
        return cls.from_dict(args.__dict__, strict=strict)

    #---------------------------------------------------------------------
    # saver
    #---------------------------------------------------------------------

    def save(self, file_path: str | Path | os.PathLike) -> None:
        suffix = Path(file_path).suffix
        if suffix == '.json':
            self.save_to_json(file_path)
        elif suffix == '.yaml':
            self.save_to_yaml(file_path)
        else:
            raise ValueError(f'Unsupported file extension: {suffix}')

    def save_to_json(self, file_path: str | Path | os.PathLike) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def save_to_yaml(self, file_path: str | Path | os.PathLike) -> None:
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    #---------------------------------------------------------------------
    # serialization
    #---------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    #---------------------------------------------------------------------
    # helpers
    #---------------------------------------------------------------------
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        for key, value in self.to_dict().items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value)
            elif isinstance(value, int):
                parser.add_argument(f'--{key}', type=int, default=value)
            elif isinstance(value, float):
                parser.add_argument(f'--{key}', type=float, default=value)
            elif isinstance(value, str):
                parser.add_argument(f'--{key}', type=str, default=value)
            elif isinstance(value, list):
                parser.add_argument(f'--{key}', nargs='+', default=value)
            elif isinstance(value, dict):
                parser.add_argument(f'--{key}', nargs='+', default=value)
            else:
                raise ValueError(f'Unsupported type: {type(value)}')
