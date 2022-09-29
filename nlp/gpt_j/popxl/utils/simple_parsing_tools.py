# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass, fields

import argparse
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable, field
from simple_parsing.helpers.serialization.decoding import register_decoding_fn
from simple_parsing.helpers.serialization.encoding import encode
from simple_parsing.utils import Dataclass, DataclassType


@dataclass
class Config(Serializable):
    pass


def flag(default: bool, *args, **kwargs):
    """Use this to have a bool argument that's true when used as a flag.
        For example: "--dropout" always means True"""
    return field(default=default, nargs="?", const=True, *args, **kwargs)


def set_dataclass_defaults(dataclass: DataclassType[Dataclass], defaults: Dataclass):
    assert isinstance(defaults, dataclass)

    for field in fields(dataclass):
        field.default = getattr(defaults, field.name)


def parse_args_with_config_file(dclass: DataclassType[Dataclass], *args) -> Dataclass:
    cparser = argparse.ArgumentParser("Config Parser", add_help=False)
    cparser.add_argument("--config", type=str)
    cargs, remaining_argv = cparser.parse_known_args(*args)

    defaults = dclass()
    if cargs.config is not None:
        set_dataclass_defaults(dclass,
                               dclass.load(cargs.config, drop_extra_fields=False))

    dclass_dest = dclass.__name__.lower()

    parser = ArgumentParser()
    parser.add_arguments(dclass, dest=dclass_dest)

    # This is here only for the help message
    parser.add_argument("--config", type=str,
                        help="Path to preset config")

    args = parser.parse_args(remaining_argv)
    if cargs.config is not None:
        set_dataclass_defaults(dclass, defaults)
    return getattr(args, dclass_dest)


def parse_args_with_presets(
        dclass: DataclassType[Dataclass],
        config_file: Union[str, Path],
        presets_key: str,
        default: str,
        custom_args: Optional[Callable[[ArgumentParser], None]] = None,
        CLI_args: Optional[str] = None,
) -> Tuple[Dataclass, argparse.Namespace]:
    config_file = str(config_file)
    if config_file.endswith((".yml", ".yaml")):
        with open(config_file) as fp:
            config_dict: Dict[str, Any] = yaml.full_load(fp)
    elif config_file.endswith(".json"):
        with open(config_file) as fp:
            config_dict: Dict[str, Any] = json.load(fp)
    else:
        raise ValueError("Unknown config file type.")

    if presets_key:
        config_dict = config_dict[presets_key]

    presets = list(config_dict.keys())

    cparser = argparse.ArgumentParser("Config Parser", add_help=False)
    cparser.add_argument("--config", choices=presets,
                         type=str, default=default)
    cargs, remaining_argv = cparser.parse_known_args(CLI_args)

    defaults = dclass()
    if cargs.config is not None:
        set_dataclass_defaults(dclass,
                               dclass.from_dict(config_dict[cargs.config], drop_extra_fields=False))

    dclass_dest = dclass.__name__.lower()

    parser = ArgumentParser()
    parser.add_arguments(dclass, dest=dclass_dest)

    # This is here only for the help message
    parser.add_argument("--config", choices=presets, type=str, default=default,
                        help=f"Preset Configs from {config_file}")

    if custom_args:
        custom_args(parser)

    args = parser.parse_args(remaining_argv)
    if cargs.config is not None:
        set_dataclass_defaults(dclass, defaults)
        args.config = cargs.config

    config = getattr(args, dclass_dest)
    return config, args


class Choice(Enum):
    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "decode"):
            def decode(value):
                return cls[value]
            setattr(cls, "decode", decode)
        register_decoding_fn(cls, cls.decode)

        if not hasattr(cls, "encode"):
            def _encode(enum):
                return str(enum.name)
            setattr(cls, "encode", _encode)
        encode.register(cls, cls.encode)
