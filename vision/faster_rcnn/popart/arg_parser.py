# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import fire

from yaml_parser import change_cfg_by_yaml_file
from config import cfg


def update_cfg(cfg, all_keys, next_keys, value):
    try:
        if len(next_keys) == 1:
            cfg[next_keys[-1]]
            cfg[next_keys[-1]] = value
        else:
            update_cfg(cfg.get(next_keys[0]), all_keys, next_keys[1:], value)
    except:
        raise KeyError(
            f"{'.'.join(all_keys)} is not a valid key in cfg See config.py for all keys.")


def collect_args_train(YAML, DATA_DIR=None, **kwargs):
    """
    Faster R-CNN training

    Argument collection:

    yaml str: Path to a config located in yamls/
    data_dir str: Path to the location of downloaded data
    kwargs str: any config=value pair that can be found in config.py.
        the format of 'config' should match that in config.py. For example
        _C.SESSION.COMMON.cachePath = 'engine_cache/' can be augmented
        via --SESSION.COMMON.cachePath=other_cache_path
    """

    # load yaml and change cfg
    change_cfg_by_yaml_file(YAML)

    # update cfg with required cmd passed args
    if DATA_DIR is not None:
        cfg["DATA_DIR"] = DATA_DIR
    cfg["YAML"] = YAML

    # update cfg with the rest of cmd passed args
    for key, value in kwargs.items():
        config_keys = key.split(".")
        update_cfg(cfg, config_keys, config_keys, value)


def collect_args_validation(YAML, DATA_DIR=None, EVAL_MODEL_NAME='', **kwargs):
    """
    Faster R-CNN validation

    Argument collection:

    yaml str: Path to a config located in yamls/
    data_dir str: Path to the location of downloaded data
    eval_model_name str: Name of model to evaluate
    kwargs str: any config=value pair that can be found in config.py.
        the format of 'config' should match that in config.py. For example
        _C.SESSION.COMMON.enableStochasticRounding = False can be augmented via
        --SESSION.COMMON.enableStochasticRounding = True
    """

    # load yaml and change cfg
    change_cfg_by_yaml_file(YAML)

    # update cfg with required cmd passed args
    cfg["YAML"] = YAML
    if DATA_DIR is not None:
        cfg["DATA_DIR"] = DATA_DIR
    cfg["EVAL_MODEL_NAME"] = EVAL_MODEL_NAME

    # update cfg with the rest of cmd passed args
    for key, value in kwargs.items():
        config_keys = key.split(".")
        update_cfg(cfg, config_keys, config_keys, value)


def parse_args(train=True):
    if train:
        fire.Fire(collect_args_train)
    else:
        fire.Fire(collect_args_validation)
