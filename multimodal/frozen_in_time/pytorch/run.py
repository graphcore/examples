# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import fire
import wandb

from configs.parse_config import ConfigParser
from modeling.trainer import TrainerIPU

config = None


def run():
    trainer = TrainerIPU(config=config)
    if config["validation_only"]:
        trainer.validate()
    else:
        trainer.train()


def parse_config(
    config_name: str, validation_only: bool = False, compile_only: bool = False, timestamp_ckpt: bool = True, **kwargs
):
    """
    Argument parser

    Additional args must be of the form key=value. Key is of the form
    A.B.C with A.B.C corresponding to a set of keys to be used as input
    to the nested dict loaded through CONFIG_NAME json file.
    See configs/*.json for example configs.

    To override values in the loaded dict, provide (nested) keys in the form A.B.C=value.
    For Example to override arch.type=FrozenInTime in webvid2m-8ipu-1f.json,
    provide the argument --arch.type=new_value.

    To remove values, set the key to None, e.g. --arch.type=None.

    Keys that do not exist in the loaded dict will automatically be added.
    """
    global config
    kwargs["validation_only"] = validation_only
    kwargs["compile_only"] = compile_only
    config = ConfigParser(config_name, timestamp=timestamp_ckpt, **kwargs)


if __name__ == "__main__":
    fire.Fire(parse_config)

    if config._config["trainer"].get("wandb", False):
        wandb.init(
            project=config["trainer"].get("project_name", "torch-frozen-in-time"),
            name=config["trainer"].get("run_name", None),
            config=config._config,
            dir="/tmp",
        )

    run()
