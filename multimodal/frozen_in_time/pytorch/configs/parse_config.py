# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import inspect
import json
import logging
import logging.config
import os
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def read_json(fname):
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def setup_logging(save_dir, log_config="configs/logger_config.json", default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # Modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


class ConfigParser:
    def __init__(self, config, timestamp=True, test=False, **kwargs):
        # Parse default and custom cli options
        self.cfg_fname = Path(config)
        self._config = read_json(self.cfg_fname)

        # Set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])
        timestamp = datetime.now().strftime(r"%m%d_%H%M%S") if timestamp else ""

        exper_name = self.config["name"]
        self._save_dir = save_dir / "models" / exper_name / timestamp
        self._web_log_dir = save_dir / "web" / exper_name / timestamp
        self._log_dir = save_dir / "log" / exper_name / timestamp

        if not test:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # If set, remove all previous experiments with the current config
        if kwargs.get("purge_exp_dir", False):
            for dirpath in (self._save_dir, self._log_dir, self._web_log_dir):
                config_dir = dirpath.parent
                existing = list(config_dir.glob("*"))
                print(f"purging {len(existing)} directories from config_dir...")
                tic = time.time()
                os.system(f"rm -rf {config_dir}")
                print(f"Finished purge in {time.time() - tic:.3f}s")

        # Save updated config file to the checkpoint dir
        if not test:
            write_json(self.config, self.save_dir / "config.json")

            # Configure logging module
            setup_logging(self.log_dir)
            self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

        # update json config with cli passed args
        if len(kwargs) > 0:
            self.update_config(**kwargs)

        # create derived config values such as global batch size
        self.set_derived_config_values()

    def update_config(self, **kwargs):
        """
        Takes cli parsed strings of the form a.b.c = value
        and sets {a: {b: {c: value}}} in self.config:
            - If any of the keys A.B.C do not exist in self._config, they
            are created
            - Deletes entries in self._config if the user provides
            a.b.c = None
            - Overrides existing values in self._config if the last
            key in the string exists in self._config
        """
        logger = self.get_logger("warning")
        logger.info("Updating config with CLI passed arguments")

        for nested_key, value in kwargs.items():
            local_config = self._config
            nested_keys = nested_key.split(".")
            for k in nested_keys[:-1]:
                local_config = local_config.setdefault(k, {})
            if nested_keys[-1] in local_config:
                if value is None:
                    logger.warn(f"Removing: {nested_key} from: {self.cfg_fname}.")
                    local_config.pop(nested_keys[-1])
                else:
                    logger.warn(f"Overriding: {nested_key} with value: {value}.")
                    local_config[nested_keys[-1]] = value
            else:
                logger.warn(
                    f"(nested) key: {nested_key} not found in: {self.cfg_fname}." f" Setting key to value: {value}."
                )
                local_config[nested_keys[-1]] = value

    def set_derived_config_values(self):
        # global batch size
        self._config["trainer"]["global_batch_size"] = (
            self.config["data_loader"]["training"]["batch_size"]
            * self.config["IPU_options"]["training"]["gradientAccumulation"]
            * self.config["IPU_options"]["training"].get("replication_factor", 1)
        )

    def initialize(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(k not in module_args for k in kwargs), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        # If parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)
        for param in signature.parameters.keys():
            if param not in module_args and param in self.config:
                module_args[param] = self[param]
                print(f"{param} was not in {name}. insert with global setting {self[param]}")

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
