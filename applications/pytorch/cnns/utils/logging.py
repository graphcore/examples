# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import random
import string
import os
import logging
import json
import sys
import csv


def get_random_str(strlen=3):
    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    rnd_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    random.setstate(random_state)
    return rnd_str


class Logger:
    @classmethod
    def setup_logging_folder(cls, opts):
        # If it's already configured, skip the reconfiguration
        if hasattr(cls, "logdirname"):
            return

        # get POPLAR_ENGINE_OPTIONS if it exists, as a Python dictionary
        eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
        profile_path = eng_opts.get("autoReport.directory", None)
        options = vars(opts)
        options["POPLAR_ENGINE_OPTIONS"] = eng_opts
        # save to wandb
        if hasattr(opts, "wandb"):
            cls.wandb_logging = opts.wandb
        else:
            cls.wandb_logging = False
        if cls.wandb_logging:
            import wandb
            try:
                wandb.init(project="pytorch-cnn", config=options)
            except:
                # Failed to connect the server --> The logging is offline
                os.environ["WANDB_MODE"] = "dryrun"
                wandb.init(project="pytorch-cnn", config=options)
                logging.info("W&B logging in offline mode")

        # Determine saving folder
        if hasattr(opts, "checkpoint_path") and not opts.checkpoint_path == "":
            cls.logdirname = opts.checkpoint_path
        elif profile_path is not None:
            cls.logdirname = profile_path
        else:
            basename = f'{opts.model}_bs{opts.batch_size}_{opts.precision}fp_r{opts.replicas}_di{opts.device_iterations}'
            while True:
                logdirname = os.path.join("logs", basename + "_" + get_random_str())
                if not os.path.exists(logdirname):
                    break
            cls.logdirname = logdirname
        if not os.path.exists(cls.logdirname):
            os.makedirs(cls.logdirname)
        with open(os.path.join(cls.logdirname, 'app.json'), "w") as f:
            json.dump(options, f)

        # Set up logging
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        # remove stderr output logging
        if len(log.handlers) > 0:
            log.handlers.pop()
        stdout = logging.StreamHandler(sys.stdout)
        stdout_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stdout.setFormatter(stdout_formatter)
        log.addHandler(stdout)
        fileh = logging.FileHandler(os.path.join(cls.logdirname, 'log.txt'), 'a')
        file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(module)s - %(funcName)s: %(message)s')
        fileh.setFormatter(file_formatter)
        log.addHandler(fileh)


    @classmethod
    def log_train_results(cls, results):
        write_to_csv(os.path.join(cls.logdirname, 'training.csv'), results)
        if cls.wandb_logging:
            write_to_wandb(results)

    @classmethod
    def log_validate_results(cls, results):
        write_to_csv(os.path.join(cls.logdirname, 'validation.csv'), results)
        if cls.wandb_logging:
            write_to_wandb(results)


def write_to_csv(filename, results):
    if os.path.exists(filename):
        new_file = False
    else:
        new_file = True
    with open(os.path.join(filename), 'a+') as f:
        w = csv.DictWriter(f, results.keys())
        if new_file:
            w.writeheader()
        w.writerow(results)


def write_to_wandb(results):
    wandb.log(results)
