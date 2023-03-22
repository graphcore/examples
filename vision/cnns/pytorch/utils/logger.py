# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import random
import string
import os
import logging
import json
import sys
import csv
import wandb
import datetime


def get_random_str(strlen=3):
    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    rnd_str = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    random.setstate(random_state)
    return rnd_str


class Logger:
    @classmethod
    def setup_logging_folder(cls, args):
        # During distributed training log only single process
        if hasattr(args, "use_popdist") and args.use_popdist and args.popdist_rank != 0:
            cls.silent_process = True
            return
        else:
            cls.silent_process = False
        # If it's already configured, skip the reconfiguration
        if hasattr(cls, "logdirname"):
            return

        # get POPLAR_ENGINE_OPTIONS if it exists, as a Python dictionary
        eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
        profile_path = eng_opts.get("autoReport.directory", None)
        options = vars(args)
        options["POPLAR_ENGINE_OPTIONS"] = eng_opts
        # save to wandb
        if hasattr(args, "wandb"):
            cls.wandb_logging = args.wandb
        else:
            cls.wandb_logging = False
        if cls.wandb_logging:
            name = f"{args.config}-{str(datetime.datetime.now())}"
            try:
                wandb.init(project="pytorch-cnn", config=options, name=name)
            except:
                # Failed to connect the server --> The logging is offline
                os.environ["WANDB_MODE"] = "dryrun"
                wandb.init(project="pytorch-cnn", config=options, name=name)
                logging.info("W&B logging in offline mode")

        # Determine saving folder
        if hasattr(args, "checkpoint_output_dir") and not args.checkpoint_output_dir == "":
            cls.logdirname = args.checkpoint_output_dir
        elif profile_path is not None:
            cls.logdirname = profile_path
        else:
            basename = (
                f"{args.model}_bs{args.micro_batch_size}_{args.precision}fp_r{args.replicas}_di{args.device_iterations}"
            )
            while True:
                logdirname = os.path.join("logs", basename + "_" + get_random_str())
                if not os.path.exists(logdirname):
                    break
            cls.logdirname = logdirname
        if not os.path.exists(cls.logdirname):
            os.makedirs(cls.logdirname)
        with open(os.path.join(cls.logdirname, "app.json"), "w") as f:
            json.dump(options, f)

        # Set up logging
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        # remove stderr output logging
        if len(log.handlers) > 0:
            log.handlers.pop()
        stdout = logging.StreamHandler(sys.stdout)
        stdout_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        stdout.setFormatter(stdout_formatter)
        log.addHandler(stdout)
        fileh = logging.FileHandler(os.path.join(cls.logdirname, "log.txt"), "a")
        file_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(module)s - %(funcName)s: %(message)s")
        fileh.setFormatter(file_formatter)
        log.addHandler(fileh)

    @classmethod
    def log_train_results(cls, results):
        if not cls.silent_process:
            write_to_csv(os.path.join(cls.logdirname, "training.csv"), results)
            if cls.wandb_logging:
                write_to_wandb(results)

    @classmethod
    def log_validate_results(cls, results):
        if not cls.silent_process:
            write_to_csv(os.path.join(cls.logdirname, "validation.csv"), results)
            if cls.wandb_logging:
                write_to_wandb(results)

    @classmethod
    def log_model_histogram(cls, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                wandb.run.history.torch.log_tensor_stats(parameter.data, name)


def write_to_csv(filename, results):
    if os.path.exists(filename):
        new_file = False
    else:
        new_file = True
    with open(os.path.join(filename), "a+") as f:
        w = csv.DictWriter(f, results.keys())
        if new_file:
            w.writeheader()
        w.writerow(results)


def write_to_wandb(results):
    wandb.log(results)
