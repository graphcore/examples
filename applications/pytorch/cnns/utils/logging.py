# Copyright 2020 Graphcore Ltd.

import random
import string
import os
import logging
import json
import sys


def get_random_str(strlen=3):
    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    rnd_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    random.setstate(random_state)
    return rnd_str


def setup_logging_folder(opts):

    # get POPLAR_ENGINE_OPTIONS if it exists, as a Python dictionary
    eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))

    # check if we are doing logging
    if any([key.startswith("autoReport") for key in eng_opts.keys()]):

        # check if autoReport.directory exists and treat it as a parent folder
        parent = eng_opts.get("autoReport.directory", ".")
        # check if parent is directory
        if os.path.exists(parent) and not os.path.isdir(parent):
            raise ValueError(f"{parent} is not a directory!")
        elif not os.path.exists(parent):
            os.makedirs(parent)

        basename = f'{opts.model}_bs{opts.batch_size}_{opts.precision}fp_r{opts.replicas}_di{opts.device_iteration}'
        for _ in range(10):
            logdirname = os.path.join(parent, basename + "_" + get_random_str())
            if not os.path.exists(logdirname):
                break
        eng_opts["autoReport.directory"] = logdirname
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(eng_opts)
        logging.info(f"logging to {logdirname}")
        logging.info(f"POPLAR_ENGINE_OPTIONS: {json.dumps(eng_opts)}")
    else:
        logging.info("No logging...")
