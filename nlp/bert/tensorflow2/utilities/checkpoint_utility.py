# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def check_loaded_weights(model, all_initial_weights):
    for layer, initial_weights in zip(model.layers, all_initial_weights):
        weights = layer.get_weights()
        logging.info(f"Layer name {layer.name}")
        logging.info(f"No. of weights in layer: {len(weights)}")
        for weight, initial_weight in zip(weights, initial_weights):
            if np.array_equal(weight, initial_weight):
                logging.warning(f"------Checkpoint does not contain weights for weight {weight.shape}------")
            else:
                logging.info(f"++++++Checkpoint contains weights for weight {weight.shape}++++++")


def load_checkpoint_into_model(model, pretrained_ckpt_path, expect_partial=False):
    """
    Find all existing checkpoints in given dir and use the newest.
    """

    ckpt_found = False
    all_initial_weights = [layer.get_weights() for layer in model.layers]

    if Path(pretrained_ckpt_path).is_dir():
        ckpt = tf.train.latest_checkpoint(pretrained_ckpt_path)
        ckpt_found = True
    elif Path(pretrained_ckpt_path).is_file() or Path(pretrained_ckpt_path + ".index").is_file():
        ckpt = pretrained_ckpt_path + ".index"
        ckpt_found = True
    else:
        logging.error(
            f"Checkpoint path provided is either invalid cannot be found. "
            "Please provide a valid path to a directory, .ckpt or .ckpt.index "
            "file. exiting."
        )
        sys.exit(1)

    loaded_ckpt = model.load_weights(ckpt)

    if expect_partial:
        loaded_ckpt.expect_partial()

    # compare initial weights for each layer to loaded weights
    check_loaded_weights(model, all_initial_weights)
    logging.info(f"Checkpoint {ckpt} loaded successfully")

    return ckpt_found
