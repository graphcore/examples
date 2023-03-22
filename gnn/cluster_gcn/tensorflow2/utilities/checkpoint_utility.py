# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

import numpy as np


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


def load_checkpoint_into_model(model, ckpt_path):
    logging.info("Attempting to load checkpoint from" f" path {ckpt_path}.")
    all_initial_weights = [layer.get_weights() for layer in model.layers]
    model.load_weights(ckpt_path)
    check_loaded_weights(model, all_initial_weights)
