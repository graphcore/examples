# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np


def check_loaded_weights(model, all_initial_weights):
    for layer, initial_weights in zip(model.layers, all_initial_weights):
        weights = layer.get_weights()
        print(f"Layer name {layer.name}")
        print(f"No. of weights in layer: {len(weights)}")
        for weight, initial_weight in zip(weights, initial_weights):
            if np.array_equal(weight, initial_weight):
                print(f'------Checkpoint does not contain weights for weight {weight.shape}------')
            else:
                print(f'++++++Checkpoint contains weights for weight {weight.shape}++++++')


def load_checkpoint_into_model(model, pretrained_ckpt_path):
    all_initial_weights = [layer.get_weights() for layer in model.layers]
    model.load_weights(pretrained_ckpt_path)
    check_loaded_weights(model, all_initial_weights)
