# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from clip.model import ModifiedResNet
from modelling import Stem

import numpy as np
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import torch
from torch import nn
from functools import partial
from configs import MagmaConfig
from popxl_addons import host_store


def test_stem(test_config):

    np.random.seed = 42
    torch.seed = 42

    # batch, channels, height, width
    resnetconfig = test_config.visual
    clip = ModifiedResNet(
        resnetconfig.layers,
        resnetconfig.embed_dim,
        resnetconfig.heads,
        resnetconfig.image_resolution,
        resnetconfig.width,
    )
    clip.bn1.running_mean = torch.rand(clip.bn1.running_mean.shape, dtype=torch.float32)
    clip.bn1.running_var = torch.rand(clip.bn1.running_var.shape, dtype=torch.float32)
    clip = clip.eval()

    batch_size = 4
    channels = 3
    width = resnetconfig.width

    input_torch = torch.rand(
        batch_size, channels, resnetconfig.image_resolution, resnetconfig.image_resolution, dtype=torch.float32
    )
    input_t = TensorInput(input_torch.detach().numpy(), dtype=resnetconfig.dtype)

    # grab stem intermediate output
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    clip.avgpool.register_forward_hook(get_activation("avgpool"))
    final_output = clip(input_torch)
    output_torch = activation["avgpool"].numpy()

    popxl_layer = Stem(resnetconfig)
    (output_popxl,) = run_module(popxl_layer, input_t, weights=partial(Stem.clip_mapping, clip))

    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-2)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_stem(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
