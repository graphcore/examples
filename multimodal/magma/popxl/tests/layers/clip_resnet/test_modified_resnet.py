# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from clip.model import ModifiedResNet as ClipModifiedResnet
from modelling import ModifiedResNet

import numpy as np
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import torch
from torch import nn
from functools import partial
from configs import MagmaConfig
from popxl_addons import host_store


def test_modified_resnet(test_config):

    np.random.seed = 42
    torch.seed = 42

    # batch, channels, height, width
    resnetconfig = test_config.visual
    clip = ClipModifiedResnet(
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

    input_torch = torch.rand(
        batch_size, channels, resnetconfig.image_resolution, resnetconfig.image_resolution, dtype=torch.float32
    )
    input_t = TensorInput(input_torch.detach().numpy(), dtype=resnetconfig.dtype)

    output_torch = clip(input_torch).detach().numpy()

    popxl_layer = ModifiedResNet(resnetconfig, pool=True)
    (output_popxl,) = run_module(
        popxl_layer, input_t, weights=partial(ModifiedResNet.clip_mapping, clip, resnetconfig, pool=True)
    )
    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-3)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_modified_resnet(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
