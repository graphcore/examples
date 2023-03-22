# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from clip.model import Bottleneck as ClipBottleneck
from modelling import Bottleneck

import numpy as np
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import torch
from functools import partial
from configs import MagmaConfig


def test_bottleneck(test_config):
    np.random.seed = 42
    torch.seed = 42

    # batch, channels, height, width
    resnetconfig = test_config.visual
    batch_size = 2
    width = resnetconfig.width
    in_channels = resnetconfig.width
    out_channels = 2 * in_channels
    input_torch = torch.rand(batch_size, in_channels, width, width, dtype=torch.float32)
    input_t = TensorInput(input_torch.detach().numpy(), dtype=resnetconfig.dtype)

    torch_layer = ClipBottleneck(in_channels, out_channels, stride=2)
    torch_layer.bn2.running_mean = torch.rand(torch_layer.bn2.running_mean.shape, dtype=torch.float32)
    torch_layer.bn2.running_var = torch.rand(torch_layer.bn2.running_var.shape, dtype=torch.float32)
    torch_layer = torch_layer.eval()
    output_torch = torch_layer(input_torch)
    output_torch = output_torch.detach().numpy()

    popxl_layer = Bottleneck(in_channels, out_channels, stride=2)
    (output_popxl,) = run_module(popxl_layer, input_t, weights=partial(Bottleneck.clip_mapping, torch_layer))
    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-3)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_bottleneck(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
