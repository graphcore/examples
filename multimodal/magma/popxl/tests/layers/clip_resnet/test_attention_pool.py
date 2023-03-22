# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from clip.model import AttentionPool2d
from modelling import AttentionPool

import numpy as np
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import torch
from functools import partial
from configs import MagmaConfig


def test_pool_attention(test_config):

    np.random.seed = 42
    torch.seed = 42

    # batch, channels, height, width
    resnetconfig = test_config.visual
    batch_size = 1
    channels = test_config.visual.embed_dim
    spatial_dim = resnetconfig.image_resolution // 32
    input_torch = torch.rand(batch_size, channels, spatial_dim, spatial_dim, dtype=torch.float32)
    input_t = TensorInput(input_torch.detach().numpy(), dtype=resnetconfig.dtype)

    torch_layer = AttentionPool2d(
        spacial_dim=spatial_dim,
        embed_dim=resnetconfig.embed_dim,
        num_heads=resnetconfig.heads,
        output_dim=resnetconfig.embed_dim,
    )
    output_torch = torch_layer(input_torch)
    output_torch = output_torch.detach().numpy()

    popxl_layer = AttentionPool(resnetconfig)
    (output_popxl,) = run_module(popxl_layer, input_t, weights=partial(AttentionPool.clip_mapping, torch_layer))
    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-3)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_pool_attention(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
