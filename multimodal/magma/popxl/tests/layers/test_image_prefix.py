# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from magma.image_prefix import ImagePrefix as MagmaImagePrefix
from magma import MultimodalConfig

from modelling import ImagePrefix

import numpy as np
from popxl_addons.testing_utils import run_module, TensorInput
import popxl
import torch
from torch import nn
from functools import partial
from configs import MagmaConfig
from popxl_addons import host_store
import popxl_addons as addons


def test_image_prefix(test_config):

    np.random.seed = 42
    torch.seed = 42
    resnetconfig = test_config.visual

    batch_size = 4
    channels = 3

    magma_config = MultimodalConfig(batch_size, 0)
    magma_config.use_image_embed_layernorm = True
    magma_config.encoder_name = "clip_resnet_large"
    magma_image_prefix = MagmaImagePrefix(magma_config, out_dim=test_config.transformer.hidden_size, device="cpu")
    resnetconfig.width = 96
    magma_config.image_size = resnetconfig.image_resolution
    magma_image_prefix = magma_image_prefix.eval()
    magma_image_prefix.float()
    # batch, channels, height, width
    input_torch = torch.rand(
        batch_size, channels, resnetconfig.image_resolution, resnetconfig.image_resolution, dtype=torch.float32
    )

    input_t = TensorInput(input_torch.detach().numpy())

    output_torch = magma_image_prefix(input_torch).detach().numpy()

    popxl_layer = ImagePrefix(test_config)
    ir = popxl.Ir()
    with ir.main_graph:
        x = popxl.constant(input_torch.detach().numpy())
        facts, g = popxl_layer.create_graph(x)
        vars = facts.init()
        (out,) = g.bind(vars).call(x)
        d2h = host_store(out)

    with popxl.Session(ir, "ipu_hw") as session:
        wmap = ImagePrefix.magma_mapping(magma_image_prefix, test_config, vars)
        session.write_variables_data(wmap)

        output_popxl = session.run()[d2h]
    np.testing.assert_allclose(output_popxl, output_torch.reshape(output_popxl.shape), rtol=10e-2)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_image_prefix(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
