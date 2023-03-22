# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from modelling import AdapterTP

import numpy as np
import popxl
import torch
from popxl_addons import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.array_munging import shard

from configs import MagmaConfig, GPTJConfig

from magma.adapters import Adapter as Torch_Adapter


def test_adapters(test_config: MagmaConfig):

    np.random.seed = 42
    torch.seed = 42

    # batch, channels, height, width
    batch_size = test_config.transformer.execution.micro_batch_size
    channels = test_config.visual.embed_dim

    downsample_factor = 8

    input_torch = torch.rand(batch_size, channels, dtype=torch.float32)

    torch_layer = Torch_Adapter(
        dim=channels,
        downsample_factor=downsample_factor,
        activation=torch.nn.ReLU,
        add_layernorm=False,
    )

    output_torch = torch_layer(input_torch)
    output_torch = output_torch.detach().numpy()

    ir = popxl.Ir(replication=test_config.transformer.execution.tensor_parallel)

    g = ir.main_graph

    with g, popxl.in_sequence():

        popxl_layer = AdapterTP(
            config=test_config.transformer, dim=channels, downsample_factor=downsample_factor, add_layernorm=False
        )

        bn_vf, bn_graph = popxl_layer.create_graph(popxl.TensorSpec((batch_size, channels), popxl.float32))

        bn_vars = bn_vf.init()

        bn_bound_graph = bn_graph.bind(bn_vars)

        t = popxl.constant(input_torch)

        (o,) = bn_bound_graph.call(t)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
        popxl.ops.host_store(o_d2h, o)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    # Needed because of replicated_all_reduce_identical_inputs() and replicated_all_reduce_identical_grad_inputs()
    apply_pre_alias_patterns(ir, level="default")

    with popxl.Session(ir, "ipu_model") as session:
        session.write_variables_data(AdapterTP.magma_mapping(test_config.transformer, torch_layer, bn_vars))
        outputs = session.run()

    output_popxl = outputs[o_d2h]

    np.testing.assert_allclose(output_popxl[0].reshape(output_torch.shape), output_torch, rtol=10e-3)
    np.testing.assert_allclose(output_popxl[1].reshape(output_torch.shape), output_torch, rtol=10e-3)


def test_adapters_ln(test_config: MagmaConfig):
    """Test adapters with the optional LayerNorm layer"""

    np.random.seed = 42
    torch.seed = 42

    # Add adapter cfg to test cfg
    test_config.transformer.ff_adapter.mode = "normal"
    test_config.transformer.ff_adapter.layer_norm = True

    # batch, channels, height, width
    batch_size = test_config.transformer.execution.micro_batch_size
    channels = test_config.visual.embed_dim

    downsample_factor = 8

    input_torch = torch.rand(batch_size, channels, dtype=torch.float32)

    torch_layer = Torch_Adapter(
        dim=channels,
        downsample_factor=downsample_factor,
        activation=torch.nn.ReLU,
        add_layernorm=True,
    )

    torch_layer.adapter[0].weight.data = torch.rand(channels, dtype=torch.float32)
    torch_layer.adapter[0].bias.data = torch.rand(channels, dtype=torch.float32)

    output_torch = torch_layer(input_torch)
    output_torch = output_torch.detach().numpy()

    ir = popxl.Ir(replication=test_config.transformer.execution.tensor_parallel)

    g = ir.main_graph

    with g, popxl.in_sequence():

        popxl_layer = AdapterTP(
            config=test_config.transformer, dim=channels, downsample_factor=downsample_factor, add_layernorm=True
        )

        bn_vf, bn_graph = popxl_layer.create_graph(popxl.TensorSpec((batch_size, channels), popxl.float32))

        bn_vars = bn_vf.init()

        bn_bound_graph = bn_graph.bind(bn_vars)

        t = popxl.constant(input_torch)

        (o,) = bn_bound_graph.call(t)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
        popxl.ops.host_store(o_d2h, o)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    # Needed because of replicated_all_reduce_identical_inputs() and replicated_all_reduce_identical_grad_inputs()
    apply_pre_alias_patterns(ir, level="default")

    with popxl.Session(ir, "ipu_model") as session:
        session.write_variables_data(AdapterTP.magma_mapping(test_config.transformer, torch_layer, bn_vars))
        outputs = session.run()

    output_popxl = outputs[o_d2h]

    np.testing.assert_allclose(output_popxl[0].reshape(output_torch.shape), output_torch, rtol=10e-3)
    np.testing.assert_allclose(output_popxl[1].reshape(output_torch.shape), output_torch, rtol=10e-3)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_config.yml")
    test_adapters(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
    test_adapters_ln(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
