# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl.ops as ops
import torch
from modelling import BatchNorm2D
from popxl_addons import NamedTensors

import popxl


def test_batch_norm():
    np.random.seed = 42
    torch.seed = 42

    batch_size = 2
    channel_num = 4
    H = 6
    W = 8

    bn_scale = np.random.uniform(low=0.5, high=1.5, size=channel_num).astype(np.float32)
    bn_bias = np.random.normal(loc=0.0, scale=1.0, size=channel_num).astype(np.float32)
    bn_mean = np.random.normal(loc=0.0, scale=1.0, size=channel_num).astype(np.float32)
    bn_var = np.random.uniform(low=0.5, high=1.5, size=channel_num).astype(np.float32)
    bn_t = 4 * np.random.normal(loc=0.0, scale=1.0, size=(batch_size, channel_num, H, W)).astype(np.float32)

    def parameters_mapping(variables: NamedTensors):
        return {
            variables.weight: bn_scale,
            variables.bias: bn_bias,
            variables.running_mean: bn_mean,
            variables.running_var: bn_var,
        }

    ir = popxl.Ir()

    g = ir.main_graph

    with g, popxl.in_sequence():

        bn_vf, bn_graph = BatchNorm2D().create_graph(popxl.TensorSpec((batch_size, channel_num, H, W), popxl.float32))

        bn_vars = bn_vf.init()

        bn_bound_graph = bn_graph.bind(bn_vars)

        t = popxl.constant(bn_t)

        (o,) = bn_bound_graph.call(t)

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_model") as session:
        session.write_variables_data(parameters_mapping(bn_vars))
        outputs = session.run()

    output_popxl = outputs[o_d2h]

    # Now let's check it against the PyTorch implementation

    bn_torch = torch.nn.BatchNorm2d(num_features=channel_num)
    bn_torch.training = False

    bn_torch.weight.data = torch.Tensor(bn_scale)
    bn_torch.bias.data = torch.Tensor(bn_bias)

    bn_torch.running_mean.data = torch.Tensor(bn_mean)
    bn_torch.running_var.data = torch.Tensor(bn_var)

    output_torch = bn_torch(torch.Tensor(bn_t)).detach().numpy()

    np.testing.assert_allclose(output_torch, output_popxl, rtol=10e-3)


if __name__ == "__main__":
    test_batch_norm()
