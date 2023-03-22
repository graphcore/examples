# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl

import popxl_addons as addons
from config import BertConfig
from modelling.feed_forward import FeedForward
from popxl_addons.testing_utils import ops_of_type
from popxl.tensor import Variable


def test_feed_forward_graph(test_config: BertConfig):
    ir = popxl.Ir()
    main = ir.main_graph

    input_shape = (
        test_config.execution.micro_batch_size * test_config.model.sequence_length,
        test_config.model.hidden_size,
    )

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(np.zeros(input_shape), popxl.float32, name="input"),
            ]
        )
        input_t = inputs_tensors[0]
        args, ff_graph = FeedForward(test_config).create_graph(input_t)

        ff = ff_graph.bind(args.init())
        call_info = ff.call_with_info(input_t)
        (act,) = call_info.outputs

    variables = [t for t in main.tensors if isinstance(t, Variable)]
    assert len(variables) == 6
    ops = ff_graph.graph._pb_graph.getOps()

    assert ops_of_type(ops, _ir.op.MatMulOp) == 2
    assert ops_of_type(ops, _ir.op.AddOp) == 3
    assert ops_of_type(ops, _ir.op.GeluOp) == 1
    assert ops_of_type(ops, _ir.op.GroupNormOp) == 1

    with main:
        grad_ff_graph = addons.autodiff(ff_graph)

        seed_gradient = popxl.constant(np.ones(act.shape), act.dtype, "seed_gradient")
        grad_ff_graph.call(seed_gradient, args=grad_ff_graph.grad_graph_info.inputs_dict(call_info))

    bs = test_config.execution.micro_batch_size
    seq = test_config.model.sequence_length
    h = test_config.model.hidden_size
    an = test_config.model.attention.heads
    ah = h // test_config.model.attention.heads

    # Check the correct activations have been attached
    acts_shapes = list(map(lambda t: t.shape, grad_ff_graph.grad_graph_info.inputs))

    assert len(acts_shapes) == 10

    # Gradient, Input to intermediateMM, Input to groupnorm
    assert acts_shapes.count((bs * seq, h)) == 3
    # groupnorm mean, groupnorm inv_std_dev
    assert acts_shapes.count((bs * seq,)) == 2
    # Input to outputMM, Input to Gelu
    assert acts_shapes.count((bs * seq, h * 4)) == 2

    # Variables
    # intermediate.weight
    assert acts_shapes.count((h, h * 4)) == 1
    # output.weight
    assert acts_shapes.count((h * 4, h)) == 1
    # groupnorm.weight
    assert acts_shapes.count((h,)) == 1
