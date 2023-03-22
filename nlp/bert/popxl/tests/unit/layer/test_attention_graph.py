# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popart._internal.ir as _ir
import popxl

import popxl_addons as addons
from config import BertConfig
from modelling.attention import SelfAttention
from popxl_addons.testing_utils import ops_of_type
from popxl.tensor import Variable


def test_attention_graph(test_config: BertConfig):
    ir = popxl.Ir()
    main = ir.main_graph

    input_shape = (
        test_config.execution.micro_batch_size * test_config.model.sequence_length,
        test_config.model.hidden_size,
    )
    mask_shape = (test_config.execution.micro_batch_size * test_config.model.sequence_length,)

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(np.zeros(input_shape), popxl.float32, name="act"),
                addons.host_load(np.zeros(mask_shape), popxl.float32, name="mask"),
            ]
        )
        act, mask = inputs_tensors
        args, attn_graph = SelfAttention(test_config).create_graph(act, mask)

        attn = attn_graph.bind(args.init())
        call_info = attn.call_with_info(act, mask)
        (act,) = call_info.outputs

    variables = [t for t in main.tensors if isinstance(t, Variable)]
    assert len(variables) == 6
    graph_ops = attn_graph.graph._pb_graph.getOps()
    assert ops_of_type(graph_ops, _ir.op.MatMulOp) == 4
    assert ops_of_type(graph_ops, _ir.op.AddOp) == 4
    assert ops_of_type(graph_ops, _ir.op.SubtractOp) == 1
    assert ops_of_type(graph_ops, _ir.op.MulOp) == 2
    assert ops_of_type(graph_ops, _ir.op.SplitOp) == 1
    assert ops_of_type(graph_ops, _ir.op.ReshapeOp) == 5
    assert ops_of_type(graph_ops, _ir.op.TransposeOp) == 4
    assert ops_of_type(graph_ops, _ir.op.GroupNormOp) == 1

    with main:
        # Don't include mask
        grads_required = [attn_graph.graph.inputs[0], *attn_graph.args.tensors]
        grad_attn_graph = addons.autodiff(attn_graph, grads_required=grads_required)

        seed_gradient = popxl.constant(np.ones(act.shape), act.dtype, "seed_gradient")
        grad_attn_graph.call(seed_gradient, args=grad_attn_graph.grad_graph_info.inputs_dict(call_info))

    bs = test_config.execution.micro_batch_size
    seq = test_config.model.sequence_length
    h = test_config.model.hidden_size
    an = test_config.model.attention.heads
    ah = h // test_config.model.attention.heads

    # Check the correct activations have been attached
    acts_shapes = list(map(lambda t: t.shape, grad_attn_graph.grad_graph_info.inputs))

    assert len(acts_shapes) == 16

    # Gradient, Input to qkvMM, Input to outputMM, Input to groupnorm
    assert acts_shapes.count((bs * seq, h)) == 4
    # groupnorm mean, groupnorm inv_std_dev
    assert acts_shapes.count((bs * seq,)) == 2
    # ???
    assert acts_shapes.count((bs, an, seq, ah)) == 2
    # ???
    assert acts_shapes.count((bs, an, ah, seq)) == 1
    # ???
    assert acts_shapes.count((bs, an, seq, seq)) == 3
    # (1 / math.sqrt(q_act.shape[-1]))
    assert acts_shapes.count(()) == 1

    # Variables
    # qkv.weight
    assert acts_shapes.count((h, h * 3)) == 1
    # output.weight
    assert acts_shapes.count((h, h)) == 1
    # groupnorm.weight
    assert acts_shapes.count((h,)) == 1

    bs = test_config.execution.micro_batch_size
    seq = test_config.model.sequence_length
    h = test_config.model.hidden_size
    an = test_config.model.attention.heads
    ah = h // test_config.model.attention.heads
