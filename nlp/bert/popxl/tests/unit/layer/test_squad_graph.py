# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl

import popxl_addons as addons
from config import BertConfig
from modelling.squad import BertSquadHead
from popxl_addons.testing_utils import ops_of_type
from popxl.tensor import Variable


def test_squad_graph(test_config: BertConfig):
    ir = popxl.Ir()
    main = ir.main_graph

    input_shape = (
        test_config.execution.micro_batch_size * test_config.model.sequence_length,
        test_config.model.hidden_size)

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(np.zeros(input_shape), popxl.float32, name="act"),
        ])
        args, attn_graph = BertSquadHead(
            test_config).create_graph(*inputs_tensors)

        attn = attn_graph.bind(args.init())
        act, = attn.call(*inputs_tensors)

    variables = [t for t in main.tensors if isinstance(t, Variable)]
    assert len(variables) == 2
    ops = attn_graph.graph._pb_graph.getOps()
    assert ops_of_type(ops, _ir.op.MatMulOp) == 1
    assert ops_of_type(ops, _ir.op.AddOp) == 1
