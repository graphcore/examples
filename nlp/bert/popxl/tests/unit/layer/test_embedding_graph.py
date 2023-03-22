# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl

import popxl_addons as addons
from config import BertConfig
from modelling.embedding import BertEmbeddings
from popxl_addons.testing_utils import ops_of_type
from popxl.tensor import Variable


def test_embedding_graph(test_config: BertConfig):
    ir = popxl.Ir()
    main = ir.main_graph

    input_shape = (test_config.execution.micro_batch_size * test_config.model.sequence_length,)

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(np.zeros(input_shape), popxl.uint32, name="words"),
                addons.host_load(np.zeros(input_shape), popxl.uint32, name="positions"),
                addons.host_load(np.zeros(input_shape), popxl.uint32, name="token_type"),
            ]
        )
        words, positions, token_type = inputs_tensors
        args, embed_graph = BertEmbeddings(test_config).create_graph(words, positions, token_type)

        embed = embed_graph.bind(args.init())
        (act,) = embed.call(words, positions, token_type)

    variables = [t for t in main.tensors if isinstance(t, Variable)]
    assert len(variables) == 5
    ops = embed_graph.graph._pb_graph.getOps()
    assert ops_of_type(ops, _ir.op.GatherOp) == 3
    assert ops_of_type(ops, _ir.op.AddOp) == 2
    assert ops_of_type(ops, _ir.op.GroupNormOp) == 1
