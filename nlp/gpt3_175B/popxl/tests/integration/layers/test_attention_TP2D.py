# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import popxl

import popxl_addons as addons
from modelling.attention import GPTSelfAttentionTP, GPTSelfAttentionTP2D
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from popxl_addons.array_munging import repeat, unshard, repeat_shard


def test_attention_TP2D_cmp_TP(test_config: GPTConfig):
    np.random.seed(42)

    # Dropout is disabled with eval: true
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # Use HuggingFace model to generate weights
    config = HFConfig(
        hidden_size=hidden_size, seq_len=seq_len, n_inner=intermediate_size, n_head=test_config.model.attention.heads
    )
    hf_model = GPT2Block(config).eval()

    def TP(x_np, dy_np):
        ir = popxl.Ir(replication=tp1)
        with ir.main_graph:
            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(x_np, popxl.float32, "x"),
                    addons.host_load(dy_np, popxl.float32, "dy"),
                ]
            )
            x, dy = inputs_tensors

            facts, graph = GPTSelfAttentionTP(test_config).create_graph(x)

            dgraph = addons.autodiff(graph)

            vars = facts.init()
            fwd_call = graph.bind(vars).call_with_info(x)
            y, *_ = fwd_call.outputs

            dx, *_ = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        weights = GPTSelfAttentionTP.hf_mapping(test_config, vars, hf_model)

        inputs = {h2d: repeat(data, tp1) for h2d, data in zip(inputs_host_steam, inputs_data)}

        with popxl.Session(ir, "ipu_hw") as session:
            session.write_variables_data(weights)
            outputs = session.run(inputs)

        output_y = outputs[y_d2h][0]  # Output from each replica should be identical
        output_dx = outputs[dx_d2h][0]

        return output_y, output_dx

    def TP2D(x_np, dy_np):
        ir = popxl.Ir(replication=tp1 * tp2)

        x_rep = repeat_shard(x_np, tp1, tp2, 1)
        dy_rep = repeat_shard(dy_np, tp1, tp2, 1)

        with ir.main_graph:
            _, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(x_rep[0], popxl.float32, "x"),
                    addons.host_load(dy_rep[0], popxl.float32, "dy"),
                ]
            )
            x, dy = inputs_tensors

            facts, graph = GPTSelfAttentionTP2D(test_config).create_graph(x)

            dgraph = addons.autodiff(graph)

            vars = facts.init()
            fwd_call = graph.bind(vars).call_with_info(x)
            y, *_ = fwd_call.outputs

            dx, *_ = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        weights = GPTSelfAttentionTP2D.hf_mapping(test_config, vars, hf_model)

        inputs = dict(zip(inputs_host_steam, [x_rep, dy_rep]))

        with popxl.Session(ir, "ipu_hw") as session:
            session.write_variables_data(weights)
            outputs = session.run(inputs)

        # Ensure tp1 shards are identical
        y_np = outputs[y_d2h]
        assert y_np.shape[0] == tp1 * tp2
        y_np_splits = np.split(y_np, tp1)
        for t in y_np_splits[1:]:
            np.testing.assert_almost_equal(t, y_np_splits[0])

        dx_np = outputs[dx_d2h]
        assert dx_np.shape[0] == tp1 * tp2
        dx_np_splits = np.split(dx_np, tp1)
        for t in dx_np_splits[1:]:
            np.testing.assert_almost_equal(t, dx_np_splits[0])

        y_np = unshard(y_np_splits[0], 1)
        dx_np = unshard(dx_np_splits[0], 1)

        return y_np, dx_np

    x = np.random.random((batch_size * seq_len, hidden_size)).astype("float32")
    dy = np.random.random((batch_size * seq_len, hidden_size)).astype("float32")

    y_local, dx_local = TP(x, dy)
    y_distributed, dx_distributed = TP2D(x, dy)

    np.testing.assert_almost_equal(y_distributed, y_local, decimal=6)
    np.testing.assert_almost_equal(dx_distributed, dx_local, decimal=5)
