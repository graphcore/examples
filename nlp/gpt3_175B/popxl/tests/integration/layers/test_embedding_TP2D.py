# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFGPT2Model

import popxl

import popxl_addons as addons
from modelling.embedding import GPTEmbeddingsTP, GPTEmbeddingsTP2D, generate_positions
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from popxl_addons.array_munging import repeat, unshard, repeat_shard, unshard2D


def test_embedding_TP2D_cmp_TP(test_config: GPTConfig):
    np.random.seed(42)

    # Dropout is disabled with eval: true
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    vocab_size = test_config.model.embedding.vocab_size
    intermediate_size = hidden_size * 4
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # Use HuggingFace model to generate weights
    config = HFConfig(
        hidden_size=hidden_size,
        seq_len=seq_len,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
    )
    hf_model = HFGPT2Model(config).eval()

    def TP(words_idx_np, dy_np):
        ir = popxl.Ir(replication=tp1)
        with ir.main_graph:
            rg_tp1 = ir.replica_grouping()

            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(words_idx_np, popxl.int32, "words"),
                    addons.host_load(dy_np, popxl.float32, "dy"),
                ]
            )
            words, dy = inputs_tensors

            pos = popxl.constant(generate_positions(test_config), popxl.int32, name="pos")

            # Get offsets
            word_offsets_np, pos_offsets_np = GPTEmbeddingsTP.get_offsets(test_config)
            word_offset = popxl.variable(
                word_offsets_np, popxl.int32, "word_offset", replica_grouping=rg_tp1.transpose()
            )
            pos_offset = popxl.variable(pos_offsets_np, popxl.int32, "pos_offset", replica_grouping=rg_tp1.transpose())

            words = words - word_offset
            pos = pos - pos_offset

            facts, graph = GPTEmbeddingsTP(test_config).create_graph(words, pos)

            dgraph = addons.autodiff(graph, grads_required=[graph.args.word.weight, graph.args.positional.weight])

            vars = facts.init()
            fwd_call = graph.bind(vars).call_with_info(words, pos)
            y, *_ = fwd_call.outputs

            dx, *_ = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        weights = GPTEmbeddingsTP.hf_mapping(test_config, vars, hf_model)

        inputs = {h2d: repeat(data, tp1) for h2d, data in zip(inputs_host_steam, inputs_data)}

        with popxl.Session(ir, "ipu_hw") as session:
            session.write_variables_data(weights)
            outputs = session.run(inputs)

        output_y = outputs[y_d2h][0]  # Output from each replica should be identical
        output_dx = unshard(outputs[dx_d2h], 0)

        output_dx = output_dx[:vocab_size, :]  # Remove padding

        return output_y, output_dx

    def TP2D(words_idx_np, dy_np):
        ir = popxl.Ir(replication=tp1 * tp2)

        words_rep = repeat(words_idx_np, tp1 * tp2)
        dy_rep = repeat_shard(dy_np, tp1, tp2, 1)

        with ir.main_graph:
            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(words_rep[0], popxl.int32, "words"),
                    addons.host_load(dy_rep[0], popxl.float32, "dy"),
                ]
            )
            words, dy = inputs_tensors

            pos = popxl.constant(generate_positions(test_config), popxl.int32, name="pos")

            facts, graph = GPTEmbeddingsTP2D(test_config).create_graph(words, pos)

            dgraph = addons.autodiff(graph, grads_required=[graph.args.word.weight, graph.args.positional.weight])

            vars = facts.init()
            fwd_call = graph.bind(vars).call_with_info(words, pos)
            y, *_ = fwd_call.outputs

            dword, dpos, *_ = dgraph.call(dy, args=dgraph.grad_graph_info.inputs_dict(fwd_call))

            y_d2h = addons.host_store(y)
            dx_d2h = addons.host_store(dword)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        weights = GPTEmbeddingsTP2D.hf_mapping(test_config, vars, hf_model)

        inputs = dict(zip(inputs_host_steam, [words_rep, dy_rep]))

        with popxl.Session(ir, "ipu_hw") as session:
            session.write_variables_data(weights)
            outputs = session.run(inputs)

        # Ensure tp1 shards are identical
        y_np = outputs[y_d2h]
        assert y_np.shape[0] == tp1 * tp2
        y_np_splits = np.split(y_np, tp1)
        for t in y_np_splits[1:]:
            np.testing.assert_almost_equal(t, y_np_splits[0])

        y_np = unshard(y_np_splits[0], 1)
        dx_np = unshard2D(outputs[dx_d2h], tp1, tp2, 0, 1)

        dx_np = dx_np[:vocab_size, :]  # Remove padding

        return y_np, dx_np

    words = np.random.random_integers(0, vocab_size, (batch_size, seq_len)).astype("int32")
    dy = np.random.random((batch_size * seq_len, hidden_size)).astype("float32")

    y_local, dx_local = TP(words, dy)
    y_distributed, dx_distributed = TP2D(words, dy)

    np.testing.assert_almost_equal(y_distributed, y_local, decimal=6)
    np.testing.assert_almost_equal(dx_distributed, dx_local, decimal=5)
