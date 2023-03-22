# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

import popxl

import popxl_addons
import popxl_addons as addons
from popxl_addons import NamedVariableFactories
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.embedding import EmbeddingTP2D
from modelling.gpt_lm import GPTLMHeadLossAndGradTP, GPTLMHeadLossTP2D, HeadFwdBwd
from popxl_addons.array_munging import repeat, shard, repeat_shard, shard2D, unshard


def test_lm_loss_and_grad_TP2D_cmp_TP(test_config: GPTConfig):
    np.random.seed(42)

    # Dropout is disabled with eval: true
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    vocab_size = test_config.model.embedding.vocab_size
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    def TP(x_np, labels_np, word_emb_np, word_emb_accum_np):
        ir = popxl.Ir(replication=tp1)

        rg_tp1 = ir.replica_grouping()

        x_rep = repeat(x_np, tp1)
        labels_rep = repeat(labels_np, tp1)
        word_emb_sharded = shard(word_emb_np, tp1, 0)
        word_emb_accum_sharded = shard(word_emb_accum_np, tp1, 0)

        with ir.main_graph:
            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(x_rep[0], popxl.float32, "x"),
                    addons.host_load(labels_rep[0], popxl.int32, "labels"),
                    addons.host_load(word_emb_sharded[0], popxl.float32, "word_emb"),
                    addons.host_load(word_emb_accum_sharded[0], popxl.float32, "word_emb_accum"),
                ]
            )
            x, labels, word_emb, word_emb_accum = inputs_tensors

            word_offset = popxl.variable(
                EmbeddingTP2D.get_offsets(vocab_size, tp1),
                popxl.int32,
                "word_offsets",
                replica_grouping=rg_tp1.transpose(),
            )

            labels_offseted = labels - word_offset

            facts, graph = GPTLMHeadLossAndGradTP(test_config).create_graph(
                x, labels_offseted, word_emb.T, word_emb_accum.T, word_offset
            )

            vars = facts.init()
            fwd_call = graph.bind(vars).call_with_info(x, labels_offseted, word_emb.T, word_emb_accum.T, word_offset)
            loss, dx = fwd_call.outputs

            loss_d2h = addons.host_store(loss)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        inputs = dict(zip(inputs_host_steam, [x_rep, labels_rep, word_emb_sharded, word_emb_accum_sharded]))

        with popxl.Session(ir, "ipu_hw") as session:
            outputs = session.run(inputs)

        output_loss = outputs[loss_d2h][0]  # Output from each replica should be identical
        output_dx = outputs[dx_d2h][0]

        return output_loss, output_dx

    def TP2D(x_np, labels_np, word_emb_np, word_emb_accum_np):
        ir = popxl.Ir(replication=tp1 * tp2)

        x_rep = repeat_shard(x_np, tp1, tp2, 1)
        labels_rep = repeat(labels_np, tp1 * tp2)
        word_emb_sharded = shard2D(word_emb_np, tp1, tp2, 0, 1)
        word_emb_accum_sharded = shard2D(word_emb_accum_np, tp1, tp2, 0, 1)

        with ir.main_graph:
            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(x_rep[0], popxl.float32, "x"),
                    addons.host_load(labels_rep[0], popxl.int32, "labels"),
                    addons.host_load(word_emb_sharded[0], popxl.float32, "word_emb"),
                    addons.host_load(word_emb_accum_sharded[0], popxl.float32, "word_emb_accum"),
                ]
            )
            x, labels, word_emb, word_emb_accum = inputs_tensors

            facts_fwd, graph_fwd = GPTLMHeadLossTP2D(test_config).create_graph(x, labels)
            facts_fwd.head.pop("word_embedding")

            graph_bwd = popxl_addons.autodiff(graph_fwd, grads_required=graph_fwd.graph.inputs[:1])
            facts_bwd = NamedVariableFactories()  # Empty

            facts_fwdbwd, graph_fwdbwd = HeadFwdBwd(
                test_config, graph_fwd, graph_bwd, facts_fwd, facts_bwd
            ).create_graph(x, labels, word_emb, word_emb_accum)

            vars = facts_fwdbwd.init()
            fwd_call = graph_fwdbwd.bind(vars).call_with_info(x, labels, word_emb, word_emb_accum)
            loss, dx = fwd_call.outputs

            loss_d2h = addons.host_store(loss)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        inputs = dict(zip(inputs_host_steam, [x_rep, labels_rep, word_emb_sharded, word_emb_accum_sharded]))

        with popxl.Session(ir, "ipu_hw") as session:
            outputs = session.run(inputs)

        # Ensure loss is identical
        output_loss = outputs[loss_d2h][0]
        for i in range(tp1 * tp2):
            np.testing.assert_almost_equal(outputs[loss_d2h][i], output_loss)

        # Ensure tp1 shards are identical
        dx_data = outputs[dx_d2h]
        assert len(dx_data) == tp1 * tp2
        dx_np_splits = np.split(dx_data, tp1)
        dx_unshard_0 = unshard(dx_np_splits[0], 1)
        for i in range(1, tp2):
            dx_unshard_i = unshard(dx_np_splits[i], 1)
            np.testing.assert_almost_equal(dx_unshard_0, dx_unshard_i)

        return output_loss, dx_unshard_0

    x = np.random.random((batch_size * seq_len, hidden_size)).astype("float32")
    labels = np.random.randint(0, vocab_size, (batch_size * seq_len)).astype("int32")
    word_emb_np = np.random.random((vocab_size, hidden_size)).astype("float32")
    word_emb_accum_np = np.random.random((vocab_size, hidden_size)).astype("float32")

    loss_local, dx_local = TP(x, labels, word_emb_np, word_emb_accum_np)
    loss_distributed, dx_distributed = TP2D(x, labels, word_emb_np, word_emb_accum_np)

    np.testing.assert_almost_equal(loss_distributed, loss_local, decimal=6)
    np.testing.assert_almost_equal(dx_distributed, dx_local, decimal=6)
