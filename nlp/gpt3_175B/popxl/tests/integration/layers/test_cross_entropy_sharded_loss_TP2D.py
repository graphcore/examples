# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

import popxl

import popxl_addons
import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.gpt_lm import CrossEntropyShardedLoss
from popxl_addons.array_munging import repeat, shard, repeat_shard, unshard


def test_cross_entropy_sharded_loss_TP2D_cmp_pytorch(test_config: GPTConfig):
    np.random.seed(42)

    # Make vocab large for test as variation in loss can be small
    test_config.model.embedding.vocab_size = 512

    # Dropout is disabled with eval: true
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    vocab_size = test_config.model.embedding.vocab_size
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    def PyTorch(logits_np, labels_np, loss_bwd_np):
        logits_torch = torch.tensor(logits_np, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(
            logits_torch, torch.tensor(labels_np, dtype=torch.long), ignore_index=0, reduction="mean"
        )

        loss.backward(torch.tensor(loss_bwd_np))
        dx = logits_torch.grad

        return loss.detach().numpy(), dx.detach().numpy()

    def TP2D(logits_np, labels_np, loss_bwd_np):
        ir = popxl.Ir(replication=tp1 * tp2)

        logits_rep = repeat_shard(logits_np, tp2, tp1, 1, "strided")
        labels_rep = repeat(labels_np, tp1 * tp2)
        loss_bwd_rep = repeat(loss_bwd_np, tp1 * tp2)

        with ir.main_graph:
            inputs_data, inputs_host_steam, inputs_tensors = zip(
                *[
                    addons.host_load(logits_rep[0], popxl.float32, "x"),
                    addons.host_load(labels_rep[0], popxl.int32, "labels"),
                    addons.host_load(loss_bwd_rep[0], popxl.float32, "loss"),
                ]
            )
            logits, labels, loss_bwd = inputs_tensors

            facts_fwd, graph_fwd = CrossEntropyShardedLoss(test_config).create_graph(logits, labels)

            graph_bwd = popxl_addons.autodiff(graph_fwd, grads_required=graph_fwd.graph.inputs[:1])

            fwd_info = graph_fwd.bind(facts_fwd.init()).call_with_info(logits, labels)
            loss, *_ = fwd_info.outputs

            dx, *_ = graph_bwd.call(loss_bwd, args=graph_bwd.grad_graph_info.inputs_dict(fwd_info))

            loss_d2h = addons.host_store(loss)
            dx_d2h = addons.host_store(dx)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        inputs = dict(zip(inputs_host_steam, [logits_rep, labels_rep, loss_bwd_rep]))

        with popxl.Session(ir, "ipu_hw") as session:
            outputs = session.run(inputs)

        # Ensure loss is identical
        output_loss = outputs[loss_d2h][0]
        for i in range(tp1 * tp2):
            np.testing.assert_almost_equal(outputs[loss_d2h][i], output_loss)

        # Ensure dx tp2 shards are identical
        dx_data = outputs[dx_d2h]
        assert len(dx_data) == tp1 * tp2
        dx_unshard = [dx_data[i::tp2] for i in range(tp2)]
        dx_unshard = [unshard(x, 1) for x in dx_unshard]
        dx_unshard_0 = dx_unshard[0]
        assert len(dx_unshard) == tp2
        assert dx_unshard_0.shape == (batch_size * seq_len, vocab_size)
        for i in range(1, tp2):
            np.testing.assert_almost_equal(dx_unshard_0, dx_unshard[i])

        return output_loss, dx_unshard_0

    logits = np.random.random((batch_size * seq_len, vocab_size)).astype("float32")
    labels = np.random.randint(0, vocab_size, (batch_size * seq_len)).astype("int32")
    loss_bwd = np.random.random(()).astype("float32")

    labels[0] = 0
    labels[-1] = 0  # Test ignore index in offsetted shard

    loss_pt, dx_pt = PyTorch(logits, labels, loss_bwd)
    loss_xl, dx_xl = TP2D(logits, labels, loss_bwd)

    np.testing.assert_almost_equal(loss_xl, loss_pt, decimal=6)
    np.testing.assert_almost_equal(dx_xl, dx_pt, decimal=6)
