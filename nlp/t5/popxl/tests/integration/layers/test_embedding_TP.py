# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from torch import nn

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.array_munging import shard
from popxl_addons.layers import Embedding
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.task_session import TaskSession

from config import T5Config
from modelling.embedding import T5EmbeddingsTP


def test_embeddings_TP_cmp_torch(test_config: T5Config):
    torch.manual_seed(42)
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    vocab_size = test_config.model.embedding.vocab_size
    hidden_size = test_config.model.hidden_size

    torch_model = nn.Embedding(vocab_size, hidden_size).eval()

    # torch forward
    input_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    outputs = torch_model(input_t)
    output_ = outputs.reshape(batch_size * seq_len, hidden_size)
    # torch backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    weights_grad_torch = torch_model.weight.grad.detach().numpy()
    output_torch = output_.detach().numpy()

    # TP
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset input
    words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards

    main = ir.main_graph
    with main:
        inputs_data, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (x,) = inputs_tensors

        emb_args, emb_graph = T5EmbeddingsTP(test_config).create_graph(x)

        emb_vars = emb_args.init()
        ff = emb_graph.bind(emb_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        # Exclude the rel_pos_weight from autodiff
        grads_required = [t for t in emb_graph.args.tensors if "rel_pos_weight" not in t.name]
        grad_emb_graph = addons.autodiff(emb_graph, grads_required=grads_required)
        gradient = popxl.constant(grad_wrt.reshape(acts.shape).numpy().copy(), acts.dtype, "gradient")
        grad_call_info = grad_emb_graph.call_with_info(
            gradient, args=grad_emb_graph.grad_graph_info.inputs_dict(fwd_info)
        )

        grad_map = grad_emb_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(grad_call_info)

        words_grad_stream = addons.host_store(grad_map[emb_graph.args.word.weight])

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    # Load torch weights into the popxl model
    dtype = popxl.float32
    word_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
    word_pad = word_shard_size * n_shards - vocab_size
    # Pad only first axis in one direction
    def pad(x, n_pad):
        return np.pad(x, ((0, n_pad), (0, 0)))

    weights = {
        emb_vars.word.weight: shard(pad(to_numpy(torch_model.weight.data, dtype), word_pad), n_shards, axis=0),
    }

    inputs = dict(zip(inputs_host_stream, [words_offsetted]))
    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]
    grad_data = outputs_popxl[words_grad_stream]

    assert len(fwd_data) == n_shards
    assert len(grad_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])
        # Note that the grad wrt to the weights will be different on each shard
        # (intended, as each IPU will have its own shard of the full weight)
    # Assemble the weight grad shards
    weights_grad_popxl = np.concatenate(grad_data, axis=0)[:vocab_size, :]
    # Assert nearly equal to torch
    np.testing.assert_almost_equal(output_torch, fwd_data[0].reshape(output_torch.shape), 5)
    np.testing.assert_almost_equal(weights_grad_torch, weights_grad_popxl.reshape(weights_grad_torch.shape), 3)


def test_embeddings_to_torch(test_config: T5Config):
    # TP
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    vocab_size = test_config.model.embedding.vocab_size
    hidden_size = test_config.model.hidden_size

    input_t = torch.randint(0, vocab_size, (batch_size, seq_len))

    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset input
    words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards

    main = ir.main_graph
    with main:
        inputs_data, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (x,) = inputs_tensors

        emb_args, emb_graph = T5EmbeddingsTP(test_config).create_graph(x)

        emb_vars = emb_args.init()
        (out,) = emb_graph.bind(emb_vars).call(x)
        fwd_d2h = addons.host_store(out)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    inputs = dict(zip(inputs_host_stream, [words_offsetted]))
    session = TaskSession(inputs, [fwd_d2h], emb_vars, ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[fwd_d2h]
        popxl_state = session.get_named_tensors_data()

    torch_model = nn.Embedding(vocab_size, hidden_size).eval()
    state_dict = {
        "weight": torch.tensor(np.concatenate(popxl_state.word.weight, axis=0)[:vocab_size], dtype=torch.float32)
    }
    torch_model.load_state_dict(state_dict)
    outputs = torch_model(input_t)
    output_torch = outputs.reshape(out[0].shape).detach().numpy()

    np.testing.assert_almost_equal(output_torch, out[0], 5)
