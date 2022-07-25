# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.embedding import GPTEmbeddingsTP
from modelling.gpt_lm import GPTLMHeadModelTP
from utils.utils import write_variables_pb, shard


def test_lm_TP_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)
    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4
    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=test_config.model.embedding.vocab_size,
        n_positions=test_config.model.embedding.max_positional_length,
        hidden_size=hidden_size,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
        tie_word_embeddings=True
    )
    hf_model = GPT2LMHeadModel(config).eval()

    # HF forward
    input_t = torch.randint(0, test_config.model.embedding.vocab_size,
                            (batch_size, test_config.model.sequence_length))
    output_HF = hf_model(input_ids=input_t)[0]
    # HF backward
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)

    words_grad_HF = hf_model.transformer.wte.weight.grad.detach().numpy()
    positions_grad_HF = hf_model.transformer.wpe.weight.grad.detach().numpy()
    output_HF = output_HF.detach().numpy()
    # n_shards
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset inputs
    words_offsetted, pos_offsetted = GPTEmbeddingsTP.offset_inputs(
        test_config, to_numpy(input_t))
    pos_offsetted = pos_offsetted.reshape(words_offsetted.shape)
    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[addons.host_load(words_offsetted[0],
                                                                                popxl.int32, name="words"), ])
        words, = inputs_tensors
        pos = popxl.variable(pos_offsetted, words.dtype,
                             name="positions", replica_grouping=replica_grouping)

        facts, graph = GPTLMHeadModelTP(test_config).create_graph(words, pos)
        vars = facts.init()
        gpt = graph.bind(vars)
        call_info = gpt.call_with_info(words, pos)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)
        word_shard_size, _ = GPTEmbeddingsTP.get_vocab_shard_sizes(test_config)
        word_pad = word_shard_size * n_shards - test_config.model.embedding.vocab_size
        grad_wrt = grad_wrt.reshape(-1, grad_wrt.shape[-1])
        grad_wrt = np.pad(
            to_numpy(grad_wrt, test_config.model.dtype), ((0, 0), (0, word_pad)))
        sharded_grads = shard(grad_wrt, n_shards, axis=-1)
        sharded_grads = sharded_grads
        gradient = popxl.variable(
            sharded_grads, act.dtype, "gradient", replica_grouping=replica_grouping)
        # Backwards
        grad_graph = addons.autodiff(graph, grads_required=graph.args.tensors)
        grad_call_info = grad_graph.call_with_info(
            gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        tensor_to_grad_tensor = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(
            grad_call_info)
        words_grad_d2h = addons.host_store(
            tensor_to_grad_tensor[graph.args.transformer.embeddings.word.weight])
        pos_grad_d2h = addons.host_store(
            tensor_to_grad_tensor[graph.args.transformer.embeddings.positional.weight])

    apply_pre_alias_patterns(ir, level='default')

    # Map weights from huggingface
    weights = GPTLMHeadModelTP.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        # TODO remove write_variables_pb once T56776 has landed
        # session.write_variables_data(weights)
        write_variables_pb(session, weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]
    assert len(fwd_data) == n_shards
    fwd_data_full = np.concatenate(
        fwd_data, axis=-1)[:, :test_config.model.embedding.vocab_size]
    np.testing.assert_almost_equal(
        output_HF, fwd_data_full.reshape(output_HF.shape), 3)

    # Grad outputs
    grads_data_words = outs[words_grad_d2h]
    grad_data_pos = outs[pos_grad_d2h]
    grad_data_words_full = np.concatenate(grads_data_words, axis=0)[
        :test_config.model.embedding.vocab_size, :]
    grad_data_pos_full = np.concatenate(grad_data_pos, axis=0)[
        :test_config.model.embedding.max_positional_length, :]

    np.testing.assert_almost_equal(words_grad_HF, grad_data_words_full, 3)
    np.testing.assert_almost_equal(positions_grad_HF, grad_data_pos_full, 3)
