# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gptj import GPTJConfig as HFConfig
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons import TaskSession
from popxl_addons.named_tensors import NamedTensors

from config import GPTJConfig
from modelling.embedding import GPTJEmbeddingsTP
from modelling.gptj_lm import GPTJLMHeadModelTP
from utils.utils import shard
from modelling.hf_mapping import load_lm_to_hf


def test_lm_TP_cmp_huggingface(test_config: GPTJConfig):
    torch.manual_seed(42)
    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4
    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=test_config.model.embedding.vocab_size,
        n_positions=test_config.model.sequence_length,
        hidden_size=hidden_size,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
        rotary_dim=test_config.model.attention.rotary_dim,
        tie_word_embeddings=False,
    )
    hf_model = GPTJForCausalLM(config).eval()

    # HF forward
    input_t = torch.randint(0, test_config.model.embedding.vocab_size, (batch_size, test_config.model.sequence_length))
    output_HF = hf_model(input_ids=input_t)[0]
    # HF backward
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)

    words_grad_HF = hf_model.transformer.wte.weight.grad.detach().numpy()
    output_HF = output_HF.detach().numpy()
    # n_shards
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset inputs
    words_offsetted = GPTJEmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))
    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors
        facts, graph = GPTJLMHeadModelTP(test_config).create_graph(words)
        vars = facts.init()
        gpt = graph.bind(vars)
        call_info = gpt.call_with_info(words)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)
        word_shard_size = GPTJEmbeddingsTP.get_vocab_shard_sizes(test_config)
        word_pad = word_shard_size * n_shards - test_config.model.embedding.vocab_size
        grad_wrt = grad_wrt.reshape(-1, grad_wrt.shape[-1])
        grad_wrt = np.pad(to_numpy(grad_wrt, test_config.model.dtype), ((0, 0), (0, word_pad)))
        sharded_grads = shard(grad_wrt, n_shards, axis=-1)
        sharded_grads = sharded_grads
        gradient = popxl.variable(sharded_grads, act.dtype, "gradient", replica_grouping=replica_grouping)
        # Backwards
        grad_graph = addons.autodiff(graph, grads_required=graph.args.tensors)
        grad_call_info = grad_graph.call_with_info(gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        tensor_to_grad_tensor = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(grad_call_info)
        words_grad_d2h = addons.host_store(tensor_to_grad_tensor[graph.args.transformer.embeddings.word.weight])

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = GPTJLMHeadModelTP.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]
    assert len(fwd_data) == n_shards
    fwd_data_full = np.concatenate(fwd_data, axis=-1)[:, : test_config.model.embedding.vocab_size]
    np.testing.assert_almost_equal(output_HF, fwd_data_full.reshape(output_HF.shape), 3)

    # Grad outputs
    grads_data_words = outs[words_grad_d2h]
    grad_data_words_full = np.concatenate(grads_data_words, axis=0)[: test_config.model.embedding.vocab_size, :]

    np.testing.assert_almost_equal(words_grad_HF, grad_data_words_full, 3)


def test_lm_to_hf(test_config: GPTJConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4

    input_t = torch.randint(0, test_config.model.embedding.vocab_size, (batch_size, test_config.model.sequence_length))

    # TP
    tp = 4
    test_config.execution.tensor_parallel = tp

    # Offset inputs
    words_offsetted = GPTJEmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = tp
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors
        facts, graph = GPTJLMHeadModelTP(test_config).create_graph(words)

        vars = facts.init()
        gpt = graph.bind(vars)
        (out,) = gpt.call(words)
        act_stream = addons.host_store(out)

    apply_pre_alias_patterns(ir, level="default")
    ir.num_host_transfers = test_config.execution.device_iterations

    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    session = TaskSession(inputs, [act_stream], NamedTensors(fwd=vars), ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[act_stream]
        popxl_state = session.get_named_tensors_data()

    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=test_config.model.embedding.vocab_size,
        n_positions=test_config.model.sequence_length,
        hidden_size=hidden_size,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
        rotary_dim=test_config.model.attention.rotary_dim,
    )

    hf_model = GPTJForCausalLM(config).eval()
    hf_model = load_lm_to_hf(session, hf_model)
    output_HF = hf_model(input_ids=input_t)[0].detach().numpy()
    fwd_data_full = np.concatenate(out, axis=-1)

    np.testing.assert_almost_equal(output_HF.reshape(fwd_data_full.shape), fwd_data_full, 3)
