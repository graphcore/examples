# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.t5 import T5Config as HFConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.array_munging import repeat, shard
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons import TaskSession
from popxl_addons.named_tensors import NamedTensors

from config import T5Config
from modelling.embedding import T5EmbeddingsTP
from modelling.t5_lm import T5LMHeadModelTP
from modelling.hf_mapping import load_lm_to_hf


def test_lm_TP_cmp_huggingface(test_config: T5Config):
    torch.manual_seed(0)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv
    intermediate_size = test_config.model.d_ff
    vocab_size = test_config.model.embedding.vocab_size
    # Use the same eps as HF
    test_config.model.eps = 1e-6

    # HuggingFace
    config = HFConfig(
        num_layers=test_config.model.layers,
        vocab_size=vocab_size,
        d_model=hidden_size,
        seq_len=seq_len,
        num_heads=n_heads,
        d_kv=d_kv,
        feed_forward_proj="gated-gelu",
        d_ff=intermediate_size,
        tie_word_embeddings=False,
    )
    hf_model = T5ForConditionalGeneration(config).eval()

    input_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    dec_input_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    dec_mask_t = torch.randint(0, 2, (batch_size, seq_len))
    # HF forward
    outputs = hf_model(input_t, mask_t, dec_input_t, dec_mask_t)
    output_HF = outputs.logits
    # HF backward
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)
    words_grad_HF = hf_model.shared.weight.grad.detach().numpy()
    output_HF = output_HF.detach().numpy()

    # TP
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset inputs
    words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))
    dec_words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(dec_input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph
    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
                addons.host_load(dec_words_offsetted[0], popxl.int32, name="dec_words"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(dec_mask_t, popxl.float32, name="dec_mask"),
            ]
        )
        words, dec_words, mask, dec_mask = inputs_tensors
        facts, graph = T5LMHeadModelTP(test_config).create_graph(words, dec_words, mask, dec_mask)

        vars = facts.init()
        t5 = graph.bind(vars)
        call_info = t5.call_with_info(words, dec_words, mask, dec_mask)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)

        # The logits will be sharded over the vocab axis, so we need to initialise a sharded grad tensor
        word_shard_size = T5EmbeddingsTP.get_vocab_shard_sizes(test_config)
        word_pad = word_shard_size * n_shards - vocab_size
        grad_wrt = grad_wrt.reshape(-1, grad_wrt.shape[-1])
        grad_wrt = np.pad(to_numpy(grad_wrt, test_config.model.dtype), ((0, 0), (0, word_pad)))
        sharded_grads = shard(grad_wrt, n_shards, axis=-1)
        gradient = popxl.variable(sharded_grads, act.dtype, "gradient", replica_grouping=replica_grouping)

        # Backwards
        grad_graph = addons.autodiff(graph, grads_required=graph.args.tensors)

        grad_call_info = grad_graph.call_with_info(gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        tensor_to_grad_tensor = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(grad_call_info)
        words_grad_d2h = addons.host_store(tensor_to_grad_tensor[graph.args.transformer.embeddings.word.weight])

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = T5LMHeadModelTP.hf_mapping(test_config, vars, hf_model)

    # We need to tp repeat the data for masks (for the input ids
    # it has already been done as part of offset_inputs())
    inputs = {}
    inputs[inputs_host_steam[0]] = words_offsetted
    inputs[inputs_host_steam[1]] = dec_words_offsetted
    for h2d, data in zip(inputs_host_steam[2:], inputs_data[2:]):
        inputs[h2d] = repeat(data, n_shards)

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]
    # Grad outputs
    grad_data = outs[words_grad_d2h]

    assert len(fwd_data) == n_shards
    assert len(grad_data) == n_shards
    # Assemble the logits shards
    fwd_data_full = np.concatenate(fwd_data, axis=-1)[:, :vocab_size]
    # Assemble the weight grad shards
    weights_grad_popxl = np.concatenate(grad_data, axis=0)[:vocab_size, :]

    np.testing.assert_almost_equal(output_HF, fwd_data_full.reshape(output_HF.shape), 4)
    np.testing.assert_almost_equal(words_grad_HF, weights_grad_popxl, 2)


def test_lm_to_hf(test_config: T5Config):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv
    intermediate_size = test_config.model.d_ff
    vocab_size = test_config.model.embedding.vocab_size
    # Use the same eps as HF
    test_config.model.eps = 1e-6

    input_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    dec_input_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    dec_mask_t = torch.randint(0, 2, (batch_size, seq_len))

    # TP
    tp = 4
    test_config.execution.tensor_parallel = tp

    # Offset inputs
    words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))
    dec_words_offsetted = T5EmbeddingsTP.offset_inputs(test_config, to_numpy(dec_input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = tp
    main = ir.main_graph
    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
                addons.host_load(dec_words_offsetted[0], popxl.int32, name="dec_words"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(dec_mask_t, popxl.float32, name="dec_mask"),
            ]
        )
        words, dec_words, mask, dec_mask = inputs_tensors
        facts, graph = T5LMHeadModelTP(test_config).create_graph(words, dec_words, mask, dec_mask)

        vars = facts.init()
        t5 = graph.bind(vars)
        (out,) = t5.call(words, dec_words, mask, dec_mask)
        act_stream = addons.host_store(out)

    apply_pre_alias_patterns(ir, level="default")
    ir.num_host_transfers = test_config.execution.device_iterations

    # We need to tp repeat the data for masks (for the input ids
    # it has already been done as part of offset_inputs())
    inputs = {}
    inputs[inputs_host_steam[0]] = words_offsetted
    inputs[inputs_host_steam[1]] = dec_words_offsetted
    for h2d, data in zip(inputs_host_steam[2:], inputs_data[2:]):
        inputs[h2d] = repeat(data, tp)

    session = TaskSession(inputs, [act_stream], NamedTensors(fwd=vars), ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[act_stream]

    # HuggingFace
    config = HFConfig(
        num_layers=test_config.model.layers,
        vocab_size=vocab_size,
        d_model=hidden_size,
        seq_len=seq_len,
        num_heads=n_heads,
        d_kv=d_kv,
        feed_forward_proj="gated-gelu",
        d_ff=intermediate_size,
        tie_word_embeddings=False,
    )

    hf_model = T5ForConditionalGeneration(config).eval()
    hf_model = load_lm_to_hf(session, hf_model)
    outputs = hf_model(input_t, mask_t, dec_input_t, dec_mask_t)
    output_HF = outputs.logits.detach().numpy()

    # Assemble the outputs from the shards
    fwd_data_full = np.concatenate(out, axis=-1)

    np.testing.assert_almost_equal(output_HF.reshape(fwd_data_full.shape), fwd_data_full, 4)
