# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as HFGPT2LMHeadModel

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.embedding import GPTEmbeddingsTP2D, generate_positions, EmbeddingTP2D
from modelling.gpt_lm import GPTLMHeadModelTP2D
from popxl_addons.array_munging import pad_axis, repeat, unshard_arrays, unshard2D, repeat_shard


def test_lm_TP2D_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4
    seq_len = test_config.model.sequence_length
    max_pos_len = test_config.model.embedding.max_positional_length
    vocab_size = test_config.model.embedding.vocab_size
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=vocab_size,
        n_positions=max_pos_len,
        hidden_size=hidden_size,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
        tie_word_embeddings=True,
    )
    hf_model = HFGPT2LMHeadModel(config).eval()

    # HF forward
    words_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    output_HF = hf_model(input_ids=words_t)[0]

    # HF backward
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)

    word_size_padded = EmbeddingTP2D.get_padded_size(vocab_size, tp1)
    pos_size_padded = EmbeddingTP2D.get_padded_size(max_pos_len, tp1)

    words_np = repeat(words_t.detach().numpy(), tp1 * tp2)

    grad_np = grad_wrt.reshape(-1, grad_wrt.shape[-1]).detach().numpy()
    grad_np = pad_axis(grad_np, word_size_padded, 1)
    grad_np = repeat_shard(grad_np, tp1, tp2, 1, sharded_tensor="strided")

    words_grad_HF = pad_axis(hf_model.transformer.wte.weight.grad.detach().numpy(), word_size_padded, 0)
    positions_grad_HF = pad_axis(hf_model.transformer.wpe.weight.grad.detach().numpy(), pos_size_padded, 0)

    # popxl
    ir = popxl.Ir(replication=tp1 * tp2)

    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_np[0], popxl.int32, name="words"),
                addons.host_load(grad_np[0], popxl.float32, name="grad"),
            ]
        )
        words, grad = inputs_tensors
        pos = popxl.constant(generate_positions(test_config), popxl.int32, name="pos")

        facts, graph = GPTLMHeadModelTP2D(test_config).create_graph(words, pos)

        vars = facts.init()
        gpt = graph.bind(vars)
        call_info = gpt.call_with_info(words, pos)
        act, *_ = call_info.outputs

        act_stream = addons.host_store(act)

        # Backwards
        emb_vars = graph.args.transformer.embeddings
        grad_graph = addons.autodiff(graph, grads_required=[emb_vars.word.weight, emb_vars.positional.weight])

        word_grad, pos_grad, *_ = grad_graph.call(grad, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        words_grad_d2h = addons.host_store(word_grad)
        pos_grad_d2h = addons.host_store(pos_grad)

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = GPTLMHeadModelTP2D.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [words_np, grad_np]))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    # Ensure tp2 shards are identical
    fwd_data = outs[act_stream]
    assert len(fwd_data) == tp1 * tp2
    y_np_splits = np.split(fwd_data, tp1 * tp2)
    fwd_unshard_0 = unshard_arrays(y_np_splits[0::tp1], 1)
    for i in range(1, tp2):
        fwd_unshard_i = unshard_arrays(y_np_splits[i::tp1], 1)
        np.testing.assert_almost_equal(fwd_unshard_0, fwd_unshard_i)

    # Grad outputs
    grad_data_words_full = unshard2D(outs[words_grad_d2h], tp1, tp2, 0, 1)
    grad_data_pos_full = unshard2D(outs[pos_grad_d2h], tp1, tp2, 0, 1)

    np.testing.assert_almost_equal(words_grad_HF, grad_data_words_full, 3)
    np.testing.assert_almost_equal(positions_grad_HF, grad_data_pos_full, 3)
