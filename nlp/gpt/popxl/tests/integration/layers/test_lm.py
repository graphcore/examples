# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from typing import List

# HF
from transformers import GPT2Tokenizer
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons

from config import GPTConfig
from modelling.gpt_lm import GPTLMHeadModel



def test_lm_cmp_huggingface(test_config: GPTConfig):
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
    output_ = hf_model(input_ids=input_t)[0]
    # HF backward
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    words_grad_HF = hf_model.transformer.wte.weight.grad.detach().numpy()
    positions_grad_HF = hf_model.transformer.wpe.weight.grad.detach().numpy()
    output_HF = output_.detach().numpy()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = test_config.execution.data_parallel

    main = ir.main_graph
    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[addons.host_load(input_t, popxl.int32, name="words"), ])
        input_t = inputs_tensors[0]
        args, graph = GPTLMHeadModel(
            test_config).create_graph(input_ids=input_t)

        vars = args.init()
        call_info = graph.bind(vars).call_with_info(input_t)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)

        # Backwards
        grads_req = graph.args.named_tensors.copy()
        grad_graph = addons.autodiff(graph, grads_required=graph.args.tensors)

        gradient = popxl.constant(grad_wrt.reshape(act.shape).numpy(
        ).copy(), act.dtype, "gradient")
        grad_call_info = grad_graph.call_with_info(
            gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        grad_map = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(
            grad_call_info)

        words_grad_d2h = addons.host_store(
            grad_map[graph.args.transformer.embeddings.word.weight])
        pos_grad_d2h = addons.host_store(
            grad_map[graph.args.transformer.embeddings.positional.weight])

    # Map weights from huggingface
    weights = GPTLMHeadModel.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, inputs_data))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    np.testing.assert_almost_equal(
        output_HF, outs[act_stream].reshape(output_HF.shape), 3)

    # Grad outputs
    grads_data_words = outs[words_grad_d2h][:test_config.model.embedding.vocab_size, :]
    grad_data_pos = outs[pos_grad_d2h][:test_config.model.embedding.max_positional_length, :]

    np.testing.assert_almost_equal(words_grad_HF, grads_data_words, 3)
    np.testing.assert_almost_equal(positions_grad_HF, grad_data_pos, 3)
