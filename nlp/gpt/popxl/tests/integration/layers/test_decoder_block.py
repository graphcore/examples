# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons

from config import GPTConfig
from modelling.decoder import GPTDecoderBlock


def test_decoder_block_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4

    # HuggingFace
    config = HFConfig(
        hidden_size=hidden_size,
        seq_len=seq_len,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads
    )
    hf_model = GPT2Block(config).eval()

    # HF forward
    input_t = torch.rand(
        (batch_size, seq_len, hidden_size), requires_grad=True)
    output_,  = hf_model(input_t)

    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    input_grad_HF = input_t.grad.detach().numpy()
    output_HF = output_.detach().numpy()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = test_config.execution.data_parallel

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(
                input_t.reshape(-1, test_config.model.hidden_size), popxl.float32, name="input"),
        ])
        input_t = inputs_tensors[0]
        args, decoder_block_graph = GPTDecoderBlock(
            test_config).create_graph(input_t)
        grad_ff_graph = addons.autodiff(decoder_block_graph)

        vars = args.init("decoder_block")
        layer = decoder_block_graph.bind(vars)
        call_info = layer.call_with_info(input_t)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)

        # Backwards
        gradient = popxl.constant(grad_wrt.reshape(
            act.shape).numpy().copy(), act.dtype, "gradient")
        grad_input, *_ = grad_ff_graph.call(
            gradient, args=grad_ff_graph.grad_graph_info.inputs_dict(call_info))

        grad_stream = addons.host_store(grad_input)

    # Map weights from huggingface
    weights = GPTDecoderBlock.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, inputs_data))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    np.testing.assert_almost_equal(
        output_HF, outs[act_stream].reshape(output_HF.shape), 3)
    np.testing.assert_almost_equal(
        input_grad_HF, outs[grad_stream].reshape(input_grad_HF.shape), 3)
