# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import torch
from transformers.models.bert import BertConfig as HFConfig
from transformers.models.bert.modeling_bert import BertAttention

import popxl_addons as addons
from config import BertConfig
from modelling.attention import SelfAttention


def test_attention_cmp_huggingface(test_config: BertConfig):
    torch.manual_seed(42)

    config = HFConfig(
        hidden_size=test_config.model.hidden_size,
        seq_len=test_config.model.sequence_length,
        num_attention_heads=test_config.model.attention.heads,
    )
    module = BertAttention(config).eval()

    # HuggingFace
    micro_bs = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = config.hidden_size

    # HF forward
    input_t = torch.rand((micro_bs, seq_len, hidden_size), requires_grad=True)
    mask = torch.randint(0, 2, (micro_bs, 1, 1, seq_len))
    mask_ = (mask - 1) * 1000.0
    output_ = module(input_t, mask_)[0]

    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    output_HF = output_.detach().numpy()
    assert input_t.grad is not None
    input_grad_HF = input_t.grad.detach().numpy()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = test_config.execution.data_parallel

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, config.hidden_size), popxl.float32, name="act"),
                addons.host_load(mask, popxl.float32, name="mask"),
            ]
        )
        act, mask = inputs_tensors
        args, attn_graph = SelfAttention(test_config).create_graph(act, mask)
        grad_attn_graph = addons.autodiff(attn_graph)

        attn = args.init("attention")
        layer = attn_graph.bind(attn)

        call_info = layer.call_with_info(act, mask)
        act, *_ = call_info.outputs
        output = addons.host_store(act)

        gradient = popxl.constant(grad_wrt.numpy().reshape(act.shape).copy(), act.dtype, "gradient")
        # type: ignore
        grad_input, *_ = grad_attn_graph.call(gradient, args=grad_attn_graph.grad_graph_info.inputs_dict(call_info))

        grad_stream = addons.host_store(grad_input)

    weights = SelfAttention.hf_mapping(test_config, attn, module)

    inputs = dict(zip(inputs_host_steam, inputs_data))
    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    np.testing.assert_almost_equal(output_HF, outs[output].reshape(output_HF.shape), 4)
    np.testing.assert_almost_equal(input_grad_HF, outs[grad_stream].reshape(input_grad_HF.shape), 4)
