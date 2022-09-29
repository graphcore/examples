# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP
from utils.utils import write_variables_pb, repeat


def test_decoder_block_TP_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4

    # HuggingFace
    config = HFConfig(hidden_size=hidden_size,
                      seq_len=seq_len,
                      n_inner=intermediate_size,
                      n_head=test_config.model.attention.heads)
    hf_model = GPT2Block(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size),
                         requires_grad=True)
    output_, = hf_model(input_t)

    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    input_grad_HF = input_t.grad.detach().numpy()
    output_HF = output_.detach().numpy()

    # TP
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards

    replica_grouping = ir.replica_grouping(stride=1, group_size=1)

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(input_t.reshape(-1,
                                             test_config.model.hidden_size),
                             popxl.float32,
                             name="input"),
        ])
        x, = inputs_tensors

        ff_args, ff_graph = GPTDecoderBlockTP(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        acts, = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        grad_ff_graph = addons.autodiff(ff_graph)

        gradient = popxl.constant(
            grad_wrt.reshape(acts.shape).numpy().copy(), acts.dtype,
            "gradient")
        grad_outputs, *_ = grad_ff_graph.call(
            gradient, args=grad_ff_graph.grad_graph_info.inputs_dict(fwd_info))

        grad_d2h = addons.host_store(grad_outputs)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    weights = GPTDecoderBlockTP.hf_mapping(test_config, ff_vars, hf_model)

    inputs = {
        h2d: repeat(data, n_shards)
        for h2d, data in zip(inputs_host_steam, inputs_data)
    }

    with popxl.Session(ir, "ipu_hw") as session:
        # TODO remove write_variables_pb once T56776 has landed
        # session.write_variables_data(weights)
        write_variables_pb(session, weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]
    grad_data = outputs_popxl[grad_d2h]

    assert len(fwd_data) == n_shards
    assert len(grad_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])
        np.testing.assert_equal(grad_data[0], grad_data[i])
    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF,
                                   fwd_data[0].reshape(output_HF.shape), 3)
    np.testing.assert_almost_equal(input_grad_HF,
                                   grad_data[0].reshape(input_grad_HF.shape),
                                   3)
