# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP2D
from popxl_addons.array_munging import repeat_shard, unshard


def test_decoder_block_TP2D_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # HuggingFace
    config = HFConfig(
        hidden_size=hidden_size, seq_len=seq_len, n_inner=intermediate_size, n_head=test_config.model.attention.heads
    )
    hf_model = GPT2Block(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    (output_,) = hf_model(input_t)

    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    input_HF = input_t.reshape(-1, test_config.model.hidden_size).detach().numpy()
    output_HF = output_.reshape(-1, test_config.model.hidden_size).detach().numpy()
    grad_HF = grad_wrt.reshape(-1, test_config.model.hidden_size).detach().numpy()
    grad_output_HF = input_t.grad.reshape(-1, test_config.model.hidden_size).detach().numpy()

    # popxl
    ir = popxl.Ir(replication=tp1 * tp2)

    input_HF_rs = repeat_shard(input_HF, tp1, tp2, 1)
    grad_HF_rs = repeat_shard(grad_HF, tp1, tp2, 1)

    with ir.main_graph:
        _, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_HF_rs[0], popxl.float32, name="x"),
                addons.host_load(grad_HF_rs[0], popxl.float32, name="grad"),
            ]
        )
        x, grad = inputs_tensors

        ff_args, ff_graph = GPTDecoderBlockTP2D(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        grad_ff_graph = addons.autodiff(ff_graph)

        grad_outputs, *_ = grad_ff_graph.call(grad, args=grad_ff_graph.grad_graph_info.inputs_dict(fwd_info))

        grad_d2h = addons.host_store(grad_outputs)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = GPTDecoderBlockTP2D.hf_mapping(test_config, ff_vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [input_HF_rs, grad_HF_rs]))

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]
    grad_data = outputs_popxl[grad_d2h]

    assert len(fwd_data) == tp1 * tp2
    assert len(grad_data) == tp1 * tp2

    # Ensure tp1 shards are identical
    fwd_data_splits = np.split(fwd_data, tp1)
    for t in fwd_data_splits[1:]:
        np.testing.assert_almost_equal(t, fwd_data_splits[0])

    grad_data_splits = np.split(grad_data, tp1)
    for t in grad_data_splits[1:]:
        np.testing.assert_almost_equal(t, grad_data_splits[0])

    fwd_data_np = unshard(fwd_data_splits[0], 1)
    grad_data_np = unshard(grad_data_splits[0], 1)

    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF, fwd_data_np, 6)
    np.testing.assert_almost_equal(grad_output_HF, grad_data_np, 3)
