# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.t5 import T5Config as HFConfig
from transformers.models.t5.modeling_t5 import T5LayerNorm as T5LayerNormHF

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.task_session import TaskSession

from config import T5Config
from modelling.layer_norm import T5LayerNorm


def test_layer_norm_cmp_huggingface(test_config: T5Config):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    # Use the same eps as HF
    test_config.model.eps = 1e-6

    # HuggingFace
    hf_model = T5LayerNormHF(hidden_size).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    outputs = hf_model(input_t)
    output_ = outputs.reshape(batch_size * seq_len, hidden_size)
    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    input_grad_HF = input_t.grad.detach().numpy()
    output_HF = output_.detach().numpy()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = 1

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
            ]
        )
        (x,) = inputs_tensors

        ln_args, ln_graph = T5LayerNorm(test_config).create_graph(x)

        ln_vars = ln_args.init()
        ff = ln_graph.bind(ln_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        grad_ln_graph = addons.autodiff(ln_graph)

        gradient = popxl.constant(grad_wrt.reshape(acts.shape).numpy().copy(), acts.dtype, "gradient")
        grad_outputs, *_ = grad_ln_graph.call(gradient, args=grad_ln_graph.grad_graph_info.inputs_dict(fwd_info))

        grad_d2h = addons.host_store(grad_outputs)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = T5LayerNorm.hf_mapping(test_config, ln_vars, hf_model)

    inputs = {h2d: data for h2d, data in zip(inputs_host_stream, inputs_data)}
    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]
    grad_data = outputs_popxl[grad_d2h]

    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF, fwd_data.reshape(output_HF.shape), 5)
    np.testing.assert_almost_equal(input_grad_HF, grad_data.reshape(input_grad_HF.shape), 3)


def test_layer_norm_to_hf(test_config: T5Config):
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    # Use the same eps as HF
    test_config.model.eps = 1e-6

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = 1

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
            ]
        )
        (x,) = inputs_tensors

        ln_args, ln_graph = T5LayerNorm(test_config).create_graph(x)

        ln_vars = ln_args.init()
        (out,) = ln_graph.bind(ln_vars).call(x)
        fwd_d2h = addons.host_store(out)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    inputs = {h2d: data for h2d, data in zip(inputs_host_stream, inputs_data)}
    session = TaskSession(inputs, [fwd_d2h], ln_vars, ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[fwd_d2h]
        popxl_state = session.get_named_tensors_data()

    config = HFConfig(d_model=hidden_size, seq_len=seq_len)
    hf_model = T5LayerNormHF(hidden_size).eval()

    state_dict = T5LayerNorm.to_hf(config, popxl_state, hf_model)
    hf_model.load_state_dict(state_dict)
    outputs = hf_model(input_t)
    output_HF = outputs.reshape(batch_size * seq_len, hidden_size).detach().numpy()

    np.testing.assert_almost_equal(output_HF.reshape(out.shape), out, 5)
