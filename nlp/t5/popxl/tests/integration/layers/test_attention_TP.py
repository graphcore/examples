# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import torch

# HF
from transformers.models.t5 import T5Config as HFConfig
from transformers.models.t5.modeling_t5 import T5Attention

import popxl

import popxl_addons as addons
from popxl_addons.array_munging import repeat
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.task_session import TaskSession
from config import T5Config
from modelling.attention import T5SelfAttentionTP, T5CrossAttentionTP


@pytest.mark.parametrize("is_decoder", [False, True])
def test_attention_TP_cmp_huggingface(test_config: T5Config, is_decoder: bool):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv

    # HuggingFace
    config = HFConfig(d_model=hidden_size, seq_len=seq_len, num_heads=n_heads, d_kv=d_kv)
    config.is_decoder = is_decoder
    hf_model = T5Attention(config, True).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    is_encoder = 0 if is_decoder else 1
    is_encoder_t = torch.tensor(is_encoder, dtype=torch.float32)
    # Transform the attention mask like the HF attention expects it
    if is_decoder:
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        causal_mask = torch.from_numpy(causal_mask).to(mask_t.dtype)
        causal_mask = causal_mask[None, None, :, :]
        mask_hf = mask_t[:, None, None, :] * causal_mask
    else:
        mask_hf = mask_t[:, None, None, :]
    mask_hf = (mask_hf - 1) * 1e9
    output_, *_ = hf_model(input_t, mask_hf)
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
    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(is_encoder_t, popxl.float32, name="is_encoder"),
            ]
        )
        x, mask, is_encoder = inputs_tensors

        attn_args, attn_graph = T5SelfAttentionTP(test_config).create_graph(x, mask, is_encoder)

        vars = attn_args.init()
        fwd_info = attn_graph.bind(vars).call_with_info(x, mask, is_encoder)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        grad_ff_graph = addons.autodiff(attn_graph)

        gradient = popxl.constant(grad_wrt.reshape(acts.shape).numpy().copy(), acts.dtype, "gradient")
        grad_outputs, *_ = grad_ff_graph.call(gradient, args=grad_ff_graph.grad_graph_info.inputs_dict(fwd_info))

        grad_d2h = addons.host_store(grad_outputs)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = T5SelfAttentionTP.hf_mapping(test_config, vars, hf_model)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
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
    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), 5)
    np.testing.assert_almost_equal(input_grad_HF, grad_data[0].reshape(input_grad_HF.shape), 3)


@pytest.mark.parametrize("is_decoder", [False, True])
def test_attention_to_hf(test_config: T5Config, is_decoder: bool):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv

    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    is_encoder = 0 if is_decoder else 1
    is_encoder_t = torch.tensor(is_encoder, dtype=torch.float32)

    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(is_encoder_t, popxl.float32, name="is_encoder"),
            ]
        )
        x, mask, is_encoder = inputs_tensors

        attn_args, attn_graph = T5SelfAttentionTP(test_config).create_graph(x, mask, is_encoder)
        vars = attn_args.init()
        (out,) = attn_graph.bind(vars).call(x, mask, is_encoder)
        fwd_d2h = addons.host_store(out)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}
    session = TaskSession(inputs, [fwd_d2h], vars, ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[fwd_d2h]
        popxl_state = session.get_named_tensors_data()

    config = HFConfig(d_model=hidden_size, seq_len=seq_len, num_heads=n_heads, d_kv=d_kv)
    config.is_decoder = is_decoder
    hf_model = T5Attention(config, True).eval()
    state_dict = T5SelfAttentionTP.to_hf(config, popxl_state, hf_model)
    hf_model.load_state_dict(state_dict)
    # Transform the attention mask like the HF attention expects it
    if is_decoder:
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        causal_mask = torch.from_numpy(causal_mask).to(mask_t.dtype)
        causal_mask = causal_mask[None, None, :, :]
        mask_hf = mask_t[:, None, None, :] * causal_mask
    else:
        mask_hf = mask_t[:, None, None, :]
    mask_hf = (mask_hf - 1) * 1e9
    outputs, *_ = hf_model(input_t, mask_hf)
    output_HF = outputs.reshape(out[0].shape).detach().numpy()

    np.testing.assert_almost_equal(output_HF, out[0], 5)


def test_crossattention_TP_cmp_huggingface(test_config: T5Config):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv

    # HuggingFace
    config = HFConfig(d_model=hidden_size, seq_len=seq_len, num_heads=n_heads, d_kv=d_kv)
    config.is_decoder = True
    hf_model = T5Attention(config, False).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    enc_out_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    # Transform the attention mask like the HF attention expects it
    mask_hf = mask_t[:, None, None, :]
    mask_hf = (mask_hf - 1) * 1e9
    output_, *_ = hf_model(input_t, mask_hf, enc_out_t)
    # HF backwards
    grad_wrt = torch.rand(output_.shape)
    output_.backward(gradient=grad_wrt)
    input_grad_HF = input_t.grad.detach().numpy()
    enc_out_grad_HF = enc_out_t.grad.detach().numpy()
    output_HF = output_.detach().numpy()

    # TP
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(enc_out_t.reshape(-1, hidden_size), popxl.float32, name="enc_out"),
            ]
        )
        x, mask, enc_out = inputs_tensors

        attn_args, attn_graph = T5CrossAttentionTP(test_config).create_graph(x, mask, enc_out)

        vars = attn_args.init()
        fwd_info = attn_graph.bind(vars).call_with_info(x, mask, enc_out)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

        # Backwards
        grad_ff_graph = addons.autodiff(attn_graph)

        gradient = popxl.constant(grad_wrt.reshape(acts.shape).numpy().copy(), acts.dtype, "gradient")
        grad_outputs = grad_ff_graph.call(gradient, args=grad_ff_graph.grad_graph_info.inputs_dict(fwd_info))
        input_grad = grad_outputs[0]
        enc_out_grad = grad_outputs[2]

        grad_d2h = addons.host_store(input_grad)
        enc_grad_d2h = addons.host_store(enc_out_grad)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = T5CrossAttentionTP.hf_mapping(test_config, vars, hf_model)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]
    grad_data = outputs_popxl[grad_d2h]
    enc_grad_data = outputs_popxl[enc_grad_d2h]

    assert len(fwd_data) == n_shards
    assert len(grad_data) == n_shards
    assert len(enc_grad_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])
        np.testing.assert_equal(grad_data[0], grad_data[i])
        np.testing.assert_equal(enc_grad_data[0], enc_grad_data[i])
    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), 5)
    np.testing.assert_almost_equal(input_grad_HF, grad_data[0].reshape(input_grad_HF.shape), 3)
    np.testing.assert_almost_equal(enc_out_grad_HF, enc_grad_data[0].reshape(enc_out_grad_HF.shape), 3)


def test_crossattention_to_hf(test_config: T5Config):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    n_heads = test_config.model.attention.heads
    d_kv = test_config.model.attention.d_kv

    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    mask_t = torch.randint(0, 2, (batch_size, seq_len))
    enc_out_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)

    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, hidden_size), popxl.float32, name="input"),
                addons.host_load(mask_t, popxl.float32, name="mask"),
                addons.host_load(enc_out_t.reshape(-1, hidden_size), popxl.float32, name="enc_out"),
            ]
        )
        x, mask, enc_out = inputs_tensors

        attn_args, attn_graph = T5CrossAttentionTP(test_config).create_graph(x, mask, enc_out)
        vars = attn_args.init()
        (out,) = attn_graph.bind(vars).call(x, mask, enc_out)
        fwd_d2h = addons.host_store(out)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}
    session = TaskSession(inputs, [fwd_d2h], vars, ir=ir, device_desc="ipu_hw")

    with session:
        out = session.run(inputs)[fwd_d2h]
        popxl_state = session.get_named_tensors_data()

    config = HFConfig(d_model=hidden_size, seq_len=seq_len, num_heads=n_heads, d_kv=d_kv)
    config.is_decoder = True
    hf_model = T5Attention(config, False).eval()
    state_dict = T5CrossAttentionTP.to_hf(config, popxl_state, hf_model)
    hf_model.load_state_dict(state_dict)
    # Transform the attention mask like the HF attention expects it
    mask_hf = mask_t[:, None, None, :]
    mask_hf = (mask_hf - 1) * 1e9
    outputs, *_ = hf_model(input_t, mask_hf, enc_out_t)
    output_HF = outputs.reshape(out[0].shape).detach().numpy()

    np.testing.assert_almost_equal(output_HF, out[0], 5)
