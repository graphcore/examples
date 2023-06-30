# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt_neox import GPTNeoXConfig as HFConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from config import DollyConfig
from modelling.attention import DollySelfAttentionTP
from popxl_addons.array_munging import repeat


def test_attention_TP_cmp_huggingface(test_config: DollyConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4

    # HuggingFace
    config = HFConfig(
        hidden_size=hidden_size,
        max_position_embeddings=seq_len,
        intermediate_size=intermediate_size,
        num_attention_heads=test_config.model.attention.heads,
        rotary_dim=test_config.model.attention.rotary_dim,
    )
    hf_model = GPTNeoXAttention(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    output_, *_ = hf_model(input_t, None)
    output_HF = output_.detach().numpy()

    # TP
    n_shards = test_config.execution.tensor_parallel

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    with ir.main_graph:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(
                    input_t.reshape(-1, test_config.model.hidden_size), test_config.model.dtype, name="input"
                ),
            ]
        )
        (x,) = inputs_tensors

        attn_args, attn_graph = DollySelfAttentionTP(test_config).create_graph(x)

        vars = attn_args.init()
        fwd_info = attn_graph.bind(vars).call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = DollySelfAttentionTP.hf_mapping(test_config, vars, hf_model)

    inputs = {h2d: repeat(data, n_shards).squeeze() for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    if n_shards > 1:
        assert len(fwd_data) == n_shards

        # Assert all IPU outputs are identical
        for i in range(1, n_shards):
            np.testing.assert_equal(fwd_data[0], fwd_data[i])
    else:
        fwd_data = np.expand_dims(fwd_data, axis=0)

    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), 4)
