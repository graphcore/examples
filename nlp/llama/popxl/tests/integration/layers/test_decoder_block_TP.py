# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.llama.configuration_llama import LlamaConfig as HFConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import LlamaConfig
from modelling.decoder import LlamaDecoderBlockTP
from popxl_addons.array_munging import repeat


def test_decoder_block_TP_cmp_huggingface(test_config: LlamaConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = test_config.model.intermediate_size
    kv_heads = test_config.model.attention.kv_heads
    heads = test_config.model.attention.heads
    layers = test_config.model.layers

    # HuggingFace
    config = HFConfig(
        hidden_size=hidden_size,
        max_position_embeddings=seq_len,
        intermediate_size=intermediate_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_hidden_layers=layers,
        rms_norm_eps=test_config.model.eps,
    )

    hf_model = LlamaDecoderLayer(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)
    mask_t = torch.tensor(1e4 * (np.tril(np.ones((seq_len, seq_len))) - 1))[None, None, ...]
    (output_,) = hf_model(input_t, mask_t)

    output_HF = output_.detach().numpy()

    # TP
    n_shards = test_config.execution.tensor_parallel
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(
                    input_t.reshape(-1, test_config.model.hidden_size), test_config.model.dtype, name="input"
                ),
            ]
        )
        (x,) = inputs_tensors

        args, graph = LlamaDecoderBlockTP(test_config).create_graph(x)

        ff_vars = args.init()
        ff = graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = LlamaDecoderBlockTP.hf_mapping(test_config, ff_vars, hf_model)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    assert len(fwd_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])
    # Assert nearly equal to HF
    dps = 4 if test_config.model.dtype == popxl.float32 else 2
    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), dps)
