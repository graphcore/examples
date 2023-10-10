# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.llama.configuration_llama import LlamaConfig as HFConfig
from transformers.models.llama.modeling_llama import LlamaMLP

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import LlamaConfig
from modelling.feed_forward import LlamaFeedForwardTP
from popxl_addons.array_munging import repeat


def test_feed_forward_TP_cmp_huggingface(test_config: LlamaConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    intermediate_size = test_config.model.intermediate_size

    # HuggingFace
    config = HFConfig(
        hidden_size=hidden_size,
        max_position_embeddings=seq_len,
        intermediate_size=intermediate_size,
        num_attention_heads=test_config.model.attention.heads,
    )
    hf_model = LlamaMLP(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size))
    outputs = hf_model(input_t)
    output_ = outputs.reshape(batch_size * seq_len, hidden_size)
    output_HF = output_.detach().numpy()

    # TP
    n_shards = test_config.execution.tensor_parallel

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

        ff_args, ff_graph = LlamaFeedForwardTP(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = LlamaFeedForwardTP.hf_mapping(test_config, ff_vars, hf_model)

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
    dps = 4 if test_config.model.dtype == popxl.float32 else 3
    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), dps)
