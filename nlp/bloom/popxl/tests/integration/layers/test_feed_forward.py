# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl_addons as addons
import torch
from popxl_addons.array_munging import repeat_shard, unshard
from popxl_addons.patterns import apply_pre_alias_patterns

# HF
from transformers.models.bloom import BloomConfig as HFConfig
from transformers.models.bloom.modeling_bloom import BloomBlock

import popxl
from config import BloomConfig
from modelling.feed_forward import BloomFeedForwardTP2D


@torch.no_grad()
def test_feed_forward(test_config: BloomConfig):
    torch.manual_seed(42)

    batch_size = 1
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # HuggingFace
    config = HFConfig(hidden_size=hidden_size)
    hf_model = BloomBlock(config).eval()

    # HF forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    z = hf_model.post_attention_layernorm(input_t)
    output_ = hf_model.mlp(z, residual=input_t)

    input_HF = input_t.reshape(-1, test_config.model.hidden_size).detach().numpy()
    output_HF = output_.reshape(-1, test_config.model.hidden_size).detach().numpy()

    input_HF = input_HF.astype(test_config.model.dtype.as_numpy())

    # popxl
    ir = popxl.Ir(replication=tp1 * tp2)

    input_HF_rs = repeat_shard(input_HF, tp1, tp2, 1)

    with ir.main_graph:
        _, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(input_HF_rs[0], test_config.model.dtype, name="x"),
            ]
        )
        (x,) = inputs_tensors

        ff_args, ff_graph = BloomFeedForwardTP2D(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = BloomFeedForwardTP2D.hf_mapping(test_config, ff_vars, hf_model)

    inputs = dict(zip(inputs_host_stream, [input_HF_rs]))

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    assert len(fwd_data) == tp1 * tp2

    # Ensure tp1 shards are identical
    fwd_data_splits = np.split(fwd_data, tp1)
    for t in fwd_data_splits[1:]:
        np.testing.assert_almost_equal(t, fwd_data_splits[0])

    fwd_data_np = unshard(fwd_data_splits[0], 1)
    # Assert nearly equal to HF
    np.testing.assert_almost_equal(output_HF, fwd_data_np, 6)
