# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# finetuneanon
from transformers.models.gpt_neo import GPTNeoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP

import popxl

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.task_session import TaskSession

from configs import MagmaConfig
from modelling import GPTJFeedForwardTP
from popxl_addons.array_munging import repeat


def test_feed_forward_TP_cmp_finetuneanon(test_config: MagmaConfig):
    torch.manual_seed(42)
    test_config = test_config.transformer
    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.sequence_length
    hidden_size = test_config.hidden_size
    intermediate_size = hidden_size * 4

    # finetuneanon
    config = GPTNeoConfig(
        hidden_size=hidden_size,
        max_position_embeddings=seq_len,
        intermediate_size=intermediate_size,
        num_heads=test_config.attention.heads,
    )
    torch_model = GPTNeoMLP(intermediate_size, config).eval()

    # torch forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    outputs = torch_model(input_t)
    output_ = outputs.reshape(batch_size * seq_len, hidden_size)
    output_torch = output_.detach().numpy()

    # TP
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t.reshape(-1, test_config.hidden_size), popxl.float32, name="input"),
            ]
        )
        (x,) = inputs_tensors

        ff_args, ff_graph = GPTJFeedForwardTP(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = GPTJFeedForwardTP.finetuneanon_mapping(test_config, ff_vars, torch_model)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}
    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    assert len(fwd_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])

    # Assert nearly equal to torch
    np.testing.assert_almost_equal(output_torch, fwd_data[0].reshape(output_torch.shape), 3)
