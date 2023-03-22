# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# finetuneanon
from transformers.models.gpt_neo import GPTNeoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons import TaskSession

from configs import MagmaConfig
from modelling import GPTJDecoderBlockTP
from popxl_addons.array_munging import repeat

from magma import Magma
import os
from configs import CONFIG_DIR
import pytest


def test_decoder_block_TP_cmp_finetuneanon(test_config: MagmaConfig):
    torch.manual_seed(42)
    test_config = test_config.transformer

    batch_size = test_config.execution.micro_batch_size
    seq_len = test_config.sequence_length
    hidden_size = test_config.hidden_size
    intermediate_size = hidden_size * 4

    # finetuneanon
    config = GPTNeoConfig(
        hidden_size=hidden_size,
        attention_layers=["global"] * test_config.layers,
        attention_types=[[["global"], test_config.layers]],
        num_layers=test_config.layers,
        max_position_embeddings=seq_len,
        intermediate_size=intermediate_size,
        num_heads=test_config.attention.heads,
        rotary=True,
        rotary_dim=test_config.attention.rotary_dim,
        jax=True,
    )
    torch_model = GPTNeoBlock(config, 0).eval()

    # finetuneanon forward
    input_t = torch.rand((batch_size, seq_len, hidden_size), requires_grad=False)
    (output_,) = torch_model(input_t)
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

        args, graph = GPTJDecoderBlockTP(test_config).create_graph(x)

        ff_vars = args.init()
        ff = graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = GPTJDecoderBlockTP.finetuneanon_mapping(test_config, ff_vars, torch_model, from_magma=False)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    assert len(fwd_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])

    # Assert nearly equal to finetuneanon
    np.testing.assert_almost_equal(output_torch, fwd_data[0].reshape(output_torch.shape), 3)


@pytest.mark.long_test
def test_decoder_block_adapters_cmp_magma(test_config: MagmaConfig):
    torch.manual_seed(42)
    # Load pretrained magma
    magma = Magma(config=os.path.join(CONFIG_DIR, "MAGMA_v1.yml"))
    magma = magma.eval()
    magma.float()

    test_config = test_config.transformer
    test_config.ff_adapter.downsample_factor = 4
    test_config.ff_adapter.mode = "normal"
    test_config.attention.heads = magma.lm.config.num_heads
    batch_size = test_config.execution.micro_batch_size
    test_config.hidden_size = magma.lm.config.hidden_size
    test_config.attention.rotary_dim = magma.lm.config.rotary_dim

    # get first block
    magma_decoder = magma.transformer[0]

    # finetuneanon forward
    input_t = torch.rand((batch_size, test_config.sequence_length, test_config.hidden_size), requires_grad=False)

    (output_,) = magma_decoder(input_t, use_cache=False)
    output_magma = output_.detach().numpy()
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

        args, graph = GPTJDecoderBlockTP(test_config).create_graph(x)

        ff_vars = args.init()
        ff = graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = GPTJDecoderBlockTP.finetuneanon_mapping(test_config, ff_vars, magma_decoder, from_magma=True)

    inputs = {h2d: repeat(data, n_shards) for h2d, data in zip(inputs_host_steam, inputs_data)}

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    fwd_data = outputs_popxl[fwd_d2h]

    assert len(fwd_data) == n_shards

    # Assert all IPU outputs are identical
    for i in range(1, n_shards):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])

    # Assert nearly equal to finetuneanon
    np.testing.assert_almost_equal(output_magma, fwd_data[0].reshape(output_magma.shape), 3)


if __name__ == "__main__":
    import os
    from utils.simple_parsing_tools import parse_args_with_config_file

    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_config.yml")
    test_decoder_block_adapters_cmp_magma(parse_args_with_config_file(MagmaConfig, ["--config", config_file]))
