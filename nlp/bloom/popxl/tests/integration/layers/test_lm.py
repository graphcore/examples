# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl_addons as addons
import torch
from popxl_addons.array_munging import repeat, unshard_arrays
from popxl_addons.patterns import apply_pre_alias_patterns

# HF
from transformers.models.bloom import BloomConfig as HFConfig
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

import popxl
from config import BloomConfig
from modelling.bloom_lm import BloomLMHeadModelTP2D


@torch.no_grad()
def test_lm(test_config: BloomConfig):
    torch.manual_seed(42)

    batch_size = 1
    hidden_size = test_config.model.hidden_size
    seq_len = test_config.model.sequence_length
    vocab_size = test_config.model.embedding.vocab_size
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_head=test_config.model.attention.heads,
    )
    hf_model = BloomForCausalLM(config).eval()

    # HF forward
    words_t = torch.randint(0, vocab_size, (1, seq_len))
    words_t = words_t.repeat(batch_size, 1)

    output_HF = hf_model(input_ids=words_t)[0]

    words_np = repeat(words_t.detach().numpy(), tp1 * tp2)

    # popxl
    ir = popxl.Ir(replication=tp1 * tp2)

    with ir.main_graph:
        _, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(words_np[0], popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors

        facts, graph = BloomLMHeadModelTP2D(test_config).create_graph(words)

        ff_vars = facts.init()
        bloom = graph.bind(ff_vars)
        call_info = bloom.call_with_info(words)
        act, *_ = call_info.outputs

        act_stream = addons.host_store(act)

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = BloomLMHeadModelTP2D.hf_mapping(test_config, ff_vars, hf_model)

    inputs = dict(zip(inputs_host_stream, [words_np]))

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    # Ensure tp2 shards are identical
    fwd_data = outs[act_stream]
    assert len(fwd_data) == tp1 * tp2
    y_np_splits = np.split(fwd_data, tp1 * tp2)
    fwd_unshard_0 = unshard_arrays(y_np_splits[0::tp1], 1)
    for i in range(1, tp2):
        fwd_unshard_i = unshard_arrays(y_np_splits[i::tp1], 1)
        np.testing.assert_almost_equal(fwd_unshard_0, fwd_unshard_i)

    # fwd_data_np = unshard(y_np_splits[0], 1)
    fwd_data_np = np.concatenate(y_np_splits[::tp1], axis=-1)
    np.testing.assert_almost_equal(output_HF.reshape(-1), fwd_data_np.reshape(-1), 5)
