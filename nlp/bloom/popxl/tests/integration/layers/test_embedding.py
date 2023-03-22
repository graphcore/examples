# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl_addons as addons
import torch
from popxl_addons.array_munging import repeat, unshard
from popxl_addons.patterns import apply_pre_alias_patterns

# HF
from transformers.models.bloom import BloomConfig as HFConfig
from transformers.models.bloom import BloomModel as HFModel

import popxl
from config import BloomConfig
from modelling.embedding import BloomEmbeddingTP2D


@torch.no_grad()
def test_embedding(test_config: BloomConfig):
    torch.manual_seed(42)

    batch_size = 1
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size
    vocab_size = test_config.model.embedding.vocab_size
    n_head = test_config.model.attention.heads
    tp1 = test_config.execution.tensor_parallel_1
    tp2 = test_config.execution.tensor_parallel_2

    # HuggingFace
    hf_config = HFConfig(vocab_size=vocab_size, hidden_size=hidden_size, n_head=n_head)
    hf_model = HFModel(hf_config).eval()

    # HF forward
    input_t = torch.randint(0, vocab_size, (batch_size, seq_len), requires_grad=False)
    output_ = hf_model.word_embeddings(input_t)
    output_ = hf_model.word_embeddings_layernorm(output_)

    input_HF = input_t.detach().numpy()
    output_HF = output_.detach().numpy()

    # popxl
    ir = popxl.Ir(replication=tp1 * tp2)

    input_HF_rs = repeat(input_HF, tp1 * tp2)

    with ir.main_graph:
        _, inputs_host_stream, inputs_tensors = zip(
            *[
                addons.host_load(input_HF_rs[0], popxl.int32, name="x"),
            ]
        )
        (x,) = inputs_tensors

        ff_args, ff_graph = BloomEmbeddingTP2D(test_config).create_graph(x)

        ff_vars = ff_args.init()
        ff = ff_graph.bind(ff_vars)
        fwd_info = ff.call_with_info(x)
        (acts,) = fwd_info.outputs

        fwd_d2h = addons.host_store(acts)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    weights = BloomEmbeddingTP2D.hf_mapping(test_config, ff_vars, hf_model)

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
    np.testing.assert_almost_equal(output_HF.reshape(fwd_data_np.shape), fwd_data_np, 6)
