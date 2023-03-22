# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# finetuneanon
from transformers.models.gpt_neo import GPTNeoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons import TaskSession
from popxl_addons.named_tensors import NamedTensors

from configs import MagmaConfig
from modelling import GPTJEmbeddingsTP
from modelling import GPTJLMHeadModelTP
from popxl_addons.array_munging import shard


def test_lm_TP_cmp_huggingface(test_config: MagmaConfig):
    torch.manual_seed(42)
    test_config = test_config.transformer

    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.hidden_size
    intermediate_size = hidden_size * 4
    seq_len = test_config.sequence_length

    # finetunenanon
    config = GPTNeoConfig(
        hidden_size=hidden_size,
        num_layers=test_config.layers,
        attention_layers=["global"] * test_config.layers,
        attention_types=[[["global"], test_config.layers]],
        max_position_embeddings=seq_len,
        vocab_size=test_config.embedding.real_vocab_size,
        intermediate_size=intermediate_size,
        num_heads=test_config.attention.heads,
        rotary=True,
        rotary_dim=test_config.attention.rotary_dim,
        use_cache=False,
        jax=True,
    )
    torch_model = GPTNeoForCausalLM(config).eval()
    if test_config.dtype == popxl.float16:
        torch_model = torch_model.half()
    else:
        torch_model = torch_model.float()

    # torch forward
    input_t = torch.randint(0, test_config.embedding.real_vocab_size, (batch_size, test_config.sequence_length))
    output_torch = torch_model(input_ids=input_t)[0].detach().numpy()

    # n_shards
    n_shards = 4
    test_config.execution.tensor_parallel = n_shards

    # Offset inputs
    words_offsetted = GPTJEmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))
    # popxl
    ir = popxl.Ir()
    ir.replication_factor = n_shards
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors
        facts, graph = GPTJLMHeadModelTP(test_config).create_graph(words)
        vars = facts.init()
        gpt = graph.bind(vars)
        call_info = gpt.call_with_info(words)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)
        word_shard_size = GPTJEmbeddingsTP.get_vocab_shard_sizes(test_config)

    apply_pre_alias_patterns(ir, level="default")

    # Map weights
    weights = GPTJLMHeadModelTP.finetuneanon_mapping(test_config, vars, torch_model, from_magma=False)

    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    ir.num_host_transfers = 1

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]
    assert len(fwd_data) == n_shards
    fwd_data_full = np.concatenate(fwd_data, axis=-1)[:, : test_config.embedding.real_vocab_size]
    np.testing.assert_almost_equal(output_torch, fwd_data_full.reshape(output_torch.shape), 3)
