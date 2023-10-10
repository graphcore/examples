# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF

from transformers.models.llama.configuration_llama import LlamaConfig as HFConfig
from transformers.models.llama.modeling_llama import LlamaModel

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns

from config import LlamaConfig
from modelling.embedding import LlamaEmbeddingsTP
from modelling.llama_model import LlamaModelTP


def test_model_TP_cmp_huggingface(test_config: LlamaConfig):
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

    hf_model = LlamaModel(config).eval()

    # HF forward
    input_t = torch.randint(0, test_config.model.embedding.vocab_size, (batch_size, test_config.model.sequence_length))
    mask_t = torch.tensor(1e4 * (np.tril(np.ones((seq_len, seq_len))) - 1))[None, None, ...]
    output_HF = hf_model(input_ids=input_t, attention_mask=mask_t)[0]

    output_HF = output_HF.detach().numpy()

    # TP
    tp = test_config.execution.tensor_parallel

    # Offset inputs
    words_offsetted = LlamaEmbeddingsTP.offset_inputs(test_config, to_numpy(input_t))

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = tp
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors

        # layer norm moved to LM head in PopXL model - but for testing HF LlamaModel only, include in LlamaModelTP
        facts, graph = LlamaModelTP(test_config, include_layer_norm=True).create_graph(words)

        vars = facts.init()
        llm = graph.bind(vars)
        call_info = llm.call_with_info(words)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = LlamaModelTP.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]

    assert len(fwd_data) == tp
    for i in range(1, tp):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])

    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), 3)
