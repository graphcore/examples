# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl_addons as addons
import torch
from rudalle.dalle.image_attention import _init_mask
from rudalle.dalle.transformer import DalleTransformerLayer

import popxl
from config import InferenceConfig
from modeling.modeling_cached_TP import BlockTP


def test_block_cmp_huggingface():
    test_config = InferenceConfig()
    test_config.dtype = popxl.float32
    torch.manual_seed(42)

    # HuggingFace
    micro_bs = 1
    seq_len = test_config.text_seq_len
    hidden_size = test_config.n_embd
    module = DalleTransformerLayer(hidden_size, test_config.n_head, 0, 0, 1e-5, True).eval()
    mask = _init_mask(text_tokens = 128, image_tokens_per_dim = 32)
    mask = mask[:seq_len, :seq_len].reshape((1, 1, seq_len, seq_len))

    # HF forward
    input_t = torch.rand((micro_bs, seq_len, hidden_size), requires_grad=False)
    output_ = module(input_t, torch.Tensor(mask), False, None)[0]
    output_HF = output_.detach().numpy()

    # popart.ir
    ir = popxl.Ir()
    ir.replication_factor = test_config.ipus
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)

    main = ir.main_graph

    input_t = input_t.repeat(test_config.ipus, 1, 1)
    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(input_t[0], popxl.float32, name="input"),
        ])
        input = inputs_tensors[0]

        args, block_graph = BlockTP(test_config, replica_grouping).create_graph(input)

        block = args.init("block")
        layer = block_graph.bind(block)
        call_info = layer.call_with_info(input)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)

    # Map weights from huggingface
    weights = BlockTP.hf_mapping(test_config, block, module)
    inputs = dict(zip(inputs_host_steam, [input_t]))

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    np.testing.assert_almost_equal(output_HF[0], outputs_popxl[act_stream][0], 5)


if __name__ == "__main__":
    test_block_cmp_huggingface()
