# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl_addons as addons
import torch
import torch.nn as nn

import popxl
from config import InferenceConfig
from modeling.modeling_cached_TP import EmbeddingsTP
from popxl.utils import to_numpy


class rudalleEmbeddings(nn.Module):
    def __init__(self, vocab_size, image_vocab_size, hidden_size,
                 text_seq_len, image_tokens_per_dim):
        super().__init__()
        self.text_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.image_embeddings = nn.Embedding(image_vocab_size, hidden_size)

        self.text_pos_embeddings = nn.Embedding(text_seq_len + 1, hidden_size)
        self.image_row_embeddings = nn.Embedding(image_tokens_per_dim, hidden_size)
        self.image_col_embeddings = nn.Embedding(image_tokens_per_dim, hidden_size)

    def forward(self, input_ids):
        pos_embeddings = self.text_pos_embeddings(torch.arange(input_ids.shape[1]))
        embeddings = self.text_embeddings(input_ids) + pos_embeddings

        return embeddings


def test_embedding_TP():
    test_config = InferenceConfig()
    test_config.dtype = popxl.float32
    torch.manual_seed(42)

    vocab_size = test_config.vocab_size + test_config.text_seq_len
    image_vocab_size = test_config.image_vocab_size
    hidden_size = test_config.n_embd
    text_seq_len = test_config.text_seq_len
    image_tokens_per_dim = int(test_config.image_seq_len**0.5)

    module = rudalleEmbeddings(vocab_size, image_vocab_size, hidden_size,
                               text_seq_len, image_tokens_per_dim).eval()

    # PyTorch
    # HF forward
    words = torch.tensor([[ 2, 4078, 4073, 4140,   77,   51, 4441, 7787,    3,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0]], dtype=torch.long)
    text_pad = test_config.vocab_size + torch.arange(test_config.text_seq_len)
    words = torch.where(words == 0, text_pad, words)
    output_torch = module(
        input_ids=words,
    )

    # Offset inputs
    words = words.to(dtype = torch.int32).reshape(-1,)
    test_config.seq_len = test_config.text_seq_len
    words_offsetted, pos_offsetted = EmbeddingsTP.offset_inputs(test_config, to_numpy(words))

    # popart.ir
    ir = popxl.Ir()
    ir.replication_factor = test_config.ipus
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(words_offsetted[0], popxl.int32, name="words"),
        ])
        x, = inputs_tensors
        pos = popxl.variable(
            pos_offsetted, popxl.int32, name="positions", replica_grouping=replica_grouping)
        args, graph = EmbeddingsTP(test_config, replica_grouping).create_graph(x, pos)

        vars = args.init("embeddings_TP")
        embed = graph.bind(vars)
        call_info = embed.call_with_info(x, pos)
        acts, = call_info.outputs
        fwd_d2h = addons.host_store(acts)

    weights = EmbeddingsTP.hf_mapping(test_config, vars, module)
    inputs = dict(zip(inputs_host_steam, [words_offsetted]))

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outputs_popxl = session.run(inputs)

    np.testing.assert_almost_equal(output_torch.detach().numpy()[0], outputs_popxl[fwd_d2h][0], 5)


if __name__ == "__main__":
    test_embedding_TP()
