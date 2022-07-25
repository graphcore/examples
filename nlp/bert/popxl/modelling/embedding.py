# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Dict, Tuple
import numpy as np

import popxl
from popxl import ops
from popxl.utils import to_numpy

from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings

import popxl_addons as addons
from popxl_addons import NamedTensors

from config import BertConfig
from popxl_addons.layers import LayerNorm, Embedding


def generate_positions(config: BertConfig) -> np.ndarray:
    pos = np.repeat(
        np.arange(0, config.model.sequence_length).reshape(1, -1),
        config.execution.micro_batch_size,
        axis=0).flatten().copy()
    return pos


class BertEmbeddings(addons.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.word = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size)
        self.positional = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.max_positional_length,
            self.config.model.hidden_size)
        self.token_type = Embedding(
            self.config.model.dtype,
            2,
            self.config.model.hidden_size)
        self.norm = LayerNorm()

    def build(self,
              words: popxl.Tensor,
              token_types: popxl.Tensor,
              positions: Optional[popxl.Tensor] = None,
              seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        if positions is None:
            positions = popxl.constant(
                generate_positions(self.config).reshape(words.shape),
                popxl.uint32,
                name="positions")

        word = self.word(words)
        positional = self.positional(positions)
        token_type = self.token_type(token_types)
        x = (word + positional) + token_type

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)
        x = self.norm(x)
        return x

    @staticmethod
    def hf_mapping(
            config: BertConfig, variables: NamedTensors, hf_model: HFBertEmbeddings) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        word_shard_size, pos_shard_size, token_shard_size = BertEmbeddings.get_vocab_shard_sizes(config)

        word_pad = word_shard_size - config.model.embedding.vocab_size
        pos_pad = pos_shard_size - \
            config.model.embedding.max_positional_length
        token_pad = token_shard_size - 2

        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))


        return {
            variables.word.weight: pad(to_numpy(hf_model.word_embeddings.weight.data, dtype), word_pad),
            variables.positional.weight: pad(to_numpy(hf_model.position_embeddings.weight.data, dtype), pos_pad),
            variables.token_type.weight: pad(to_numpy(hf_model.token_type_embeddings.weight.data, dtype), token_pad),
            variables.norm.weight: to_numpy(hf_model.LayerNorm.weight.data, dtype),
            variables.norm.bias: to_numpy(hf_model.LayerNorm.bias.data, dtype),
        }

    @staticmethod
    def get_vocab_shard_sizes(config: BertConfig) -> Tuple[int, int, int]:
        n_shards = 1

        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)
        pos_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.max_positional_length, n_shards)
        token_shard_size = Embedding.get_vocab_shard_size(2, n_shards)

        return word_shard_size, pos_shard_size, token_shard_size
