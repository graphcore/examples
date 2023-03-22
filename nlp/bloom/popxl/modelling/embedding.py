# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil
from typing import Dict

import numpy as np
import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import pad_axis, shard, shard2D
from popxl_addons.layers import Embedding
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.utils import WeightsDict
from scipy.stats import truncnorm
from transformers.models.bloom.modeling_bloom import BloomModel as HFModel

import popxl
from config import BloomConfig
from popxl import ops
from popxl.utils import to_numpy
from utils.utils import tp2d_replica_groups

# The dynamic slice libraries particularly like multiples of 16.
MULTISLICE_MAGIC_NUMBER = 16


class BloomEmbeddingTP2D(addons.Module):
    def __init__(self, config: BloomConfig, axis: int = 0):
        super().__init__()
        self.dtype = config.model.dtype
        self.vocab_size = config.model.embedding.vocab_size
        self.hidden_size = config.model.hidden_size
        self.axis = axis
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        assert self.hidden_size % self.rg_tp2.group_size == 0

        self.vocab_shard_size = self.get_vocab_shard_size(self.vocab_size, self.rg_tp1.group_size)
        self.hidden_shard_size = self.hidden_size // self.rg_tp2.group_size
        self.offsets = self.get_offsets(self.vocab_size, self.rg_tp1.group_size)

        self.post_embedding_norm = LayerNormDistributed(self.rg_tp2)

    @staticmethod
    def get_vocab_shard_size(vocab_size: int, n_shards: int) -> int:
        """Vocab size per shard."""
        shard_size = ceil(vocab_size / n_shards)
        return MULTISLICE_MAGIC_NUMBER * ceil(shard_size / MULTISLICE_MAGIC_NUMBER)

    @staticmethod
    def get_padded_size(vocab_size: int, n_shards: int) -> int:
        """Total size of vocab including padding."""
        vocab_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
        return n_shards * vocab_shard_size

    @staticmethod
    def get_offsets(vocab_size: int, n_shards: int) -> np.ndarray:
        """Indices offset per vocab shard."""
        vocab_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
        return np.arange(n_shards * vocab_shard_size, step=vocab_shard_size)

    def build(self, indices: popxl.Tensor) -> popxl.Tensor:
        """

        Args:
            indices: token indices. Shape (...,sequence_length, vocab_size)

        Returns:
            Embedded vectors for each index. Shape (...,sequence_length, hidden_size)
            Embedding corresponds to a table lookup: each index in `indices` selects a row in the embedding weights matrix.
            If using sharding, out of range indices will be automatically set to zero.
        """
        # indices: identical tp1 & tp2
        indices = indices.flatten()
        self.offset = self.add_variable_input(
            "offset",
            iter(self.get_offsets(self.vocab_size, self.rg_tp1.group_size)),
            indices.dtype,
            replica_grouping=self.rg_tp1.transpose(),
        )

        indices_offsetted = indices - self.offset

        # Split embedding layer into two tensors.
        # required to get around maximum transfer size limitation in poplar.
        assert self.hidden_shard_size % 2 == 0, "Hidden shard size is not perfectly divisible by 2."
        self.weight_1 = self.add_variable_input(
            "weight_1",
            partial(
                truncnorm.rvs,
                -2,
                2,
                loc=0,
                scale=0.02,
                size=(self.vocab_shard_size, self.hidden_shard_size // 2),
            ),
            self.dtype,
            replica_grouping=self.rg_tp_all.transpose(),
        )
        z1 = ops.gather(self.weight_1, indices_offsetted, axis=self.axis, zero_OOR=True)

        self.weight_2 = self.add_variable_input(
            "weight_2",
            partial(
                truncnorm.rvs,
                -2,
                2,
                loc=0,
                scale=0.02,
                size=(self.vocab_shard_size, self.hidden_shard_size // 2),
            ),
            self.dtype,
            replica_grouping=self.rg_tp_all.transpose(),
        )
        z2 = ops.gather(self.weight_2, indices_offsetted, axis=self.axis, zero_OOR=True)

        z = ops.concat((z1, z2), axis=-1)
        z = replicated_all_reduce(z, group=self.rg_tp1)
        z = self.post_embedding_norm(z)
        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        vocab_size = config.model.embedding.vocab_size
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2
        assert hf_model.word_embeddings.weight.data.shape[0] == vocab_size

        word_padded = BloomEmbeddingTP2D.get_padded_size(vocab_size, tp1)

        embedding_chunks = np.split(
            shard2D(
                pad_axis(
                    to_numpy(hf_model.word_embeddings.weight.data, dtype),
                    word_padded,
                    0,
                ),
                tp1,
                tp2,
                0,
                1,
            ),
            2,  # split embedding layer into two chunks
            axis=-1,
        )

        return WeightsDict(
            {
                variables.weight_1: embedding_chunks[0],
                variables.weight_2: embedding_chunks[1],
                variables.post_embedding_norm.weight: shard(
                    to_numpy(hf_model.word_embeddings_layernorm.weight.data, dtype), tp2, 0
                ),
                variables.post_embedding_norm.bias: shard(
                    to_numpy(hf_model.word_embeddings_layernorm.bias.data, dtype), tp2, 0
                ),
                variables.offset: np.array(
                    BloomEmbeddingTP2D.get_offsets(config.model.embedding.vocab_size, tp1),
                    dtype=np.int32,
                ),
            }
        )
