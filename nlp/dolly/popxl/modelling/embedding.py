# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons import NamedTensors
from popxl.tensor import HostTensor

from popxl_addons.array_munging import shard, repeat
from config import DollyConfig

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel as HFModel


class DollyEmbeddingsTP(addons.Module):
    def __init__(self, config: DollyConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        self.word = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size,
            replica_grouping=self.replica_grouping,
        )

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:
        """`input_ids` are offset. Identical outputs across shards."""
        input_ids = input_ids.flatten()
        x = self.word(input_ids)

        x = replicated_all_reduce(x, group=self.replica_grouping.transpose())

        return x

    @staticmethod
    def hf_mapping(config: DollyConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        """
        When using sharding, if `vocab_size` is not a multiple of `n_shards`, the resulting embedding layers have a larger
        effective `vocab_size` with respect to `hf_model`. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel
        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)

        word_pad = word_shard_size * n_shards - config.model.embedding.vocab_size

        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        return {
            variables.word.weight: shard(
                pad(to_numpy(hf_model.embed_in.weight.data, dtype), word_pad), n_shards, axis=0
            ),
        }

    @staticmethod
    def get_offsets(config: DollyConfig) -> np.ndarray:
        n_shards = config.execution.tensor_parallel

        word_offsets = Embedding.get_offsets(config.model.embedding.vocab_size, n_shards)
        return word_offsets

    @staticmethod
    def get_vocab_shard_sizes(config: DollyConfig) -> int:
        n_shards = config.execution.tensor_parallel

        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)

        return word_shard_size

    @staticmethod
    def _offset_input(data: HostTensor, offsets: HostTensor, n_shards, axis: int = 0):
        def bc_shape(t):
            # Shape for broadcasting. `slice(None, None)` represents all like `array[:]`
            shape = [np.newaxis] * len(t.shape)
            shape.insert(axis, slice(None, None))
            return tuple(shape)

        data_offsetted = repeat(data, n_shards, axis) - offsets[bc_shape(data)]
        return data_offsetted

    @classmethod
    def offset_inputs(cls, config: DollyConfig, words: HostTensor, axis: int = 0):
        n_shards = config.execution.tensor_parallel
        word_offsets = DollyEmbeddingsTP.get_offsets(config)

        words_offsetted = cls._offset_input(words, word_offsets, n_shards, axis)
        return words_offsetted

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: DollyConfig):
        return data - (
            i * Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, config.execution.tensor_parallel)
        )
