# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from typing import Optional, Dict, Tuple

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_grad_inputs
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData
from popxl.tensor import HostTensor

from popxl_addons.array_munging import shard, repeat
from configs import GPTJConfig

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoModel as HFModel
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF


class GPTJEmbeddingsTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=1)
        self.word = Embedding(
            self.config.dtype,
            self.config.embedding.real_vocab_size,
            self.config.hidden_size,
            replica_grouping=self.replica_grouping,
        )

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:
        """
        input_ids: (bs*seq_len, ) with values in range(0, real_vocab_size).
        They are offsetted so that indices outside the shard domain are OOR
        """

        input_ids = input_ids.flatten()
        x = self.word(input_ids)

        x = replicated_all_reduce_identical_grad_inputs(x, group=self.replica_grouping.transpose())

        #: (bs*seq_len, hidden_size)
        return x

    @staticmethod
    def finetuneanon_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel
    ) -> Dict[popxl.Tensor, np.ndarray]:
        """
        When using sharding, if real_vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective real_vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.dtype
        n_shards = config.execution.tensor_parallel
        word_shard_size = Embedding.get_vocab_shard_size(config.embedding.real_vocab_size, n_shards)

        word_pad = word_shard_size * n_shards - config.embedding.real_vocab_size
        # NOTE: Needed if we use vocab_size = 50400
        extra_pad = config.embedding.real_vocab_size - hf_model.wte.weight.data.shape[0]
        word_pad += extra_pad

        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        return {
            variables.word.weight: shard(pad(to_numpy(hf_model.wte.weight.data, dtype), word_pad), n_shards, axis=0),
        }

    @staticmethod
    def get_offsets(config: GPTJConfig) -> np.ndarray:
        n_shards = config.execution.tensor_parallel

        word_offsets = Embedding.get_offsets(config.embedding.real_vocab_size, n_shards)
        return word_offsets

    @staticmethod
    def get_vocab_shard_sizes(config: GPTJConfig) -> int:
        n_shards = config.execution.tensor_parallel

        word_shard_size = Embedding.get_vocab_shard_size(config.embedding.real_vocab_size, n_shards)

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
    def offset_inputs(cls, config: GPTJConfig, words: HostTensor, axis: int = 0):
        n_shards = config.execution.tensor_parallel
        word_offsets = GPTJEmbeddingsTP.get_offsets(config)

        words_offsetted = cls._offset_input(words, word_offsets, n_shards, axis)
        return words_offsetted

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: GPTJConfig):
        return data - (
            i * Embedding.get_vocab_shard_size(config.embedding.real_vocab_size, config.execution.tensor_parallel)
        )
