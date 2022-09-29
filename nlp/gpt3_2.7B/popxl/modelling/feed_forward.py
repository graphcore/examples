# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Dict

import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFModel

import popxl_addons as addons
from config import GPTConfig
from popxl_addons.layers import Linear, LayerNorm
import numpy as np

from popxl_addons import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs)
from utils.utils import shard


class GPTFeedForwardTP(addons.Module):
    def __init__(self, config: GPTConfig, ff_size: Optional[int] = None):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.n_shards = tp
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # Also known as the intermediate size
        self.ff_size = 4 * config.model.hidden_size if ff_size is None else ff_size
        assert self.ff_size % self.n_shards == 0
        # ----- Layers -----
        # in original hf impl layer norm ln_2 and final add are in the decoder block,
        # but we are incorporating them in this layer for better model partitioning

        # Identical across devices
        self.ln_2 = LayerNorm()

        # Sharded across devices - column wise
        self.intermediate = Linear(self.ff_size // self.n_shards,
                                   replica_grouping=self.replica_grouping)

        # Sharded across devices - row wise (bias applied separately)
        self.output = Linear(config.model.hidden_size,
                             bias=False,
                             replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Identical computation -----
        z = self.ln_2(x)

        z = replicated_all_reduce_identical_inputs(z, group = self.replica_grouping.transpose())

        # ----- Sharded computation -----

        # Shard column-wise since gelu is not linear.
        # Indeed, sharding row wise requires a sum AllReduce at the end,
        # but gelu is not linear: gelu(x+y) != gelu(x) + gelu(y)
        z = self.intermediate(z)
        z = ops.gelu(z)
        # Here, x is already sharded across devices. Since we don't have non linearities,
        # we can shard row-wise (which requires both X and the weight matrix to be sharded)
        # and then perform an all reduce
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group = self.replica_grouping.transpose())

        # ----- Identical computation -----

        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input(
            'bias', lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = z + x

        return z

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        return {
            # HF GPT2Block
            variables.ln_2.weight: to_numpy(hf_model.ln_2.weight.data, dtype),
            variables.ln_2.bias: to_numpy(hf_model.ln_2.bias.data, dtype),
            # HF GPT2MLP
            variables.intermediate.weight: shard(to_numpy(hf_model.mlp.c_fc.weight.data, dtype), n_shards, axis=-1),
            variables.intermediate.bias: shard(to_numpy(hf_model.mlp.c_fc.bias.data, dtype), n_shards, axis=-1),
            variables.output.weight: shard(to_numpy(hf_model.mlp.c_proj.weight.data, dtype), n_shards, axis=0),
            variables.bias: to_numpy(hf_model.mlp.c_proj.bias.data, dtype),
        }
