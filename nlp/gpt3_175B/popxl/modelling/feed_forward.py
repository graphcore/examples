# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Optional

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFModel

import popxl_addons as addons
from config import GPTConfig
from popxl_addons.layers import Linear, LayerNorm
import numpy as np

from popxl_addons import NamedTensors
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.utils import WeightsDict
from utils.utils import tp2d_replica_groups
from popxl_addons.array_munging import shard, shard2D


class GPTFeedForwardTP(addons.Module):
    def __init__(self, config: GPTConfig, ff_size: Optional[int] = None):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel_1
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
        self.intermediate = Linear(self.ff_size // self.n_shards, replica_grouping=self.replica_grouping)

        # Sharded across devices - row wise (bias applied separately)
        self.output = Linear(config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Identical computation -----

        z = self.ln_2(x)

        z = replicated_all_reduce_identical_inputs(z, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----
        z = self.intermediate(z)
        z = ops.gelu(z)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----

        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input("bias", lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = z + x

        return z

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel_1

        return WeightsDict(
            {
                # HF GPT2Block
                variables.ln_2.weight: to_numpy(hf_model.ln_2.weight.data, dtype),
                variables.ln_2.bias: to_numpy(hf_model.ln_2.bias.data, dtype),
                # HF GPT2MLP
                variables.intermediate.weight: shard(to_numpy(hf_model.mlp.c_fc.weight.data, dtype), n_shards, axis=-1),
                variables.intermediate.bias: shard(to_numpy(hf_model.mlp.c_fc.bias.data, dtype), n_shards, axis=-1),
                variables.output.weight: shard(to_numpy(hf_model.mlp.c_proj.weight.data, dtype), n_shards, axis=0),
                variables.bias: to_numpy(hf_model.mlp.c_proj.bias.data, dtype),
            }
        )


class GPTFeedForwardTP2D(addons.Module):
    def __init__(self, config: GPTConfig, ff_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        # Also known as the intermediate size
        self.ff_size = 4 * config.model.hidden_size if ff_size is None else ff_size

        self.ln_2 = LayerNormDistributed(self.rg_tp2)

        assert self.ff_size % self.rg_tp1.group_size == 0
        self.intermediate = Linear(
            self.ff_size // self.rg_tp1.group_size, bias=False, replica_grouping=self.rg_tp_all.transpose()
        )

        assert config.model.hidden_size % self.rg_tp2.group_size == 0
        self.output = Linear(
            config.model.hidden_size // self.rg_tp2.group_size, bias=False, replica_grouping=self.rg_tp_all.transpose()
        )

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """Identical input (x, seed) and identical output across shards."""
        # x: [b*s, h/tp2]
        # x: identical tp1, sharded tp2

        z = self.ln_2(x)

        z = replicated_all_reduce_identical_inputs(z, group=self.rg_tp1)
        # z: identical tp1, sharded tp2

        z = self.intermediate(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.rg_tp2)
        # z: sharded tp1, identical tp2

        self.intermediate_bias = self.add_variable_input(
            "intermediate_bias", partial(np.zeros, z.shape[-1]), z.dtype, replica_grouping=self.rg_tp1.transpose()
        )
        z = z + self.intermediate_bias

        z = replicated_all_reduce_identical_inputs(z, group=self.rg_tp2)
        # z: sharded tp1, identical tp2

        z = ops.gelu(z)

        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.rg_tp1)
        # z: identical tp1, sharded tp2

        self.output_bias = self.add_variable_input(
            "output_bias", partial(np.zeros, z.shape[-1]), z.dtype, replica_grouping=self.rg_tp2.transpose()
        )
        z = z + self.output_bias

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = z + x

        # output data: identical tp1, sharded tp2
        return z

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        dtype = config.model.dtype
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2

        return WeightsDict(
            {
                # HF GPT2Block
                variables.ln_2.weight: shard(to_numpy(hf_model.ln_2.weight.data, dtype), tp2, 0),
                variables.ln_2.bias: shard(to_numpy(hf_model.ln_2.bias.data, dtype), tp2, 0),
                # HF GPT2MLP
                variables.intermediate.weight: shard2D(to_numpy(hf_model.mlp.c_fc.weight.data, dtype), tp1, tp2, 1, 0),
                variables.intermediate_bias: shard(to_numpy(hf_model.mlp.c_fc.bias.data, dtype), tp1, 0),
                variables.output.weight: shard2D(to_numpy(hf_model.mlp.c_proj.weight.data, dtype), tp1, tp2, 0, 1),
                variables.output_bias: shard(to_numpy(hf_model.mlp.c_proj.bias.data, dtype), tp2, 0),
            }
        )
