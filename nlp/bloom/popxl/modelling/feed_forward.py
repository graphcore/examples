# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Optional

import numpy as np
import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard, shard2D
from popxl_addons.layers import Linear
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.utils import WeightsDict
from transformers.models.bloom.modeling_bloom import BloomBlock as HFModel

import popxl
from config import BloomConfig
from popxl import ops
from popxl.utils import to_numpy
from utils.utils import tp2d_replica_groups


class BloomFeedForwardTP2D(addons.Module):
    def __init__(self, config: BloomConfig, ff_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        self.ff_size = 4 * config.model.hidden_size if ff_size is None else ff_size

        self.ln_2 = LayerNormDistributed(self.rg_tp2)

        assert self.ff_size % self.rg_tp1.group_size == 0
        self.intermediate = Linear(
            self.ff_size // self.rg_tp1.group_size,
            bias=False,
            replica_grouping=self.rg_tp_all.transpose(),
        )

        assert config.model.hidden_size % self.rg_tp2.group_size == 0
        self.output = Linear(
            config.model.hidden_size // self.rg_tp2.group_size,
            bias=False,
            replica_grouping=self.rg_tp_all.transpose(),
        )

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """Identical input (x) and identical output across shards."""

        z = self.ln_2(x)
        z = self.intermediate(z)

        z = replicated_all_reduce(z, group=self.rg_tp2)

        self.intermediate_bias = self.add_variable_input(
            "intermediate_bias",
            partial(np.zeros, z.shape[-1]),
            z.dtype,
            replica_grouping=self.rg_tp1.transpose(),
        )
        z = z + self.intermediate_bias

        z = ops.gelu(z)
        z = self.output(z)

        z = replicated_all_reduce(z, group=self.rg_tp1)

        self.output_bias = self.add_variable_input(
            "output_bias",
            partial(np.zeros, z.shape[-1]),
            z.dtype,
            replica_grouping=self.rg_tp2.transpose(),
        )
        z = z + self.output_bias

        z = z + x

        return z

    @staticmethod
    def hf_mapping(config: BloomConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        dtype = config.model.dtype
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2

        return WeightsDict(
            {
                variables.ln_2.weight: shard(
                    to_numpy(hf_model.post_attention_layernorm.weight.data, dtype),
                    tp2,
                    0,
                ),
                variables.ln_2.bias: shard(to_numpy(hf_model.post_attention_layernorm.bias.data, dtype), tp2, 0),
                variables.intermediate.weight: shard2D(
                    to_numpy(hf_model.mlp.dense_h_to_4h.weight.data.T, dtype),
                    tp1,
                    tp2,
                    1,
                    0,
                ),
                variables.intermediate_bias: shard(to_numpy(hf_model.mlp.dense_h_to_4h.bias.data, dtype), tp1, 0),
                variables.output.weight: shard2D(
                    to_numpy(hf_model.mlp.dense_4h_to_h.weight.data.T, dtype),
                    tp1,
                    tp2,
                    0,
                    1,
                ),
                variables.output_bias: shard(to_numpy(hf_model.mlp.dense_4h_to_h.bias.data, dtype), tp2, 0),
            }
        )
