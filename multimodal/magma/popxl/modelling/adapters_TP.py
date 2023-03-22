# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, List
import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from configs import GPTJConfig
from popxl_addons.layers import Linear
from popxl_addons.layers.layer_norm import LayerNorm
import numpy as np

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.array_munging import shard


class AdapterTP(addons.Module):
    def __init__(self, config: GPTJConfig, dim: int, downsample_factor: int = 4, add_layernorm: bool = False):

        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        self.n_shards = tp
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=1)

        self.ln = None
        if add_layernorm:
            self.ln = LayerNorm()

        # Also known as the intermediate size
        self.ff_size = dim // downsample_factor
        assert self.ff_size % self.n_shards == 0
        # ----- Layers -----
        # Sharded across devices - column wise
        self.compress = Linear(self.ff_size // self.n_shards, replica_grouping=self.replica_grouping)

        # Sharded across devices - row wise (bias applied separately)
        self.expand = Linear(dim, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        residual = x
        if self.ln:
            x = self.ln(x)

        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----

        z = self.compress(z)
        z = ops.relu(z)
        z = self.expand(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----

        self.bias = self.add_variable_input("bias", lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        return z + residual

    @staticmethod
    def magma_mapping(config: GPTJConfig, magma_model, variables: NamedTensors):
        n_shards = config.execution.tensor_parallel

        assert len(magma_model.adapter) == 4 or len(magma_model.adapter) == 3

        state_dict = {}

        if len(magma_model.adapter) == 4:  # Case with LayerNorm
            state_dict.update(
                {
                    variables.ln.weight: to_numpy(magma_model.adapter[0].weight.data),
                    variables.ln.bias: to_numpy(magma_model.adapter[0].bias.data),
                }
            )
            adapt_idx = [1, 3]
        else:
            adapt_idx = [0, 2]

        state_dict.update(
            {
                variables.compress.weight: shard(
                    to_numpy(magma_model.adapter[adapt_idx[0]].weight.data.T), n_shards, axis=-1
                ),
                variables.compress.bias: shard(
                    to_numpy(magma_model.adapter[adapt_idx[0]].bias.data), n_shards, axis=-1
                ),
                variables.expand.weight: shard(
                    to_numpy(magma_model.adapter[adapt_idx[1]].weight.data.T), n_shards, axis=0
                ),
                variables.bias: to_numpy(magma_model.adapter[adapt_idx[1]].bias.data),
            }
        )

        return state_dict
