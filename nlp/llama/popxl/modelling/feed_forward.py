# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Dict
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.llama.modeling_llama import LlamaMLP as HFModel

import popxl_addons as addons
from config import LlamaConfig
from popxl_addons.layers import Linear
import numpy as np

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.array_munging import shard


class LlamaFeedForwardTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.n_shards = tp
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        self.intermediate_size = self.config.model.intermediate_size
        self.hidden_size = self.config.model.hidden_size

        # ----- Layers -----
        # Sharded across devices - column wise
        self.gate_proj = Linear(
            self.intermediate_size // self.n_shards, bias=False, replica_grouping=self.replica_grouping
        )
        self.up_proj = Linear(
            self.intermediate_size // self.n_shards, bias=False, replica_grouping=self.replica_grouping
        )

        # Sharded across devices - row wise (no bias)
        self.down_proj = Linear(self.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Sharded computation -----

        # Shard column-wise since gelu is not linear.
        # Indeed, sharding row wise requires a sum AllReduce at the end,
        # but swish is not linear: swish(x+y) != swish(x) + swish(y)
        up = self.up_proj(x)

        gp = self.gate_proj(x)
        gp_act = ops.swish(gp)
        # Here, x is already sharded across devices. Since we don't have non linearities,
        # we can shard row-wise (which requires both X and the weight matrix to be sharded)
        # and then perform an all reduce
        z = gp_act * up

        z = self.down_proj(z)
        z = replicated_all_reduce(z, group=self.replica_grouping.transpose())
        return z

    @staticmethod
    def hf_mapping(config: LlamaConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        return {
            variables.gate_proj.weight: shard(to_numpy(hf_model.gate_proj.weight.data.T, dtype), n_shards, axis=-1),
            variables.up_proj.weight: shard(to_numpy(hf_model.up_proj.weight.data.T, dtype), n_shards, axis=-1),
            variables.down_proj.weight: shard(to_numpy(hf_model.down_proj.weight.data.T, dtype), n_shards, axis=0),
        }
