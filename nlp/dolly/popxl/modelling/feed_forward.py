# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Dict
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP as HFModel

import popxl_addons as addons
from config import DollyConfig
from popxl_addons.layers import Linear
import numpy as np

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.array_munging import shard


class DollyFeedForwardTP(addons.Module):
    def __init__(self, config: DollyConfig, ff_size: Optional[int] = None):
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
        # Sharded across devices - column wise
        self.intermediate = Linear(self.ff_size // self.n_shards, replica_grouping=self.replica_grouping)

        # Sharded across devices - row wise (bias applied separately)
        self.output = Linear(config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Sharded computation -----

        # Shard column-wise since gelu is not linear.
        # Indeed, sharding row wise requires a sum AllReduce at the end,
        # but gelu is not linear: gelu(x+y) != gelu(x) + gelu(y)
        z = self.intermediate(x)
        z = ops.gelu(z)
        # Here, x is already sharded across devices. Since we don't have non linearities,
        # we can shard row-wise (which requires both X and the weight matrix to be sharded)
        # and then perform an all reduce
        z = self.output(z)

        z = replicated_all_reduce(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----

        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input("bias", lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        return z

    @staticmethod
    def hf_mapping(config: DollyConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        return {
            # HF DollyMLP
            variables.intermediate.weight: shard(
                to_numpy(hf_model.dense_h_to_4h.weight.data.T, dtype), n_shards, axis=-1
            ),
            variables.intermediate.bias: shard(to_numpy(hf_model.dense_h_to_4h.bias.data, dtype), n_shards, axis=-1),
            variables.output.weight: shard(to_numpy(hf_model.dense_4h_to_h.weight.data.T, dtype), n_shards, axis=0),
            variables.bias: to_numpy(hf_model.dense_4h_to_h.bias.data, dtype),
        }
