# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Dict
import torch
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.t5.modeling_t5 import T5DenseGatedActDense as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

import popxl_addons as addons
from config import T5Config
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.layers import Linear
import numpy as np

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.array_munging import shard


class T5FeedForwardTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.n_shards = tp
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # Also known as the intermediate size
        self.ff_size = config.model.d_ff
        assert self.ff_size % self.n_shards == 0
        # ----- Layers -----
        # Sharded across devices - column wise
        self.intermediate = Linear(
            2 * self.ff_size // self.n_shards, bias=False, replica_grouping=self.replica_grouping
        )

        # Sharded across devices - row wise (bias applied separately)
        self.output = Linear(config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)
        self.scale_ff = config.model.scale_ff

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----

        # Shard column-wise since gelu is not linear.
        # Indeed, sharding row wise requires a sum AllReduce at the end,
        # but gelu is not linear: gelu(x+y) != gelu(x) + gelu(y)
        # The following computation is called GeGLU (https://arxiv.org/abs/2002.05202)
        # We combine the 2 initial ff into a single matmul, then we split the result
        z = self.intermediate(z)
        z_gelu, z_lin = ops.split(z, 2, axis=-1)
        z_gelu = ops.gelu(z_gelu)
        z = z_gelu * z_lin
        # Here, z is already sharded across devices. Since we don't have non linearities,
        # we can shard row-wise (which requires both X and the weight matrix to be sharded)
        # and then perform an all reduce
        # Scale down if needed
        if self.scale_ff > 1:
            z = z / self.scale_ff
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----

        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        return z

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        hf_wi0 = to_numpy(hf_model.wi_0.weight.data, dtype).T
        hf_wi1 = to_numpy(hf_model.wi_1.weight.data, dtype).T
        wi0 = shard(hf_wi0, n_shards, -1)
        wi1 = shard(hf_wi1, n_shards, -1)

        return {
            variables.intermediate.weight: np.ascontiguousarray(
                np.concatenate([np.concatenate([wi0[i], wi1[i]], axis=-1)[np.newaxis, ...] for i in range(n_shards)])
            ),
            variables.output.weight: shard(to_numpy(hf_model.wo.weight.data.T, dtype), n_shards, axis=0),
        }

    @staticmethod
    def to_hf(config: T5ConfigHF, popxl_state_dict: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        """
        Given variables for the popxl layer, returns the state dictionary for the corresponding HF layer.
        Usage:
            ...
            with session:
                variables_data = session.get_tensors_data(vars.tensors)
            ...
            state_dict = T5FeedForwardTP.to_hf(hf_config, variables_data, hf_model)
            hf_model.load_state_dict(state_dict)
            hf_model.to(device)
        """
        wi0, wi1 = np.split(popxl_state_dict.intermediate.weight, 2, axis=-1)
        state_dict = {}
        state_dict["wi_0.weight"] = torch.tensor(
            np.concatenate(wi0.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["wi_1.weight"] = torch.tensor(
            np.concatenate(wi1.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["wo.weight"] = torch.tensor(
            np.concatenate(popxl_state_dict.output.weight, axis=0).T, dtype=config.torch_dtype
        )

        return state_dict
