# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Dict
import torch
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.gptj.modeling_gptj import GPTJMLP as HFModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF

import popxl_addons as addons
from config import GPTJConfig
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.layers import Linear, LayerNorm
import numpy as np

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from utils.utils import shard


class GPTJFeedForwardTP(addons.Module):
    def __init__(self, config: GPTJConfig, ff_size: Optional[int] = None):
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

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> List[popxl.Tensor]:
        """Identical input (x, seed) and identical output across shards."""
        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

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

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----

        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input("bias", lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        return z

    @staticmethod
    def hf_mapping(config: GPTJConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        return {
            # HF GPTJMLP
            variables.intermediate.weight: shard(to_numpy(hf_model.fc_in.weight.data.T, dtype), n_shards, axis=-1),
            variables.intermediate.bias: shard(to_numpy(hf_model.fc_in.bias.data, dtype), n_shards, axis=-1),
            variables.output.weight: shard(to_numpy(hf_model.fc_out.weight.data.T, dtype), n_shards, axis=0),
            variables.bias: to_numpy(hf_model.fc_out.bias.data, dtype),
        }

    @staticmethod
    def to_hf(config: GPTJConfigHF, popxl_state_dict: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        """
        Given variables for the popxl layer, returns the state dictionary for the corresponding HF layer.
        Usage:
            ...
            with session:
                variables_data = session.get_tensors_data(vars.tensors)
            ...
            state_dict = GPTJFeedForwardTP.to_hf(hf_config, variables_data, hf_model)
            hf_model.load_state_dict(state_dict)
            hf_model.to(device)
        """
        state_dict = {}
        state_dict["fc_in.weight"] = torch.tensor(
            np.concatenate(popxl_state_dict.intermediate.weight.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["fc_in.bias"] = torch.tensor(
            np.concatenate(popxl_state_dict.intermediate.bias, axis=0), dtype=config.torch_dtype
        )
        state_dict["fc_out.weight"] = torch.tensor(
            np.concatenate(popxl_state_dict.output.weight, axis=0).T, dtype=config.torch_dtype
        )
        state_dict["fc_out.bias"] = torch.tensor(popxl_state_dict.bias, dtype=config.torch_dtype)

        return state_dict
