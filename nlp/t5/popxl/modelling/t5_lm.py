# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Optional, Dict
import torch

# HF
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.layers import Linear
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_inputs
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.array_munging import shard

from modelling.layer_norm import T5LayerNorm
from modelling.t5_model import T5ModelTP
from config import T5Config
from math import ceil


def generate_greedy_tp(config: T5Config, logits: popxl.Tensor, last_token_index: popxl.Tensor):
    """
    Generate a new token based on greedy choice.
    Args:
        logits (popxl.Tensor, float16): Sharded logits for the whole sequence. Shape (micro_batch_size * seq_len, vocab_shard_size)
        last_token_index (popxl.Tensor, int32): Indices locating the last valid (non-padded) token for each batch. Logits at that indices correspond
                                                to the logits for the new token. It should be of shape (micro_batch_size,).
    Returns:
        (popxl.Tensor, int32): new token ids, of shape (micro_batch_size,)
    """
    tp = config.execution.tensor_parallel
    # indices for next token logits in each batch
    offsetted_batch_indices = popxl.constant(
        np.arange(0, config.execution.micro_batch_size) * config.model.sequence_length, dtype=popxl.int32
    )
    offsetted_batch_indices = last_token_index + offsetted_batch_indices
    # next token logits, sharded
    next_token_logits = logits[offsetted_batch_indices]  # (mb_size, vocab_shard_size)

    # gather tensor parallel shards and get full logits: (tp, mb_size, vocab_shard_size)
    next_token_logits = ops.collectives.replicated_all_gather(
        next_token_logits, group=popxl.gcg().ir.replica_grouping(group_size=tp), output_shape="new_axis"
    )
    next_token_logits = next_token_logits.transpose((1, 0, 2)).reshape_(
        (config.execution.micro_batch_size, config.model.embedding.vocab_size)
    )

    # (mb_size,)
    return ops.argmax(next_token_logits, dim=1)


class T5LMHeadTP(addons.Module):
    def __init__(self, config: T5Config):
        """
        Language model head for T5, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        """
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # identical
        self.ln_f = T5LayerNorm(self.config)
        shard_size = ceil(self.config.model.embedding.vocab_size / tp)
        self.head = Linear(shard_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.ln_f(x)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        # sharded
        x = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        logits = self.head(x)
        return logits

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel
        weights = {
            variables.head.weight: shard(to_numpy(hf_model.lm_head.weight.data.T, dtype), n_shards, axis=-1),
            variables.ln_f.weight: to_numpy(hf_model.decoder.final_layer_norm.weight.data, dtype),
        }
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        state_dict = {}
        state_dict["lm_head.weight"] = torch.tensor(
            np.concatenate(variables_data.head.weight.transpose(0, 2, 1), axis=0), dtype=config.torch_dtype
        )
        dec_ln_f = T5LayerNorm.to_hf(config, variables_data.ln_f, hf_model.decoder.final_layer_norm)
        state_dict.update({"decoder.final_layer_norm." + k: v for k, v in dec_ln_f.items()})
        return state_dict


class T5LMHeadModelTP(addons.Module):
    def __init__(self, config: T5Config):
        """T5 model with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = T5ModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = T5LMHeadTP(config)

    def build(
        self,
        input_ids: popxl.Tensor,
        dec_input_ids: popxl.Tensor,
        mask: popxl.Tensor,
        dec_mask: popxl.Tensor,
    ) -> popxl.Tensor:
        x = self.transformer(input_ids, dec_input_ids, mask, dec_mask)
        x = self.lm_head(x)
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        weights = T5ModelTP.hf_mapping(config, variables.transformer, hf_model, layer_norm=False)
        weights.update(T5LMHeadTP.hf_mapping(config, variables.lm_head, hf_model))
        return weights

    @staticmethod
    def to_hf(variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        state_dict = T5ModelTP.to_hf(variables_data.transformer, hf_model, layer_norm=False)
        state_dict.update(T5LMHeadTP.to_hf(hf_model.config, variables_data.lm_head, hf_model))
        return state_dict


class T5LMHeadLossAndGradTP(addons.Module):
    # Static variable
    ignore_idx: int = -100

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

    def build(self, x: popxl.Tensor, labels: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        fwd_facts, fwd_graph = T5LMHeadTP(self.config).create_graph(x, seed=seed)
        ts = self.add_variable_inputs("fwd", fwd_facts)

        # The ignore index is used to ignore labels corresponding to pad tokens
        ignore_index = popxl.constant(T5LMHeadLossAndGradTP.ignore_idx, popxl.int32, "ignore_index")
        loss_graph = GraphWithNamedArgs(
            fwd_graph.graph._ir.create_graph(
                cross_entropy_sharded_loss,
                fwd_graph.graph.outputs[0],
                labels,
                ignore_index=ignore_index,
                reduction="mean",
                replica_grouping=self.replica_grouping.transpose(),
            )
        )

        required_grads = [fwd_graph.graph.inputs[0]]
        accums = list(fwd_graph.args.tensors)
        replica_groupings = fwd_facts.replica_groupings.copy()

        bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(
            fwd_graph,
            tensors_to_accumulate_grads=accums,
            grads_required=required_grads,
            replica_groupings=replica_groupings,
        )
        loss_bwd = addons.transforms.autodiff(loss_graph, grads_required=(loss_graph.graph.inputs[0],))

        fwd_info = fwd_graph.bind(ts).call_with_info(x, seed)

        logits = fwd_info.parent_output(0)
        loss_fwd_info = loss_graph.call_with_info(logits, labels, ignore_index)
        loss = loss_fwd_info.parent_output(0)

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        (dx,) = loss_bwd.call(loss_scaling, args=loss_bwd.grad_graph_info.inputs_dict(loss_fwd_info))

        bwd_weights = self.add_variable_inputs("bwd", bwd_facts.copy())

        (dx,) = bwd_graph.bind(bwd_weights).call(dx, args=bwd_graph.grad_graph_info.inputs_dict(fwd_info))

        return loss, dx

    @staticmethod
    def get_offsets(config: T5Config):
        n_shards = config.execution.tensor_parallel
        vocab_shard_size = ceil(config.model.embedding.vocab_size / config.execution.tensor_parallel)
        return np.arange(n_shards * vocab_shard_size, step=vocab_shard_size)

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: T5Config):
        # We need to offset all indices in data by i * shard_size
        ignore_idx = T5LMHeadLossAndGradTP.ignore_idx
        mask = data == ignore_idx
        data = data - (i * ceil(config.model.embedding.vocab_size / config.execution.tensor_parallel))
        # Leave untouched the indices corresponding to the ignore_index
        data[mask] = ignore_idx
        return data
