# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
from typing import Optional, Dict, Tuple, Callable, List, Union
import logging
import torch

# HF
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM as HFModel
from transformers.models.gptj.modeling_gptj import GPTJModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.layers import Linear
from popxl_addons.layers.linear_gq import LinearGQ, group_quantize_compress_numpy
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
)
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss
from popxl_addons.layers import LayerNorm
from popxl_addons.named_replica_grouping import fill_none_group
from popxl_addons.named_tensors import NamedTensorData

from modelling.gptj_model import GPTJModelTP
from config import GPTJConfig
from utils.utils import shard
from math import ceil


def generate_greedy_tp(config: GPTJConfig, logits: popxl.Tensor, last_token_index: popxl.Tensor):
    """
    Generate a new token based on greedy choice.
    Args:
        logits (popxl.Tensor, int32): Sharded logits for the whole sequence. Shape (seq_len, vocab_shard_size)
        last_token_index (popxl.Tensor, int32): Indices locating the last valid (non-padded) token for each batch. Logits at that indices correspond
                                                to the logits for the new token. It should be of shape (micro_batch_size,).
    Returns:
        (popxl.Tensor, int32): new token ids, of shape (micro_batch_size,)
    """
    tp = config.execution.tensor_parallel
    # indices for next token logits in each batch
    offsetted_batch_indices = popxl.constant(
        np.arange(0, config.execution.micro_batch_size) * config.model.sequence_length,
        dtype=popxl.int32,
    )
    offsetted_batch_indices = last_token_index + offsetted_batch_indices
    # next token logits, sharded
    next_token_logits = logits[offsetted_batch_indices]  # (tp, mb_size, vocab_shard_size)

    # gather tensor parallel shards and get full logits: (mb_size, vocab_size)
    next_token_logits = ops.collectives.replicated_all_gather(
        next_token_logits,
        group=popxl.gcg().ir.replica_grouping(group_size=tp),
        output_shape="new_axis",
    )
    next_token_logits = next_token_logits.transpose((1, 0, 2)).reshape_(
        (config.execution.micro_batch_size, config.model.embedding.vocab_size)
    )

    # (mb_size,)
    return ops.argmax(next_token_logits, dim=1)


class GPTJLMHeadTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        """
        Language model head for GPTJ, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        """
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # identical
        self.ln_f = LayerNorm()
        shard_size = ceil(self.config.model.embedding.vocab_size / tp)
        layer_kwargs = {}
        if config.execution.group_quantise_weights > 0:
            layer = LinearGQ
            layer_kwargs["group_size"] = config.execution.group_quantise_weights
            layer_kwargs["dim"] = 0
        else:
            layer = Linear
        self.head = layer(shard_size, replica_grouping=self.replica_grouping, **layer_kwargs)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        # sharded
        x = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        logits = self.head(x)
        return logits

    @staticmethod
    def hf_mapping(config: GPTJConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        head_weight = shard(to_numpy(hf_model.lm_head.weight.data.T, dtype), n_shards, axis=-1)

        if config.execution.group_quantise_weights:
            # Output dimension (vocab size) is not divisible by 4, so force to be input dimension
            gqdim = 0
            layer_name = variables.head.weight_compressed.name.replace(".head.weight_compressed", "")
            logging.info(f"Quantizing {layer_name} weights")

            # shift group_quantise_dim to account for TP sharding dim
            head_weight = group_quantize_compress_numpy(
                head_weight,
                config.execution.group_quantise_weights,
                gqdim + 1,
            )
            head_state_dict = dict(
                zip(
                    (
                        variables.head.weight_compressed,
                        variables.head.weight_decompression_scale,
                        variables.head.weight_decompression_bias,
                    ),
                    head_weight,
                ),
            )
        else:
            head_state_dict = {variables.head.weight: head_weight}

        weights = {
            **head_state_dict,
            variables.head.bias: shard(to_numpy(hf_model.lm_head.bias.data, dtype), n_shards, axis=-1),
            variables.ln_f.weight: to_numpy(hf_model.transformer.ln_f.weight.data, dtype),
            variables.ln_f.bias: to_numpy(hf_model.transformer.ln_f.bias.data, dtype),
        }
        return weights

    @staticmethod
    def to_hf(config: GPTJConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, "torch.Tensor"]:
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotFoundError("PyTorch is not installed.")

        state_dict = {}
        state_dict["lm_head.weight"] = torch.tensor(
            np.concatenate(variables_data.head.weight.transpose(0, 2, 1), axis=0),
            dtype=config.torch_dtype,
        )
        state_dict["lm_head.bias"] = torch.tensor(
            np.concatenate(variables_data.head.bias, axis=0), dtype=config.torch_dtype
        )
        state_dict["transformer.ln_f.weight"] = torch.tensor(variables_data.ln_f.weight, dtype=config.torch_dtype)
        state_dict["transformer.ln_f.bias"] = torch.tensor(variables_data.ln_f.bias, dtype=config.torch_dtype)

        return state_dict


class GPTJLMHeadModelTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        """GPTJ model with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = GPTJModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = GPTJLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:
        x = self.transformer(input_ids)
        x = self.lm_head(x)

        return x

    @staticmethod
    def hf_mapping(config: GPTJConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        weights = GPTJModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTJLMHeadTP.hf_mapping(config, variables.lm_head, hf_model))

        return weights

    @staticmethod
    def to_hf(variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        state_dict = {
            "transformer." + k: v
            for k, v in GPTJModelTP.to_hf(variables_data.transformer, hf_model.transformer, layer_norm=False).items()
        }
        state_dict.update(GPTJLMHeadTP.to_hf(hf_model.config, variables_data.lm_head, hf_model.lm_head))
        return state_dict


class GPTJLMHeadLossAndGradTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

    def build(self, x: popxl.Tensor, labels: popxl.Tensor):
        fwd_facts, fwd_graph = GPTJLMHeadTP(self.config).create_graph(x)
        ts = self.add_variable_inputs("fwd", fwd_facts)

        loss_graph = GraphWithNamedArgs(
            fwd_graph.graph._ir.create_graph(
                cross_entropy_sharded_loss,
                fwd_graph.graph.outputs[0],
                labels,
                reduction="mean",
                replica_grouping=self.replica_grouping.transpose(),
            )
        )

        required_grads = [fwd_graph.graph.inputs[0]]
        accums = list(fwd_graph.args.tensors)
        replica_groupings = fwd_facts.replica_groupings.copy()
        replica_groupings.insert("word_embedding", self.replica_grouping)

        bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(
            fwd_graph,
            tensors_to_accumulate_grads=accums,
            grads_required=required_grads,
            replica_groupings=replica_groupings,
        )
        loss_bwd = addons.transforms.autodiff(loss_graph, grads_required=(loss_graph.graph.inputs[0],))

        fwd_info = fwd_graph.bind(ts).call_with_info(x)

        logits = fwd_info.parent_output(0)
        loss_fwd_info = loss_graph.call_with_info(logits, labels)
        loss = loss_fwd_info.parent_output(0)

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        (dx,) = loss_bwd.call(loss_scaling, args=loss_bwd.grad_graph_info.inputs_dict(loss_fwd_info))

        bwd_weights = self.add_variable_inputs("bwd", bwd_facts.copy())

        (dx,) = bwd_graph.bind(bwd_weights).call(dx, args=bwd_graph.grad_graph_info.inputs_dict(fwd_info))

        return loss, dx

    @staticmethod
    def get_offsets(config: GPTJConfig):
        n_shards = config.execution.tensor_parallel
        vocab_shard_size = ceil(config.model.embedding.vocab_size / config.execution.tensor_parallel)
        return np.arange(n_shards * vocab_shard_size, step=vocab_shard_size)

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: GPTJConfig):
        return data - (i * ceil(config.model.embedding.vocab_size / config.execution.tensor_parallel))
