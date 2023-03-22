# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.utils import WeightsDict
from transformers.models.bloom import BloomForCausalLM as HFBloomLMHeadModel
from transformers.models.bloom.modeling_bloom import BloomModel

import popxl
from config import BloomConfig
from modelling.bloom_model import BloomModelTP2D
from popxl import ops
from popxl.utils import to_numpy
from utils.utils import tp2d_replica_groups


def gather_logits_tp(
    config: BloomConfig,
    logits: popxl.Tensor,
    last_token_index: popxl.Tensor,
) -> popxl.Tensor:
    tp1 = config.execution.tensor_parallel_1
    rg_tp1, *_ = tp2d_replica_groups(config)

    next_token_logits = logits[last_token_index]
    next_token_logits = ops.collectives.replicated_all_gather(next_token_logits, group=rg_tp1)
    next_token_logits = next_token_logits.flatten()
    return next_token_logits


class BloomLMHeadTP2D(addons.Module):
    def __init__(self, config: BloomConfig):
        """
        Language model head for Bloom, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        No bias is applied.
        In Bloom, the weight matrix is tied to the word embedding weights. word_embedding needs to be provided as input
        and need to be sharded as well.
        Moreover, a single embedding tensor is too large to fit in a single
        transfer. It has therefore been split in two along the hidden dimension
        """
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)
        self.ln_f = LayerNormDistributed(self.rg_tp2)

    @popxl.in_sequence(True)
    def build(self, x: popxl.Tensor, embedding_1: popxl.Tensor, embedding_2: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        x1, x2 = ops.split(x, 2, axis=-1)

        # Embedding tensor is split, hence matmul must also be decomposed
        # x @ E.T == x_{:h/2} @ E_{:h/2}.T + x_{h/2:} @ E_{h/2:}.T
        logits = x1 @ embedding_1.T + x2 @ embedding_2.T
        logits = replicated_all_reduce(logits, group=self.rg_tp2)

        return logits

    @staticmethod
    def hf_mapping(config: BloomConfig, variables: NamedTensors, hf_model: BloomModel) -> WeightsDict:
        dtype = config.model.dtype
        tp2 = config.execution.tensor_parallel_2
        weights = WeightsDict(
            {
                variables.ln_f.weight: shard(to_numpy(hf_model.ln_f.weight.data, dtype), tp2, 0),
                variables.ln_f.bias: shard(to_numpy(hf_model.ln_f.bias.data, dtype), tp2, 0),
            }
        )

        return weights


class BloomLMHeadModelTP2D(addons.Module):
    def __init__(self, config: BloomConfig):
        """Bloom model with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = BloomModelTP2D(config, include_layer_norm=False)  # move layer norm to the head
        self.head = BloomLMHeadTP2D(config)

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:
        x = self.transformer(input_ids)
        x = self.head(x, self.transformer.embedding.weight_1, self.transformer.embedding.weight_2)
        return x

    @staticmethod
    def hf_mapping(config: BloomConfig, variables: NamedTensors, hf_model: HFBloomLMHeadModel) -> WeightsDict:
        weights = WeightsDict()
        weights.update(BloomModelTP2D.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False))
        weights.update(BloomLMHeadTP2D.hf_mapping(config, variables.head, hf_model.transformer))
        return weights
