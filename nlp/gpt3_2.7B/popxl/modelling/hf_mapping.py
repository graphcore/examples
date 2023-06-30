# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np
from scipy.stats import truncnorm
import torch

from transformers.models.gpt2 import GPT2Model as HFModel
from transformers.models.gpt2 import GPT2LMHeadModel as HFLMHeadModel

import popxl
from modelling.mnli import GPTMnliLossHead
from popxl_addons import TaskSession

from config import GPTConfig
from modelling.gpt_model import GPTModelTP
from modelling.gpt_lm import GPTLMHeadModelTP, GPTLMHeadTP, GPTLMHeadModelLossTP
from popxl_addons.utils import WeightsDict


def hf_mapping_lm_TP(config: GPTConfig, session: TaskSession, hf_model: HFLMHeadModel) -> WeightsDict:
    """Mapping used for pretraining.

    Session naming for gpt with lm head variables:
        transformer
            embeddings
                word
                positional
            decoder[i]
                attention
                    ln_1 (moved inside attention, in HF is outside)
                    heads
                        qkv
                    output
                feed_forward
                    ln_2 (moved inside ff, in HF is outside)
                    intermediate
                    output
            ln_f
        lm_head (linear layer - only if not tied)
    """
    weights = GPTLMHeadModelLossTP.hf_mapping(config, session.state, hf_model)
    return weights


def hf_mapping_lm_gen_inference_TP(config: GPTConfig, session: TaskSession, hf_model: HFLMHeadModel) -> WeightsDict:
    """Mapping used for pretraining.

    Session naming for gpt with lm head variables:
        transformer
            embeddings
                word
                positional
            decoder[i]
                attention
                    ln_1 (moved inside attention, in HF is outside)
                    heads
                        qkv
                    output
                feed_forward
                    ln_2 (moved inside ff, in HF is outside)
                    intermediate
                    output
            ln_f
        lm_head (linear layer - only if not tied)
    """
    weights = GPTLMHeadModelTP.hf_mapping(config, session.state, hf_model)
    return weights


def hf_mapping_lm_to_class_inference_TP(config: GPTConfig, session: TaskSession, hf_model: HFModel) -> WeightsDict:
    """Mapping used to convert a language model to Classification inference (score head is not included)

    Session naming for gpt with lm head variables:
        transformer
            embeddings
                word
                positional
            decoder[i]
                attention
                    ln_1 (moved inside attention, in HF is outside)
                    heads
                        qkv
                    output
                feed_forward
                    ln_2 (moved inside ff, in HF is outside)
                    intermediate
                    output
            ln_f
        lm_head (linear layer - only if not tied)
    """
    variables = session.state
    # Initialise new tokens: PAD, SEP CLS
    new_tokens = 3
    assert hf_model.transformer.wte.weight.shape[0] + new_tokens == config.model.embedding.vocab_size
    initial_weights = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(new_tokens, config.model.hidden_size)))
    hf_model.transformer.wte.weight.data = torch.concat([hf_model.transformer.wte.weight.data, initial_weights], dim=0)

    # WeightDict
    weights = GPTModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
    weights.update(GPTMnliLossHead.hf_mapping(config, variables, hf_model, include_score=False))
    return weights
