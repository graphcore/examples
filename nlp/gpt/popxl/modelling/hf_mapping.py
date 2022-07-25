# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np

from transformers.models.gpt2 import GPT2Model as HFModel
from transformers.models.gpt2 import GPT2LMHeadModel as HFLMHeadModel

import popxl
from popxl_addons import TaskSession

from config import GPTConfig
from modelling.gpt_model import GPTModel, GPTModelTP
from modelling.gpt_lm import GPTLMHeadModel, GPTLMHeadModelTP


def hf_mapping_lm(config: GPTConfig, session: TaskSession,
                  pretrained: HFLMHeadModel) -> Dict[popxl.Tensor, np.ndarray]:
    """"
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
        lm_head (only if not tied)
    """
    weights = GPTLMHeadModel.hf_mapping(config, session.model, pretrained)
    return weights


def hf_mapping_lm_tp(config: GPTConfig, session: TaskSession,
                     pretrained: HFLMHeadModel
                     ) -> Dict[popxl.Tensor, np.ndarray]:
    """"
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
        lm_head (only if not tied)
    """
    weights = GPTLMHeadModelTP.hf_mapping(config, session.model, pretrained)
    return weights


def hf_mapping(config: GPTConfig, session: TaskSession,
               pretrained: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
    """
    Session naming for gpt variables:
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
    """
    weights = GPTModel.hf_mapping(config, session.model, pretrained)
    return weights


def hf_mapping_TP(config: GPTConfig, session: TaskSession,
                  pretrained: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
    """
    Session naming for gpt variables:
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
    """
    weights = GPTModelTP.hf_mapping(config, session.model, pretrained)
    return weights
