# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Dict, MutableMapping
import numpy as np

from transformers.models.gpt2 import GPT2LMHeadModel as HFGPT2LMHeadModel

import popxl
from popxl_addons import TaskSession
from popxl_addons.utils import WeightsDict

from config import GPTConfig
from modelling.gpt_lm import GPTLMHeadModelTP, GPTLMHeadModelTP2D


def hf_mapping_lm_tp(config: GPTConfig, session: TaskSession, pretrained: HFGPT2LMHeadModel) -> WeightsDict:
    """ "
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
    weights = GPTLMHeadModelTP.hf_mapping(config, session.state, pretrained)
    return weights


def hf_mapping_lm_tp2d(config: GPTConfig, session: TaskSession, pretrained: HFGPT2LMHeadModel) -> WeightsDict:
    """ "
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
    weights = GPTLMHeadModelTP2D.hf_mapping(config, session.state, pretrained)
    return weights
