# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np

from transformers.models.llama import LlamaModel as HFModel
from transformers.models.llama import LlamaForCausalLM as HFLMHeadModel

import popxl
from popxl_addons import TaskSession

from config import LlamaConfig
from modelling.llama_model import LlamaModelTP
from modelling.llama_lm import LlamaLMHeadModelTP


def hf_mapping_lm_tp(
    config: LlamaConfig, session: TaskSession, pretrained: HFLMHeadModel
) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = LlamaLMHeadModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def hf_mapping_TP(config: LlamaConfig, session: TaskSession, pretrained: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = LlamaModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def load_lm_to_hf(session: TaskSession, hf_model: HFLMHeadModel) -> HFLMHeadModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd
    state_dict = LlamaLMHeadModelTP.to_hf(weights, hf_model)
    # check only missing keys are mask-related keys
    hf_state_keys = hf_model.state_dict().keys()
    popxl_keys = state_dict.keys()

    def should_check(k: str):
        return "attn.bias" not in k and "attn.masked_bias" not in k

    for k in hf_state_keys:
        if should_check(k) and k not in popxl_keys:
            raise KeyError(f"key {k} not found in session state")

    hf_model.load_state_dict(state_dict, strict=False)
    return hf_model


def load_to_hf(session: TaskSession, hf_model: HFModel) -> HFModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd

    state_dict = LlamaModelTP.to_hf(weights, hf_model)
    # check only missing keys are mask-related keys
    hf_state_keys = hf_model.state_dict().keys()
    popxl_keys = state_dict.keys()

    def should_check(k: str):
        return "attn.bias" not in k and "attn.masked_bias" not in k

    for k in hf_state_keys:
        if should_check(k) and k not in popxl_keys:
            raise KeyError(f"key {k} not found in session state")

    hf_model.load_state_dict(state_dict, strict=False)
    return hf_model
