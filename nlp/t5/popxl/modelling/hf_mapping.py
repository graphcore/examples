# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Dict
import numpy as np

from transformers.models.t5 import T5Model as HFModel
from transformers.models.t5 import T5ForConditionalGeneration as HFLMHeadModel

import popxl
from popxl_addons import TaskSession

from config import T5Config
from modelling.t5_model import T5ModelTP
from modelling.t5_lm import T5LMHeadModelTP


def hf_mapping_lm_tp(
    config: T5Config, session: TaskSession, pretrained: HFLMHeadModel
) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = T5LMHeadModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def hf_mapping_TP(config: T5Config, session: TaskSession, pretrained: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = T5ModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def load_lm_to_hf(session: TaskSession, hf_model: HFLMHeadModel) -> HFLMHeadModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd
    state_dict = T5LMHeadModelTP.to_hf(weights, hf_model)
    hf_model.load_state_dict(state_dict)
    return hf_model


def load_to_hf(session: TaskSession, hf_model: HFModel) -> HFModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd
    state_dict = T5ModelTP.to_hf(weights, hf_model)
    hf_model.load_state_dict(state_dict)
    return hf_model
