# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np

from transformers.models.gpt_neo import GPTNeoForCausalLM as HFLMHeadModel
from transformers.models.gpt_neo import GPTNeoModel as HFModel

from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF

import popxl
from popxl_addons import TaskSession

from configs import GPTJConfig
from modelling.gptj.gptj_model import GPTJModelTP


def finetuneanon_mapping_lm_tp(
    config: GPTJConfig, session: TaskSession, pretrained: HFLMHeadModel
) -> Dict[popxl.Tensor, np.ndarray]:
    weights = GPTJLMHeadModelTP.finetuneanon_mapping(config, session.state.fwd, pretrained)
    return weights


def finetuneanon_mapping_tp(
    config: GPTJConfig, session: TaskSession, pretrained: HFModel
) -> Dict[popxl.Tensor, np.ndarray]:
    weights = GPTJModelTP.finetuneanon_mapping(config, session.state.fwd, pretrained)
    return weights
