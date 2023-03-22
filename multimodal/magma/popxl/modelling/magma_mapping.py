# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np
from torch import nn

from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

import popxl
from popxl_addons import TaskSession

from configs import MagmaConfig, GPTJConfig, CONFIG_DIR
from modelling.image_prefix import ImagePrefix
from modelling.gptj.gptj_lm import GPTJLMHeadModelTP

from magma.image_encoders import clip_encoder
from magma.magma import Magma
import os


def load_magma(path, config: MagmaConfig, check_config: bool = True) -> nn.Module:
    """
    Loads magma checkpoint.
    """
    model = Magma.from_checkpoint(
        config_path=os.path.join(CONFIG_DIR, "MAGMA_v1.yml"),
        checkpoint_path="./mp_rank_00_model_states.pt",
        device="cpu",
    )
    if config.visual.precision == "float16":
        model.image_prefix.half()
    if config.transformer.precision == "float16":
        model.lm.half()

    if check_config:
        finetuneanon_lm_config_check(config.transformer, model.lm.config)

    return model


def magma_mapping(config: MagmaConfig, session: TaskSession, magma: nn.Module) -> Dict[popxl.Tensor, np.ndarray]:
    weights = ImagePrefix.magma_mapping(magma.image_prefix, config, session.state.fwd.image_prefix)
    weights.update(GPTJLMHeadModelTP.finetuneanon_mapping(config.transformer, session.state.fwd, magma.lm))
    return weights


def finetuneanon_lm_config_check(config: GPTJConfig, finetuneanon_config: GPTNeoConfig):
    """
    Compare a GPTJConfig with a finetuneanon GPTNeoConfig config and ensure they match.
    Required if loading a pre-trained model
    """
    if finetuneanon_config.jax == False:
        raise ValueError(
            "GPTNeo model in https://github.com/finetuneanon/transformers is equivalent to gptj only with jax=True"
        )
    if finetuneanon_config.rotary == False:
        raise ValueError(
            "GPTNeo model in https://github.com/finetuneanon/transformers is equivalent to gptj only if rotary embedding is used"
        )
    for attn in finetuneanon_config.attention_layers:
        if attn != "global":
            raise ValueError(
                'GPTNeo model in https://github.com/finetuneanon/transformers is equivalent to gptj only if "global" attention is used'
            )
    attn_type = finetuneanon_config.attention_types[0][0]
    if attn_type != "global":
        raise ValueError(
            'GPTNeo model in https://github.com/finetuneanon/transformers is equivalent to gptj only if "global" attention is used'
        )

    params = [
        ("hidden_size", config.hidden_size, finetuneanon_config.hidden_size),
        ("heads", config.attention.heads, finetuneanon_config.num_heads),
        ("layers", config.layers, finetuneanon_config.num_layers),
        ("vocab_size", config.embedding.real_vocab_size, finetuneanon_config.vocab_size),
        ("rotary_dim", config.attention.rotary_dim, finetuneanon_config.rotary_dim),
    ]

    if not all(xl == hf for _, xl, hf in params):
        not_eq_str = ", ".join(f"\n`{name}` not equal, config: {xl}, hf: {hf}" for name, xl, hf in params if xl != hf)
        raise ValueError(
            f"Config does not match the GPTNeo pre-trained model from https://github.com/finetuneanon/transformers. Not matching: {not_eq_str}"
        )
