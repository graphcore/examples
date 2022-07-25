# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np

from transformers.models.bert import BertModel as BertModelHF
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention

import popxl
from popxl_addons import TaskSession

from config import BertConfig
from modelling.attention import SelfAttention
from modelling.embedding import BertEmbeddings
from modelling.feed_forward import FeedForward


def hf_mapping(config: BertConfig, session: TaskSession, pretrained: BertModelHF) -> Dict[popxl.Tensor, np.ndarray]:
    weights = BertEmbeddings.hf_mapping(config, session.model.embeddings, pretrained.embeddings)

    for i in range(config.model.layers):
        # XL NamedTensor layers
        layer = session.model.layer[i]
        attn = layer.attention
        ff = layer.feed_forward

        # HF layers
        attn_hf: BertAttention = pretrained.encoder.layer[i].attention  # type: ignore
        intermediate_hf: BertIntermediate = pretrained.encoder.layer[i].intermediate  # type: ignore
        output_hf: BertOutput = pretrained.encoder.layer[i].output  # type: ignore

        weights.update(SelfAttention.hf_mapping(config, attn, attn_hf))
        weights.update(FeedForward.hf_mapping(config, ff, intermediate_hf, output_hf))

    return weights
