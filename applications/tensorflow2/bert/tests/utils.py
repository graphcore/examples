# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers.models.bert.modeling_tf_bert import TFBertEmbeddings
from transformers.modeling_tf_utils import get_initializer


def create_sample(batch_size, seq_length, input_ids_only=False):
    input_ids = np.random.randint(
        0, high=1000, size=(batch_size, seq_length), dtype=np.int32)
    if input_ids_only:
        token_type_ids = None
    else:
        token_type_ids = np.random.randint(
            0, high=2, size=(batch_size, seq_length), dtype=np.int32)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids}


def get_bert_embeddings_model(embedding_layer, config, serialization_factor=None):

    class EmbeddingModel(tf.keras.Model):

        def __init__(self, bert_config):
            super(EmbeddingModel, self).__init__()
            if serialization_factor is None:
                self.embedding = embedding_layer(bert_config)
            else:
                self.embedding = embedding_layer(
                    bert_config,
                    serialization_factor=serialization_factor
                )

        def call(self, inputs):
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            return self.embedding(input_ids, None, token_type_ids)

    return EmbeddingModel(config)


def get_bert_lm_prediction_head_model(config, lm_prediction_head):

    class LMPredictionHead(tf.keras.Model):

        def __init__(self, bert_config, lm_prediction_head):
            super(LMPredictionHead, self).__init__()
            self.embedding = TFBertEmbeddings(bert_config)
            self.embedding.weight = self.embedding.add_weight(
                name="weight",
                shape=[bert_config.vocab_size, bert_config.hidden_size],
                initializer=get_initializer(self.embedding.initializer_range),
            )
            self.lm_prediction_head = lm_prediction_head(bert_config,
                                                         self.embedding)

        def call(self, inputs):
            return self.lm_prediction_head(inputs)

    return LMPredictionHead(config, lm_prediction_head)


def get_app_root_dir():
    return Path(__file__).parent.parent.resolve()
