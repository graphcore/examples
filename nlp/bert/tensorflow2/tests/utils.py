# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers.models.bert.modeling_tf_bert import TFBertEmbeddings, TFBertSelfAttention
from transformers.modeling_tf_utils import get_initializer


def create_sample(batch_size, seq_length, input_ids_only=False):
    input_ids = np.random.randint(0, high=1000, size=(batch_size, seq_length), dtype=np.int32)
    if input_ids_only:
        token_type_ids = None
    else:
        token_type_ids = np.random.randint(0, high=2, size=(batch_size, seq_length), dtype=np.int32)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids}


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, bert_config, embeddings_class):
        super(EmbeddingLayer, self).__init__()
        self.embedding = embeddings_class(bert_config, name="embeddings")

    def call(self, inputs):
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        return self.embedding(input_ids, None, token_type_ids)


class EmbeddingModel(tf.keras.Model):
    def __init__(self, bert_config, embeddings_class):
        super(EmbeddingModel, self).__init__()
        self.embedding_layer = EmbeddingLayer(bert_config, embeddings_class)

    def call(self, inputs):
        return self.embedding_layer(inputs)


class LMPredictionHeadModel(tf.keras.Model):
    def __init__(self, bert_config, lm_prediction_head_class):
        super(LMPredictionHeadModel, self).__init__()
        self.embedding = TFBertEmbeddings(bert_config)
        self.embedding.weight = self.embedding.add_weight(
            name="weight",
            shape=[bert_config.vocab_size, bert_config.hidden_size],
            initializer=get_initializer(self.embedding.initializer_range),
        )
        self.lm_prediction_head = lm_prediction_head_class(bert_config, self.embedding)

    def call(self, inputs):
        return self.lm_prediction_head(inputs)


class TFBertSelfOutputModel(tf.keras.Model):
    def __init__(self, bert_config, self_output_class, batch_size, len_seq):
        super(TFBertSelfOutputModel, self).__init__()
        self.attention = TFBertSelfAttention(bert_config)
        self.attention.weight = self.attention.add_weight(
            name="weight",
            shape=[batch_size, len_seq, bert_config.hidden_size],
            initializer=get_initializer(bert_config.initializer_range),
        )
        self.self_output = self_output_class(bert_config)

    def call(self, inputs):
        return self.self_output(inputs, self.attention.weight)


class SelfAttentionModel(tf.keras.Model):
    def __init__(self, bert_config, self_attention_class):
        super(SelfAttentionModel, self).__init__()
        self.self_attention = self_attention_class(bert_config)

    def call(self, inputs):
        output = self.self_attention(
            inputs,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        )
        return output


def get_app_root_dir():
    return Path(__file__).parent.parent.resolve()
