# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import tensorflow as tf
from tensorflow.python.ipu.ops import embedding_ops
from transformers.models.bert.modeling_tf_bert import shape_list, TFBertEmbeddings


class IpuTFBertEmbeddings(TFBertEmbeddings):
    """Modified IpuTFBertEmbeddings object that contains a serialized
    input embedding to improve memory layout. This is used in
    conjunction with the IpuTFBertLMPredictionHead which uses the input
    embeddings weights in a serialized matmul. For best results, the
    serialization factor should be the same in both cases."""

    def __init__(self, config, serialization_factor=1, **kwargs):
        super().__init__(config, **kwargs)
        self.serialization_factor = serialization_factor

    def build(self, input_shape: tf.TensorShape):
        # Build these layers here so they are accessible when copying
        # the weights
        # Input_shape is shape: [batch_size, sequence_length, hidden_size]
        self.LayerNorm.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            inputs_embeds = embedding_ops.embedding_lookup(
                self.weight, ids=input_ids, name="word_embedding_lookup", serialization_factor=self.serialization_factor
            )

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        position_embeds = embedding_ops.embedding_lookup(
            self.position_embeddings, ids=position_ids, name="position_embeddings_lookup"
        )
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))

        token_type_embeds = embedding_ops.embedding_lookup(
            self.token_type_embeddings, ids=token_type_ids, name="token_type_embeddings_lookup"
        )

        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
