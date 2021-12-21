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
from tensorflow.python import ipu
from transformers.models.bert.modeling_tf_bert import (
    BertConfig,
    shape_list,
    TFBertLMPredictionHead,
)


class IpuTFBertLMPredictionHead(TFBertLMPredictionHead):
    """Modified TFBertLMPredictionHead object that contains a serialized
       matmul to improve memory."""
    def __init__(
        self,
        config: BertConfig,
        input_embeddings: tf.keras.layers.Layer,
        serialization_factor=1,
        **kwargs
    ):
        super().__init__(config, input_embeddings, **kwargs)
        self.serialization_factor = serialization_factor

    def build(self, input_shape: tf.TensorShape):
        # Build these layers here so they are accessible when copying
        # the weights
        # Input_shape is shape of hidden_states:
        # [batch_size, sequence_length, hidden_size]
        self.transform.dense.build(input_shape)
        self.transform.LayerNorm.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = ipu.math_ops.serialized_matmul(a=hidden_states,
                                                       b=self.input_embeddings.weight,
                                                       transpose_b=True,
                                                       serialization_factor=self.serialization_factor,
                                                       serialization_dimension="b_rows")
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states
