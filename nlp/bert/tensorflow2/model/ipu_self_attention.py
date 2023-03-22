# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import math
from typing import Tuple

import tensorflow as tf
from transformers.modeling_tf_utils import get_initializer
from transformers.models.bert.modeling_tf_bert import BertConfig, shape_list


class IpuTFBertSelfAttention(tf.keras.layers.Layer):
    """Modified TFBertSelfAttention object with the options of using merged QKV or disabled biases to improve memory."""

    def __init__(self, config: BertConfig, use_qkv_bias=False, use_qkv_split=False, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.use_qkv_bias = use_qkv_bias
        self.use_qkv_split = use_qkv_split

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def build(self, input_shape: tf.TensorShape):
        # Build these layers here so they are accessible when copying
        # the weights
        if not self.use_qkv_split:
            with tf.name_scope("qkv"):
                self.qkv_weight = self.add_weight(
                    name="qkv_weight",
                    shape=[self.all_head_size, self.all_head_size * 3],
                    initializer=get_initializer(self.config.initializer_range),
                )
        else:
            self.query = tf.keras.layers.Dense(
                units=self.all_head_size,
                kernel_initializer=get_initializer(self.config.initializer_range),
                use_bias=self.use_qkv_bias,
                name="query",
            )
            self.key = tf.keras.layers.Dense(
                units=self.all_head_size,
                kernel_initializer=get_initializer(self.config.initializer_range),
                use_bias=self.use_qkv_bias,
                name="key",
            )
            self.value = tf.keras.layers.Dense(
                units=self.all_head_size,
                kernel_initializer=get_initializer(self.config.initializer_range),
                use_bias=self.use_qkv_bias,
                name="value",
            )
            self.query.build(input_shape)
            self.key.build(input_shape)
            self.value.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        #  test
        if encoder_hidden_states is not None:
            raise RuntimeError("encoder_hidden_states is not supported")
        elif encoder_attention_mask is not None:
            raise RuntimeError("encoder_attention_mask is not supported")
        if not self.use_qkv_split:
            qkv_layer = tf.matmul(hidden_states, self.qkv_weight)
            query_layer, key_layer, value_layer = [
                self.transpose_for_scores(layer, batch_size)
                for layer in tf.split(qkv_layer, num_or_size_splits=3, axis=-1)
            ]
        else:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)
            query_layer = self.transpose_for_scores(self.query(inputs=hidden_states), batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.multiply(attention_scores, 1.0 / dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs
