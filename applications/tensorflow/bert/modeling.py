# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from tensorflow.python import ipu


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 max_predictions_per_seq=20,
                 use_attention_projection_bias=True,
                 use_cls_layer=False,
                 use_qkv_bias=False,
                 use_qkv_split=False,
                 task='pretraining',
                 matmul_serialize_factor=6,
                 static_mask=False,
                 compute_acc = False,
                 dtype=tf.float32):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
            max_predictions_per_seq: Number of masked tokens which need to be predicted in MLM task.
            use_attention_projection_bias: Whether to use bias in linear projection behind attention layer.
                This is for model optimization.
            use_cls_layer: Include the CLS layer in pretraining.
                This layer comes after the encoders but before the projection for the MLM loss.
            use_qkv_bias: Whether to use bias in QKV calculation of attention layer.
                This is for model optimization.
            dtype: Data type.
        """
        assert hidden_size % num_attention_heads == 0,\
            "The hidden size (%d) is not a multiple of the number of attention " \
            "heads (%d)" % (hidden_size, num_attention_heads)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.dtype = dtype
        self.attention_head_size = int(
            self.hidden_size / self.num_attention_heads)
        self.max_predictions_per_seq = max_predictions_per_seq
        self.use_attention_projection_bias = use_attention_projection_bias
        self.use_cls_layer = use_cls_layer
        self.use_qkv_bias = use_qkv_bias
        self.use_qkv_split = use_qkv_split
        self.task = task
        self.matmul_serialize_factor = matmul_serialize_factor
        self.static_mask = static_mask
        self.compute_acc = compute_acc

    @classmethod
    def from_json(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config_json = dict()
        for (key, value) in six.iteritems(json_object):
            config_json[key] = value
        return config_json

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            if key in config.__dict__:
                config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_json(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """
    BERT model ("Bidirectional Encoder Representations from Transformers").
    """

    def __init__(self, config, is_training):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        self.layer_count = 0
        self.bert_config = config
        self.is_training = is_training
        if not is_training:
            self.bert_config.hidden_dropout_prob = 0.0
            self.bert_config.attention_probs_dropout_prob = 0.0

    def embedding(self, input_ids, embedding_size, name, num_splits=1):
        shape = [embedding_size, self.bert_config.hidden_size]

        embedding_table = tf.get_variable(
            dtype=self.bert_config.dtype,
            trainable=True,
            name=name,
            shape=shape,
            initializer=create_initializer(self.bert_config.initializer_range))

        if name == "word_embeddings":
            self.embedding_table = embedding_table

        output = ipu.ops.embedding_ops.embedding_lookup(
            embedding_table, tf.reshape(input_ids, [-1]), serialization_factor=num_splits)
        return tf.reshape(output, [input_ids.shape[0], input_ids.shape[1], -1])

    def embeddings_layer(self, input_ids, input_mask, segment_ids):
        """Combine word embeddings, position embeddings and segmentation embeddings."""
        word_embeddings = self.embedding(
            input_ids,
            self.bert_config.vocab_size,
            name="word_embeddings",
            num_splits=self.bert_config.matmul_serialize_factor)
        _batch_size, _seq_len = word_embeddings.shape[:2]
        dummy_pos_index = tf.reshape(
            tf.tile(tf.range(_seq_len), [_batch_size]), [-1, _seq_len])
        position_embeddings = self.embedding(
            dummy_pos_index, self.bert_config.max_position_embeddings, name="position_embeddings")
        seg_onehot = tf.one_hot(segment_ids,
                                depth=self.bert_config.type_vocab_size,
                                dtype=self.bert_config.dtype)
        seg_weights = tf.get_variable(dtype=self.bert_config.dtype,
                                      name="token_type_embeddings",
                                      shape=[self.bert_config.type_vocab_size,
                                             self.bert_config.hidden_size],
                                      initializer=create_initializer(
                                          self.bert_config.initializer_range),
                                      trainable=True)
        segment_embeddings = tf.matmul(seg_onehot, seg_weights)

        full_embeddings = tf.add(word_embeddings, position_embeddings)
        full_embeddings = tf.add(full_embeddings, segment_embeddings)
        full_embeddings = layer_norm_and_dropout(
            full_embeddings, self.bert_config.hidden_dropout_prob)

        return full_embeddings

    def self_attention(self, input_tensor, mask=None):
        """Performs multi-headed self-attention on `input_tensor`.

        This is an implementation of multi-headed attention based on "Attention
        is all you Need". Each timestep in `input_tensor` attends to the
        corresponding sequence in `input_tensor` itself, and returns a fixed-with vector.

        This function first projects `input_tensor` into a "query" tensor and
        `input_tensor` into "key" and "value" tensors. These are (effectively) a list
        of tensors of length `num_attention_heads`, where each tensor is of shape
        [batch_size, seq_length, size_per_head].

        Then, the query and key tensors are dot-producted and scaled. These are
        softmaxed to obtain attention probabilities. The value tensors are then
        interpolated by these probabilities, then concatenated back to a single
        tensor and returned.

        In practice, the multi-headed attention are done with transposes and
        reshapes rather than actual separate tensors.

        Args:
            input_tensor: float Tensor of shape [batch_size, seq_length,
                hidden_size].
            mask: (optional) float32 Tensor of shape [batch_size,
                seq_length, seq_length]. The values should be -1000 or 0. The
                attention scores will effectively be set to -infinity for any positions in
                the mask that are 0, and will be unchanged for positions that are 1.

        Returns:
            float Tensor of shape [batch_size * seq_length,
                num_attention_heads * size_per_head]

        Raises:
            ValueError: Any of the arguments or tensor shapes are invalid.
        """
        input_shape = get_shape_list(input_tensor, expected_rank=[2, 3])
        assert len(input_shape) in [2, 3], \
            f"Input shape of attention moduler should be `[batch_size, seq_length]` or `[batch_size, seq_length, seq_length]`."
        batch_size, seq_length = input_shape[:2]

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   S = `input_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        num_attention_heads = int(self.bert_config.num_attention_heads)
        size_per_head = int(self.bert_config.attention_head_size)

        input_tensor_2d = reshape_to_matrix(input_tensor)

        # We combine the query, key and value layers to reduce memory consume.
        # `qkv_layer` = [B*S, 3N*H]    if use_qkv_split:
        if self.bert_config.use_qkv_split:
            head_shape = [num_attention_heads*size_per_head, num_attention_heads*size_per_head]
            with tf.variable_scope('query'):
                q_weight = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="kernel",
                    shape=head_shape,
                    initializer=create_initializer(self.bert_config.initializer_range),
                    trainable=True)
            with tf.variable_scope('key'):
                k_weight = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="kernel",
                    shape=head_shape,
                    initializer=create_initializer(self.bert_config.initializer_range),
                    trainable=True)
            with tf.variable_scope('value'):
                v_weight = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="kernel",
                    shape=head_shape,
                    initializer=create_initializer(self.bert_config.initializer_range),
                    trainable=True)
            qkv_weight = tf.concat([q_weight, k_weight, v_weight], axis=-1)

        else:
            with tf.variable_scope('kernel'):
                qkv_weight = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="qkv_weight",
                    shape=[num_attention_heads*size_per_head, 3*num_attention_heads*size_per_head],
                    initializer=create_initializer(self.bert_config.initializer_range),
                    trainable=True)

        @ipu.outlined_function
        def inner_attention_func():
            qkv = tf.matmul(input_tensor_2d, qkv_weight)

            if self.bert_config.use_qkv_bias:
                qkv_bias = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="qkv_bias",
                    shape=[3*num_attention_heads*size_per_head],
                    initializer=tf.zeros_initializer(),
                    trainable=True
                )
                qkv = tf.nn.bias_add(qkv, qkv_bias)
            # Split and transpose to [B, N, S, H]
            query_layer, key_layer, value_layer = [
                transpose_for_scores(layer, int(batch_size), int(
                    num_attention_heads), int(seq_length), int(size_per_head))
                for layer in tf.split(qkv, [int(num_attention_heads*size_per_head)]*3, axis=-1, name='qkv_split')
            ]

            # `attention_scores` = [B, N, S, S]
            attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
            attention_scores = tf.multiply(
                attention_scores, 1.0 / math.sqrt(float(size_per_head)))
            if mask is not None:
                # `mask` = [B, 1, 1, S]
                attention_scores = tf.add(
                    attention_scores, tf.expand_dims(mask, axis=1))

            # `attention_probs` = [B, N, S, S]
            attention_probs = tf.nn.softmax(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = dropout(
                attention_probs, self.bert_config.attention_probs_dropout_prob)

            # `context_layer` = [B, N, S, H]
            context_layer = tf.matmul(attention_probs, value_layer)

            # `context_layer` = [B, S, N, H]
            context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

            # `context_layer` = [B*S, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * seq_length, self.bert_config.hidden_size])

            return context_layer
        context_layer = inner_attention_func()
        return context_layer

    def attention_projection(self, input_tensor, attention_output):
        with tf.variable_scope("projection"):
            attention_output = dense_layer(
                attention_output,
                self.bert_config.hidden_size,
                kernel_initializer=create_initializer(
                    self.bert_config.initializer_range),
                use_bias=self.bert_config.use_attention_projection_bias)
            attention_output = tf.reshape(attention_output, input_tensor.shape)
            attention_output = dropout_residual_add_layer_norm(
                attention_output, input_tensor,
                self.bert_config.hidden_dropout_prob)
        return attention_output

    def feed_forward(self, attention_output):
        # The activation is only applied to the "intermediate" hidden layer.
        with tf.variable_scope("intermediate"):
            intermediate_output = dense_layer(
                attention_output,
                self.bert_config.intermediate_size,
                activation=gelu,
                kernel_initializer=create_initializer(self.bert_config.initializer_range))
        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
            feed_forward_output = dense_layer(
                intermediate_output,
                self.bert_config.hidden_size,
                kernel_initializer=create_initializer(self.bert_config.initializer_range))
            feed_forward_output = dropout_residual_add_layer_norm(
                feed_forward_output,
                attention_output,
                self.bert_config.hidden_dropout_prob
            )
        return feed_forward_output

    def encoder(self, input_tensor, attention_mask, masked_lm_positions=None):
        """Encoder layer."""
        original_input_shape = input_tensor.shape

        with tf.variable_scope("bert"):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("layer_%d" % self.layer_count):
                    with tf.variable_scope("attention"):
                        attention_heads = []
                        with tf.variable_scope("self"):
                            attention_head = self.self_attention(
                                input_tensor, mask=attention_mask)
                            attention_heads.append(attention_head)

                        attention_output = None
                        if len(attention_heads) == 1:
                            attention_output = attention_heads[0]
                        else:
                            # In the case where we have other sequences, we just concatenate
                            # them to the self-attention head before the projection.
                            attention_output = tf.concat(
                                attention_heads, axis=-1)

                        # Run a linear projection of `hidden_size` then add a residual
                        # with `layer_input`.
                        attention_output = self.attention_projection(
                            input_tensor, attention_output)
                    input_tensor = self.feed_forward(attention_output)

        input_tensor = ipu.pipelining_ops.recomputation_checkpoint(input_tensor)
        self.layer_count += 1
        # Reshape the last hidden layer outputs.
        if self.layer_count == self.bert_config.num_hidden_layers:
            input_tensor = reshape_from_matrix(
                input_tensor, original_input_shape)
            # Return the masked tokens tensor to avoid major changes in `multi_stage_wrapper`.
            # However this might be optimized later.
            if self.bert_config.task.lower() == 'pretraining':
                masked_tokens_tensor = self.lm_projection(
                    input_tensor, masked_lm_positions)

                return {
                    'layer_output': input_tensor,
                    'masked_tokens_tensor': masked_tokens_tensor,
                }
        return {
            "input_tensor": input_tensor
        }

    def pooler(self, input_tensor, cls_position=0):
        with tf.variable_scope("pooler"):
            # Pool out the [CLS] token.
            if self.bert_config.static_mask:
                cls_position = self.bert_config.max_predictions_per_seq
            cls_token_tensor = tf.squeeze(
                input_tensor[:, cls_position:cls_position+1, :], axis=1)  # [batch_size, hidden_size]
            pooled_output = tf.layers.dense(
                cls_token_tensor,
                self.bert_config.hidden_size,
                activation=tf.tanh,
                kernel_initializer=create_initializer(self.bert_config.initializer_range))
        return pooled_output

    def lm_projection(self, input_tensor, masked_lm_positions):
        if self.bert_config.static_mask:
            masked_tokens_tensor = tf.slice(input_tensor, [0, 0, 0], [-1, self.bert_config.max_predictions_per_seq, -1])
            masked_tokens_tensor = tf.reshape(masked_tokens_tensor, [-1, masked_tokens_tensor.shape[2]])
        else:
            masked_tokens_tensor = gather_indexes(input_tensor, masked_lm_positions)

        if self.bert_config.use_cls_layer:
            with tf.variable_scope("cls/predictions/transform"):
                masked_tokens_tensor = tf.layers.dense(
                    masked_tokens_tensor,
                    units=self.bert_config.hidden_size,
                    activation=get_activation(self.bert_config.hidden_act),
                    kernel_initializer=create_initializer(
                        self.bert_config.initializer_range))
                masked_tokens_tensor = layer_norm(masked_tokens_tensor)
        return masked_tokens_tensor

    def nsp_head(self, input_tensor):
        """Extract [CLS] tokens and do a linear projection"""
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights",
                dtype=self.bert_config.dtype,
                shape=[2, self.bert_config.hidden_size],
                initializer=create_initializer(self.bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias",
                dtype=self.bert_config.dtype,
                shape=[2],
                initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
        return logits

    def mlm_head(self, masked_tokens_tensor):
        """Slice out the masked tokens and do a linear projection."""
        with tf.variable_scope("cls/predictions"):
            logits = ipu.math_ops.serialized_matmul(masked_tokens_tensor,
                                                    self.embedding_table,
                                                    serialization_factor=self.bert_config.matmul_serialize_factor,
                                                    serialization_dimension="b_rows",
                                                    transpose_b=True)
        return {"mlm_logits": logits}

    def squad_head(self, input_tensor, dtype):
        """Take linear projection on last hidden layer output."""
        with tf.variable_scope("cls/squad"):
            input_tensor = tf.cast(input_tensor, dtype=dtype)
            batch_size, seq_length, hidden_size = input_tensor.shape

            output_weights = tf.get_variable(
                name="output_weights",
                shape=[2, self.bert_config.hidden_size],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                name="output_bias",
                shape=[2],
                dtype=dtype,
                initializer=tf.zeros_initializer())

            final_hidden_matrix = tf.reshape(input_tensor,
                                             [batch_size * seq_length, hidden_size])

            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(logits, [batch_size, seq_length, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            start_logits, end_logits = tf.unstack(logits, axis=0)
        return start_logits, end_logits

    def embedding_lookup_layer(self,
                               input_ids,
                               ):
        """
        pipeline stages of embedding_lookup
        """
        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                word_embeddings = self.embedding(
                    input_ids, self.bert_config.vocab_size,
                    name="word_embeddings",
                    num_splits=self.bert_config.matmul_serialize_factor)
                return {"word_embeddings": word_embeddings
                        }

    def embedding_postprocessor_layer(self,
                                      word_embeddings,
                                      input_ids,
                                      input_mask=None,
                                      segment_ids=None,
                                      input_position=None,
                                      mask_padding_index=None,
                                      seq_padding_index=None
                                      ):
        """
        pipeline stages of embedding_postprocessor
        """
        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                _batch_size, _seq_len = word_embeddings.shape[:2]
                if self.bert_config.static_mask:
                    position_embeddings = self.embedding(input_position, self.bert_config.max_position_embeddings, name="position_embeddings")
                else:
                    dummy_pos_index = tf.reshape(
                        tf.tile(tf.range(_seq_len), [_batch_size]), [-1, _seq_len])
                    position_embeddings = self.embedding(
                        dummy_pos_index, self.bert_config.max_position_embeddings, name="position_embeddings")

                seg_onehot = tf.one_hot(segment_ids,
                                        depth=self.bert_config.type_vocab_size,
                                        dtype=self.bert_config.dtype)
                seg_weights = tf.get_variable(dtype=self.bert_config.dtype,
                                              name="token_type_embeddings",
                                              shape=[self.bert_config.type_vocab_size,
                                                     self.bert_config.hidden_size],
                                              initializer=create_initializer(
                                                  self.bert_config.initializer_range),
                                              trainable=True)
                segment_embeddings = tf.matmul(seg_onehot, seg_weights)
                full_embeddings = tf.add(word_embeddings, position_embeddings)
                full_embeddings = tf.add(full_embeddings, segment_embeddings)
                full_embeddings = layer_norm_and_dropout(
                    full_embeddings, self.bert_config.hidden_dropout_prob)

                if self.bert_config.static_mask:
                    attention_mask = attention_static_remasking(
                        mask_padding_index, _seq_len.value, seq_padding_index,
                        self.bert_config.max_predictions_per_seq,
                        self.bert_config.dtype)
                else:
                    attention_mask = create_attention_mask_from_input_mask(
                        input_ids, input_mask, self.bert_config.dtype)

                return {
                    "input_tensor": full_embeddings,
                    "attention_mask": attention_mask
                }

    def get_next_sentence_output_layer(self,
                                       layer_output,
                                       mlm_logits,
                                       masked_lm_ids,
                                       masked_lm_weights,
                                       next_sentence_labels):
        with tf.variable_scope('bert'):
            pooled_output = self.pooler(layer_output)
        nsp_logits = self.nsp_head(pooled_output)
        # Calculate MLM loss
        with tf.variable_scope("cls/predictions"):
            log_probs = tf.nn.log_softmax(mlm_logits, axis=-1)
            label_ids = tf.reshape(masked_lm_ids, [-1])
            label_weights = tf.reshape(masked_lm_weights, [-1])
            one_hot_labels = tf.one_hot(
                tf.cast(label_ids, dtype=tf.int32), depth=self.bert_config.vocab_size, dtype=self.bert_config.dtype)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            mlm_loss = numerator / denominator
            if self.bert_config.compute_acc:
                # Calculate `mlm_acc`
                results = tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32)
                predictions = tf.cast(tf.equal(results, label_ids), dtype=tf.float16)
                predictions = tf.cast(predictions * label_weights, dtype=tf.float32)

                mlm_acc = tf.reduce_sum(predictions)
                total_attempted = tf.cast(tf.reduce_sum(label_weights), dtype=tf.float32)
                mlm_acc = mlm_acc / total_attempted
            else:
                mlm_acc = tf.get_variable('mlm_acc', initializer = -1.0, trainable = False, dtype = tf.float32)

        # Calculate NSP loss
        with tf.variable_scope("cls/seq_relationship"):
            log_probs = tf.nn.log_softmax(nsp_logits, axis=-1)
            next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
            one_hot_labels = tf.one_hot(
                next_sentence_labels, depth=2, dtype=self.bert_config.dtype)
            per_example_loss = - \
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            nsp_loss = tf.reduce_mean(per_example_loss)

            if self.bert_config.compute_acc:
                #  Calculate the `nsp_acc`
                nsp_acc = tf.reduce_mean(tf.cast(tf.equal(
                    tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32),
                    next_sentence_labels), dtype=tf.float32))
            else:
                nsp_acc = tf.get_variable('nsp_acc', initializer = -1.0, trainable = False, dtype = tf.float32)

        outfeed_mlm_loss = tf.cast(mlm_loss, dtype = tf.float32)
        outfeed_nsp_loss = tf.cast(nsp_loss, dtype = tf.float32)

        return {"mlm_loss": outfeed_mlm_loss, "nsp_loss": outfeed_nsp_loss,
                "mlm_acc": mlm_acc, "nsp_acc": nsp_acc}

    def get_loc_logic_output_layer(self,
                                   start_positions,
                                   end_positions,
                                   input_tensor,
                                   ):
        # This is the loss and accuracy for SQuAD
        dtype_loss = tf.float32
        start_logits, end_logits = self.squad_head(
            input_tensor, dtype_loss)

        if not self.is_training:
            return {'start_logits': start_logits, 'end_logits': end_logits}

        def compute_loss(logits, positions):
            seq_len = logits.shape[1]
            logits_fp32 = tf.cast(logits, dtype=dtype_loss)
            one_hot_positions = tf.one_hot(positions, depth=seq_len, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits_fp32, axis=-1)
            loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            loss = tf.cast(loss, dtype=logits.dtype)
            return loss

        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)

        total_loss = (start_loss + end_loss) / 2.0

        return {'total_loss': total_loss}


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    batch_size, seq_length, width = get_shape_list(
        sequence_tensor, expected_rank=3)

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = ipu.ops.embedding_ops.embedding_lookup(
        flat_sequence_tensor, flat_positions, serialization_factor=1)
    output_tensor = tf.reshape(output_tensor, [-1, width])
    return output_tensor


def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                         seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    return ipu.nn_ops.gelu(x)


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
            activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = []
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map.append(name_to_variable[name])
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob=None):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
            *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    # We use the IPU-specific dropout.
    output = ipu.ops.rand_ops.dropout(input_tensor, rate=dropout_prob)
    return output


def layer_norm(input_tensor, name='LayerNorm'):
    """Run layer normalization on the last dimension of the tensor."""

    x_reshaped = tf.reshape(input_tensor, (-1, input_tensor.shape[-1]))
    # We use the IPU-specific group_norm() operation.
    y = ipu.normalization_ops.group_norm(
        x_reshaped, groups=1, epsilon=0.001, scope=name)
    return tf.reshape(y, input_tensor.shape)


def layer_norm_and_dropout(input_tensor, dropout_prob, name="LayerNorm"):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def create_attention_mask_from_input_mask(from_tensor, to_mask, dtype):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size, from_seq_length = from_shape[:2]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=dtype)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    mask = (1.0 - mask) * -1000.0

    return mask


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def attention_static_remasking(mask_padding_index, seq_length,
                               seq_padding_index, num_masked_inputs,
                               data_dtype):
    """Creates the attention mask when the "static_mask" mode is used.
    In this mode the first `num_masked_inputs` tokens are always
    masked. This function handles variable number of predicted tokens
    and sequence lengths.
    """
    batch_size = int(mask_padding_index.shape[0])
    base_value = np.arange(seq_length)
    base = tf.constant(base_value, dtype=tf.int32)

    # Tokens mask
    mmask = tf.less(base, mask_padding_index)
    _mask = tf.constant(np.greater_equal(base_value, num_masked_inputs),
                        np.bool)
    mmask = tf.logical_or(mmask, _mask)

    # Sequence mask
    smask = tf.less(base, seq_padding_index)
    final_mask = tf.logical_and(mmask, smask)
    final_mask = tf.reshape(final_mask, [batch_size, 1, seq_length])

    final_mask = (1.0 - tf.cast(final_mask, data_dtype)) * -1000.0

    return final_mask


def dropout_residual_add_layer_norm(input_tensor,
                                    residual_tensor,
                                    dropout_prob):
    @ipu.outlined_function
    def inner_func():
        output = residual_tensor + dropout(input_tensor, dropout_prob)
        output = layer_norm(output)
        return output
    return inner_func()


def dense_layer(input_tensor,
                num_units,
                kernel_initializer,
                activation=None,
                use_bias=True):
    @ipu.outlined_function
    def inner_func():
        return tf.layers.dense(input_tensor,
                               num_units,
                               use_bias=use_bias,
                               activation=activation,
                               kernel_initializer=kernel_initializer)
    return inner_func()
