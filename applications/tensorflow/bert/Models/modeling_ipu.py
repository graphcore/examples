# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd.
# Copyright 2018 The Google AI Language Team Authors.
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

"""The main BERT model and related functions."""

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
from functools import partial

from ipu_utils import function_decorator


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 hidden_layers_per_stage=1,
                 max_predictions_per_seq=20,
                 use_attention_projection_bias=True,
                 use_cls_layer=False,
                 use_qkv_bias=False,
                 ipu_function=True,
                 logits_matmul_serialization_factor=6,
                 dtype=np.float32):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
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
            hidden_layers_per_stage: Number of hidden layers putting on the same pipeline stage.
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
        self.hidden_act = get_activation(hidden_act)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.dtype = dtype
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.max_predictions_per_seq = max_predictions_per_seq
        self.ipu_function = ipu_function
        self.use_attention_projection_bias = use_attention_projection_bias
        self.use_cls_layer = use_cls_layer
        self.use_qkv_bias = use_qkv_bias
        self.logits_matmul_serialization_factor = logits_matmul_serialization_factor

        # The intermediate size is automatically computed
        self.intermediate_size = 4*self.hidden_size

        # Make hidden_layers_per_stage a list
        if not isinstance(hidden_layers_per_stage, (list, tuple)):
            num_stages = num_hidden_layers // hidden_layers_per_stage
            self.hidden_layers_per_stage = [hidden_layers_per_stage] * num_stages
        else:
            self.hidden_layers_per_stage = hidden_layers_per_stage

        assert self.num_hidden_layers == sum(self.hidden_layers_per_stage)

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

    @classmethod
    def from_json_file_for_squad(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings.
            scope: (optional) variable scope. Defaults to "bert".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        self.bert_config = config
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

        return ipu.ops.embedding_ops.embedding_lookup(embedding_table, input_ids, serialization_factor=num_splits)
    """
    pipeline stages of embedding_lookup
    wrapper of embedding_postprocessor
    """

    def embedding_lookup_stage(self,
                               learning_rate,
                               input_ids,
                               input_position,
                               segment_ids,
                               mask_padding_index,
                               seq_padding_index,
                               masked_labels,
                               masked_lm_weights,
                               next_sentence_labels):
        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                output = self.embedding(input_ids, self.bert_config.vocab_size, name="word_embeddings", num_splits=self.bert_config.logits_matmul_serialization_factor)
                masked_lm_weights = tf.cast(masked_lm_weights, dtype=self.bert_config.dtype)
                return (learning_rate,
                        output,
                        input_ids,
                        input_position,
                        segment_ids,
                        mask_padding_index,
                        seq_padding_index,
                        masked_labels,
                        next_sentence_labels,
                        masked_lm_weights)

    """
    pipeline stages of embedding_postprocessor
    wrapper of embedding_postprocessor
    """

    def embedding_postprocessor_stage(self,
                                      learning_rate,
                                      embedding_output,
                                      input_ids,
                                      input_position,
                                      segment_ids,
                                      mask_padding_index,
                                      seq_padding_index,
                                      masked_labels,
                                      next_sentence_labels,
                                      masked_lm_weights):
        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                pos_out = self.embedding(input_position, self.bert_config.max_position_embeddings,
                                         name="position_embeddings")
                seg_onehot = tf.one_hot(segment_ids, depth=self.bert_config.type_vocab_size,
                                        dtype=self.bert_config.dtype)
                seg_weights = tf.get_variable(
                    dtype=self.bert_config.dtype,
                    name="token_type_embeddings",
                    shape=[self.bert_config.type_vocab_size, self.bert_config.hidden_size],
                    initializer=create_initializer(self.bert_config.initializer_range),
                    trainable=True)
                seg_out = tf.matmul(seg_onehot, seg_weights)
                output_tensor = tf.add(embedding_output, pos_out)
                output_tensor = tf.add(output_tensor, seg_out)
                output_tensor = layer_norm_and_dropout(output_tensor, self.bert_config.hidden_dropout_prob)
                attention_mask = attention_remask(mask_padding_index, int(output_tensor.shape[-2]),
                                                  seq_padding_index, self.bert_config.dtype)
                return (learning_rate,
                        output_tensor,
                        attention_mask,
                        masked_labels,
                        next_sentence_labels,
                        masked_lm_weights)

    def self_attention_ff_stage(self,
                                layer_idx,
                                learning_rate,
                                input_tensor,
                                attention_mask,
                                masked_labels,
                                next_sentence_labels,
                                masked_lm_weights):

        @function_decorator(use_ipu_function=self.bert_config.ipu_function)
        def inner_self_attention_ff_stage(input_tensor, attention_mask):
            with tf.variable_scope("bert"):
                with tf.variable_scope("encoder"):
                    with tf.variable_scope("layer_%d" % layer_idx):
                        with tf.variable_scope("attention"):
                            attention_heads = []
                            with tf.variable_scope("self"):
                                attention_head = attention_layer(
                                    from_tensor=input_tensor,
                                    to_tensor=input_tensor,
                                    attention_mask=attention_mask,
                                    num_attention_heads=self.bert_config.num_attention_heads,
                                    size_per_head=self.bert_config.attention_head_size,
                                    attention_probs_dropout_prob=self.bert_config.attention_probs_dropout_prob,
                                    initializer_range=self.bert_config.initializer_range,
                                    do_return_2d_tensor=True,
                                    batch_size=input_tensor.shape[0],
                                    from_seq_length=input_tensor.shape[1],
                                    to_seq_length=input_tensor.shape[1],
                                    use_qkv_bias=self.bert_config.use_qkv_bias,
                                    dtype=self.bert_config.dtype)
                                attention_heads.append(attention_head)

                            attention_output = None
                            if len(attention_heads) == 1:
                                attention_output = attention_heads[0]
                            else:
                                # In the case where we have other sequences, we just concatenate
                                # them to the self-attention head before the projection.
                                attention_output = tf.concat(attention_heads, axis=-1)

                                # Run a linear projection of `hidden_size` then add a residual
                                # with `layer_input`.
                            with tf.variable_scope("projection"):
                                attention_output = tf.layers.dense(
                                    attention_output,
                                    self.bert_config.hidden_size,
                                    kernel_initializer=create_initializer(self.bert_config.initializer_range),
                                    use_bias=self.bert_config.use_attention_projection_bias)
                                attention_output = tf.reshape(
                                    attention_output,
                                    [input_tensor.shape[0] * input_tensor.shape[1] * self.bert_config.hidden_size])
                                attention_output = dropout(attention_output, self.bert_config.hidden_dropout_prob)
                                attention_output = attention_output + \
                                    tf.reshape(input_tensor, [input_tensor.shape[0] *
                                                              input_tensor.shape[1] * self.bert_config.hidden_size])
                                attention_output = tf.reshape(
                                    attention_output,
                                    [input_tensor.shape[0], input_tensor.shape[1], self.bert_config.hidden_size])
                                attention_output = layer_norm(attention_output)

                        # The activation is only applied to the "intermediate" hidden layer.
                        with tf.variable_scope("intermediate"):
                            intermediate_output = tf.layers.dense(
                                attention_output,
                                self.bert_config.intermediate_size,
                                activation=gelu,
                                kernel_initializer=create_initializer(self.bert_config.initializer_range))

                        # Down-project back to `hidden_size` then add the residual.
                        with tf.variable_scope("output"):
                            layer_output = tf.layers.dense(
                                intermediate_output,
                                self.bert_config.hidden_size,
                                kernel_initializer=create_initializer(self.bert_config.initializer_range))
                            layer_output = tf.reshape(
                                layer_output,
                                [input_tensor.shape[0] * input_tensor.shape[1] * self.bert_config.hidden_size])
                            layer_output = dropout(layer_output, self.bert_config.hidden_dropout_prob)
                            layer_output = layer_output + \
                                tf.reshape(attention_output, [input_tensor.shape[0] *
                                                              input_tensor.shape[1] * self.bert_config.hidden_size])
                            layer_output = tf.reshape(
                                layer_output,
                                [input_tensor.shape[0], input_tensor.shape[1], self.bert_config.hidden_size])
                            layer_output = layer_norm(layer_output)

                return layer_output

        layer_output = inner_self_attention_ff_stage(input_tensor, attention_mask)
        # Added the line to set a recomputation checkpoint
        layer_output = ipu.pipelining_ops.recomputation_checkpoint(layer_output)

        if layer_idx == self.bert_config.num_hidden_layers - 1:
            layer_output = reshape_from_matrix(layer_output, input_tensor.shape)
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                # After remask, [CLS] token locate at first position,
                # then follow `max_predictions_per_seq` [MASK]. If not have much [MASK],
                # padding it until to `max_predictions_per_seq`. So [MASK] tokens should locate in [1, max_predictions_per_seq+1].
                cls_position = self.bert_config.max_predictions_per_seq
                first_token_tensor = tf.squeeze(
                    layer_output[:, cls_position:cls_position+1, :], axis=1)   # [batch_size, hidden_size]
                print(f"first_token_sensor: {first_token_tensor}")
                pooled_output = tf.layers.dense(
                    first_token_tensor,
                    self.bert_config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(self.bert_config.initializer_range))
                # substract the [MASK] tokens
                layer_output = tf.slice(layer_output, [0, 0, 0], [-1, self.bert_config.max_predictions_per_seq, -1])
                return (learning_rate,
                        layer_output,
                        pooled_output,
                        masked_labels,
                        next_sentence_labels,
                        masked_lm_weights)

        return (learning_rate, layer_output, attention_mask, masked_labels, next_sentence_labels, masked_lm_weights)

    def embeddings_and_hidden_layers_stage(self, num_hidden_layer_in_stage, first_layer_idx, learning_rate, **inputs):
        """An IPU pipeline stage that combines word embeddings, positional embeddings and
        `num_hidden_layer_in_stage` of Transformer hidden layers in a single stage
        Args:
            num_hidden_layer_in_stage: Number of hidden layers of Transformer encoder
                in this stage of the pipeline
            first_layer_idx: Index of the first hidden layer, in the entire network.
        """
        word_embedding_outputs = self.embedding_lookup_stage(learning_rate, **inputs)
        embeddings_outputs = self.embedding_postprocessor_stage(*word_embedding_outputs)
        outputs = partial(self.multi_hidden_layers_stage, num_hidden_layer_in_stage,
                          first_layer_idx)(*embeddings_outputs)
        return outputs

    def multi_hidden_layers_stage(self,
                                  first_layer_idx,
                                  num_hidden_layers,
                                  learning_rate,
                                  input_tensor,
                                  attention_mask,
                                  masked_labels,
                                  next_sentence_labels,
                                  masked_lm_weights):
        """An IPU pipeline stage that combines multiple hidden layers together"""

        for layer_idx in range(first_layer_idx, first_layer_idx + num_hidden_layers):
            learning_rate, input_tensor, attention_mask, masked_labels, next_sentence_labels, masked_lm_weights = \
                self.self_attention_ff_stage(layer_idx, learning_rate, input_tensor, attention_mask, masked_labels,
                                             next_sentence_labels, masked_lm_weights)

        # Last layer is pooled
        if first_layer_idx + num_hidden_layers == self.bert_config.num_hidden_layers:
            pooled_output = attention_mask
            return (learning_rate,
                    input_tensor,
                    pooled_output,
                    masked_labels,
                    next_sentence_labels,
                    masked_lm_weights)
        else:
            return (learning_rate,
                    input_tensor,
                    attention_mask,
                    masked_labels,
                    next_sentence_labels,
                    masked_lm_weights)

    def mlm_head(self, input_tensor):
        """Slice out the masked tokens and do a linear projection."""
        with tf.variable_scope("cls/predictions"):
            if self.bert_config.use_cls_layer:
                # We apply one more non-linear transformation before the output layer.
                # This matrix is not used after pre-training.
                with tf.variable_scope("transform"):
                    input_tensor = tf.layers.dense(
                        input_tensor,
                        units=self.bert_config.hidden_size,
                        activation=get_activation(self.bert_config.hidden_act),
                        kernel_initializer=create_initializer(
                            self.bert_config.initializer_range))
                    input_tensor = layer_norm(input_tensor)

            logits = ipu.math_ops.serialized_matmul(
                input_tensor,
                self.embedding_table,
                serialization_factor=self.bert_config.logits_matmul_serialization_factor,
                serialization_dimension="b_rows",
                transpose_b=True
            )
        return logits

    def nsp_head(self, pooled_output):
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

            logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
        return logits


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

    # output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    output = ipu.ops.rand_ops.dropout(input_tensor, rate=dropout_prob)
    return output


def layer_norm(input_tensor, name='GroupNorm'):
    """Run layer normalization on the last dimension of the tensor."""

    x_reshaped = tf.reshape(input_tensor, (-1, input_tensor.shape[-1]))
    y = ipu.normalization_ops.group_norm(x_reshaped, groups=1, epsilon=0.001, scope=name)
    return tf.reshape(y, input_tensor.shape)
    # return tf.contrib.layers.layer_norm(
    #         inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope="GroupNorm")


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
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
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

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

    return mask


def attention_remask(mask_padding_index, seq_length, seq_padding_index, data_dtype):
    batch_size = int(mask_padding_index.shape[0])
    num_masked_inputs = math.ceil(seq_length*0.15)
    base_value = np.arange(seq_length)
    base = tf.constant(base_value, dtype=tf.int32)
    # Mask tokens mask
    mmask = tf.less(base, mask_padding_index)
    # No constexpr for greater. Create as const instead
    _mask = tf.constant(np.greater_equal(base_value, num_masked_inputs), np.bool)
    mmask = tf.logical_or(mmask, _mask)
    # Sequence mask
    smask = tf.less(base, seq_padding_index)
    final_mask = tf.logical_and(mmask, smask)
    final_mask = tf.reshape(final_mask, [batch_size, 1, 1, seq_length])

    return final_mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_qkv_bias=False,
                    dtype=tf.float32):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    # `qkv_layer` = [B*F, 3N*H]
    qkv_weight = tf.get_variable(dtype=dtype,
                                 name="qkv_weight",
                                 shape=[num_attention_heads*size_per_head, 3*num_attention_heads*size_per_head],
                                 initializer=create_initializer(initializer_range),
                                 trainable=True)
    qkv = tf.matmul(from_tensor_2d, qkv_weight)
    if use_qkv_bias:
        qkv_bias = tf.get_variable(
            dtype=self.bert_config.dtype,
            name="qkv_bias",
            shape=[3*num_attention_heads*size_per_head],
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        qkv = tf.nn.bias_add(qkv, qkv_bias)
    # split first then transpose to [B, N, F, H]
    query_layer, key_layer, value_layer = [
        transpose_for_scores(layer, int(batch_size), int(num_attention_heads), int(from_seq_length), int(size_per_head))
        for layer in tf.split(qkv, [int(num_attention_heads*size_per_head)]*3, axis=-1, name='qkv_split')
    ]

    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, 1, F]
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # # Normalize the attention scores to probabilities.
    # # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, int(num_attention_heads * size_per_head)])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, int(num_attention_heads * size_per_head)])

    return context_layer


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
