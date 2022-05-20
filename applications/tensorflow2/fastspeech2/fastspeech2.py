# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Copyright 2020 The FastSpeech Authors, The HuggingFace Inc. team and Minh Nguyen (@dathudeptrai)
# Copyright 2020 The FastSpeech2 Authors and Minh Nguyen (@dathudeptrai)
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
"""
This script has been adapated from some of the original TensorSpeech/TensorFlowTTS repo found here:
[
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/models/fastspeech.py,
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/models/fastspeech2.py,
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/configs/fastspeech.py,
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/configs/fastspeech2.py
]

Main changes:
  Combine configs and models related to FastSpeech2.
  Use IPU specific/optimized layers.
  Add scripts to build functional model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow import keras


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


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
                                (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return ipu.nn_ops.swish(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
    We assume that anything that's not a string is already an activation
    function, so we just return it.

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

    activation_mappings = {
        "identity": tf.keras.layers.Activation("linear"),
        "tanh": tf.keras.layers.Activation("tanh"),
        "gelu": tf.keras.layers.Activation(gelu),
        "relu": tf.keras.layers.Activation("relu"),
        "swish": tf.keras.layers.Activation(swish),
        "gelu_new": tf.keras.layers.Activation(gelu_new),
        "mish": tf.keras.layers.Activation(mish),
    }
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_mappings[activation_string.lower()]
    if not act:
        raise ValueError("Unsupported activation: %s" %
                         activation_string.lower())
    return act


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def sincos_embedding(hidden_size, max_positional_embedding):
    position_enc = np.array([
        [pos / np.power(10000, 2.0 * (i // 2) / hidden_size)
         for i in range(hidden_size)]
        for pos in range(max_positional_embedding + 1)])

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    # pad embedding.
    position_enc[0] = 0.0
    return position_enc


SelfAttentionConfig = collections.namedtuple(
    "SelfAttentionConfig",
    [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "attention_head_size",
        "intermediate_size",
        "intermediate_kernel_size",
        "hidden_act",
        "output_attentions",
        "output_hidden_states",
        "initializer_range",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "layer_norm_eps",
        "max_position_embeddings",
        "dtype"
    ],
)


class FastSpeechConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
        self,
        vocab_size=70,
        encoder_hidden_size=384,
        encoder_num_hidden_layers=4,
        encoder_num_attention_heads=2,
        encoder_intermediate_size=1024,
        encoder_intermediate_kernel_size=3,
        encoder_hidden_act="mish",
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=1024,
        decoder_intermediate_kernel_size=3,
        decoder_hidden_act="mish",
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        max_position_embeddings=2048,
        duration_predictor_num_conv_layers=2,
        duration_predictor_filters=256,
        duration_predictor_kernel_size=3,
        num_mels=80,
        duration_predictor_dropout_probs=0.1,
        use_postnet=True,
        postnet_num_conv_layers=5,
        postnet_conv_filters=512,
        postnet_conv_kernel_size=5,
        postnet_dropout_rate=0.1,
        max_seq_length=135,
        max_wave_length=870,
        dtype=tf.float32,
        **kwargs
    ):
        """Init parameters for Fastspeech model."""
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_length = max_seq_length
        self.max_wave_length = max_wave_length
        self.dtype = dtype
        # encoder params
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_attention_head_size = int(
            encoder_hidden_size/encoder_num_attention_heads)
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_intermediate_kernel_size = encoder_intermediate_kernel_size
        self.encoder_hidden_act = encoder_hidden_act
        self.encoder_self_attention_params = SelfAttentionConfig(
            hidden_size=encoder_hidden_size,
            num_hidden_layers=encoder_num_hidden_layers,
            num_attention_heads=encoder_num_attention_heads,
            attention_head_size=self.encoder_attention_head_size,
            hidden_act=encoder_hidden_act,
            intermediate_size=encoder_intermediate_size,
            intermediate_kernel_size=encoder_intermediate_kernel_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype
        )

        # decoder params
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_attention_head_size = int(
            decoder_hidden_size/decoder_num_attention_heads)
        self.decoder_intermediate_size = decoder_intermediate_size
        self.decoder_intermediate_kernel_size = decoder_intermediate_kernel_size
        self.decoder_hidden_act = decoder_hidden_act
        self.decoder_self_attention_params = SelfAttentionConfig(
            hidden_size=decoder_hidden_size,
            num_hidden_layers=decoder_num_hidden_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_size=self.decoder_attention_head_size,
            hidden_act=decoder_hidden_act,
            intermediate_size=decoder_intermediate_size,
            intermediate_kernel_size=decoder_intermediate_kernel_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype
        )

        self.duration_predictor_dropout_probs = duration_predictor_dropout_probs
        self.duration_predictor_num_conv_layers = duration_predictor_num_conv_layers
        self.duration_predictor_filters = duration_predictor_filters
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.num_mels = num_mels

        # postnet
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet_num_conv_layers = postnet_num_conv_layers
            self.postnet_conv_filters = postnet_conv_filters
            self.postnet_conv_kernel_size = postnet_conv_kernel_size
            self.postnet_dropout_rate = postnet_dropout_rate

    @classmethod
    def from_json(cls, json_object):
        """Constructs a `FastSpeechConfig` from a Python dictionary of parameters."""
        config_json = dict()
        for (key, value) in six.iteritems(json_object):
            config_json[key] = value
        return config_json

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `FastSpeechConfig` from a Python dictionary of parameters."""
        config = FastSpeechConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            if key in config.__dict__:
                config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `FastSpeechConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_json(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FastSpeech2Config(FastSpeechConfig):
    """Initialize FastSpeech2 Config."""

    def __init__(
        self,
        variant_predictor_num_conv_layers=2,
        variant_predictor_filter=256,
        variant_predictor_kernel_size=3,
        variant_predictor_dropout_rate=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant_predictor_num_conv_layers = variant_predictor_num_conv_layers
        self.variant_predictor_kernel_size = variant_predictor_kernel_size
        self.variant_predictor_dropout_rate = variant_predictor_dropout_rate
        self.variant_predictor_filter = variant_predictor_filter


class Embedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.gather(self.embeddings, inputs)
        return outputs


class FastSpeechEmbeddings(tf.keras.layers.Layer):
    """Construct charactor/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.encoder_self_attention_params.hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings + 1,
            self.hidden_size,
            weights=[sincos_embedding(
                self.hidden_size, self.config.max_position_embeddings
            )],
            name="position_embeddings",
            trainable=False,
            dtype=self.config.dtype
        )

    def build(self, input_shape):
        """Build shared charactor/phoneme embedding layers."""
        with tf.name_scope("charactor_embeddings"):
            self.charactor_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=create_initializer(
                    initializer_range=self.initializer_range),
                trainable=True,
                dtype=self.config.dtype
            )
        super().build(input_shape)

    def call(self, input_ids, training=False):
        """Get charactor embeddings of inputs.

        Args:
            1. charactor, Tensor (int32) shape [batch_size, length].
        Returns:
            Tensor (float32) shape [batch_size, length, embedding_size].

        """
        seq_length = tf.shape(input_ids)[1]

        position_ids = tf.range(
            1, seq_length + 1, dtype=tf.int32)[tf.newaxis, :]

        # create embeddings
        inputs_embeds = tf.gather(self.charactor_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # sum embedding
        embeddings = inputs_embeds + \
            tf.cast(position_embeddings, inputs_embeds.dtype)

        return embeddings

    def resize_positional_embeddings(self, new_size):
        self.position_embeddings = tf.keras.layers.Embedding(
            new_size,
            self.hidden_size,
            weights=[sincos_embedding(self.hidden_size, new_size)],
            name="position_embeddings",
            trainable=False,
            dtype=self.config.dtype
        )


class SelfAttention(tf.keras.layers.Layer):
    """Self attention module for fastspeech."""

    def __init__(self, config, dtype=tf.float32, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * config.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=create_initializer(config.initializer_range),
            name="query",
            dtype=self.config.dtype
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=create_initializer(config.initializer_range),
            name="key",
            dtype=self.config.dtype
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=create_initializer(config.initializer_range),
            name="value",
            dtype=self.config.dtype
        )

        self.dropout = ipu.keras.layers.Dropout(
            config.attention_probs_dropout_prob, dtype=config.dtype)

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_attention_heads,
             self.config.attention_head_size),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs
        batch_size = tf.shape(hidden_states)[0]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(
            tf.shape(key_layer)[-1], attention_scores.dtype
        )  # scale attention_scores

        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:,
                                                     tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(
                extended_attention_mask, attention_scores.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        attention_probs = tf.cast(attention_probs, value_layer.dtype)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size))

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class SelfAttentionOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        kwargs["dtype"] = config.dtype
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=create_initializer(config.initializer_range),
            name="dense",
            dtype=config.dtype
        )
        self.LayerNorm = ipu.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm", dtype=config.dtype
        )
        self.dropout = ipu.keras.layers.Dropout(
            config.hidden_dropout_prob, dtype=config.dtype)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(tf.keras.layers.Layer):
    """Fastspeech attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = SelfAttention(config, name="self")
        self.dense_output = SelfAttentionOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor], training=training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=attention_output.dtype
        )
        # add attentions if we output them
        outputs = (masked_attention_output,) + self_outputs[1:]
        return outputs


class Intermediate(tf.keras.layers.Layer):
    """Intermediate representation module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            config.intermediate_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=create_initializer(config.initializer_range),
            padding="same",
            name="conv1d_1",
            dtype=config.dtype
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=create_initializer(config.initializer_range),
            padding="same",
            name="conv1d_2",
            dtype=config.dtype
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, inputs):
        """Call logic."""
        # inputs[0]: [B, S, H]
        hidden_states, attention_mask = inputs
        hidden_states = self.conv1d_1(hidden_states)
        # We use static paddings that all data
        # had been padded to max length. It will
        # have impact on Convolution calculation
        # which consider extra paddings is slightly incorrect.
        hidden_states = hidden_states * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=hidden_states.dtype
        )
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)

        masked_hidden_states = hidden_states * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=hidden_states.dtype
        )
        return masked_hidden_states


class IntermediateOutput(tf.keras.layers.Layer):
    """Output module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.LayerNorm = ipu.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm", dtype=config.dtype
        )
        self.dropout = ipu.keras.layers.Dropout(
            config.hidden_dropout_prob, dtype=config.dtype)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderBlock(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = Attention(config, name="attention")
        self.intermediate = Intermediate(config, name="intermediate")
        self.bert_output = IntermediateOutput(config, name="output")

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs
        attention_outputs = self.attention(
            [hidden_states, attention_mask], training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(
            [attention_output, attention_mask], training=training
        )
        layer_output = self.bert_output(
            [intermediate_output, attention_output], training=training
        )
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=layer_output.dtype
        )
        # add attentions if we output them
        outputs = (masked_layer_output,) + attention_outputs[1:]
        return outputs


class Encoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            EncoderBlock(config, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs
        all_hidden_states = ()
        all_attentions = ()
        for idx, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, attention_mask], training=training
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class Decoder(Encoder):
    """Fast Speech decoder module."""

    def __init__(self, config, **kwargs):
        self.is_compatible_encoder = kwargs.pop("is_compatible_encoder", True)
        super().__init__(config, **kwargs)
        self.config = config
        self.init_pos_embedding = sincos_embedding(
            self.config.hidden_size, self.config.max_position_embeddings)
        # create decoder positional embedding
        self.decoder_positional_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings+1,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.Constant(
                self.init_pos_embedding),
            name="position_embeddings",
            trainable=False,
            dtype=config.dtype
        )

        if self.is_compatible_encoder is False:
            self.project_compatible_decoder = tf.keras.layers.Dense(
                units=config.hidden_size, name="project_compatible_decoder", dtype=config.dtype
            )

    def call(self, inputs, training=False):
        hidden_states, encoder_mask, decoder_pos = inputs

        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)

        # calculate new hidden states.
        hidden_states += tf.cast(
            self.decoder_positional_embeddings(
                decoder_pos), hidden_states.dtype
        )

        return super().call([hidden_states, encoder_mask], training=training)


class TacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 postnet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.postnet_num_conv_layers):
            conv = tf.keras.layers.Conv1D(
                filters=config.postnet_conv_filters
                if i < config.postnet_num_conv_layers - 1
                else config.num_mels,
                kernel_size=config.postnet_conv_kernel_size,
                padding="same",
                name="conv_._{}".format(i),
                dtype=config.dtype
            )
            # We use LN instead of BN here due to the small batch size.
            # Confirm that there is no much difference when switching to LN on GPU.
            batch_norm = tf.keras.layers.LayerNormalization(
                name="batch_norm_._{}".format(i), epsilon=config.layer_norm_eps, dtype=config.dtype
            )
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = ipu.keras.layers.Dropout(
            rate=config.postnet_dropout_rate, dtype=config.dtype, name="dropout"
        )
        self.activation = [tf.nn.tanh] * \
            (config.postnet_num_conv_layers - 1) + [tf.identity]
        self.config = config

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        extended_mask = tf.cast(tf.expand_dims(
            mask, axis=2), self.config.dtype)
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
            outputs = self.dropout(outputs, training=training)
            outputs = tf.cast(outputs, self.config.dtype)
        return outputs * extended_mask


class PostnetBlock(tf.keras.layers.Layer):
    def __init__(self, config, activation, num_filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=config.postnet_conv_kernel_size,
            padding="same",
            name="conv",
            dtype=config.dtype
        )
        self.batch_norm = tf.keras.layers.LayerNormalization(
            axis=-1, name="batch_norm", epsilon=config.layer_norm_eps, dtype=config.dtype
        )
        self.dropout = ipu.keras.layers.Dropout(
            rate=config.postnet_dropout_rate, dtype=config.dtype, name="dropout"
        )
        self.activation = activation
        self.config = config

    def call(self, inputs, training=False):
        outputs, mask = inputs
        outputs = self.conv(outputs)
        outputs = self.batch_norm(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = tf.cast(outputs, self.config.dtype)
        outputs = outputs * mask
        return outputs, mask


class VariantPredictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        # Add namescope for clarity
        with tf.name_scope(kwargs["name"]):
            self.conv_layers = []
            for i in range(config.variant_predictor_num_conv_layers):
                conv1d = tf.keras.layers.Conv1D(
                    config.variant_predictor_filter,
                    config.variant_predictor_kernel_size,
                    padding="same",
                    name="conv_._{}".format(i),
                    dtype=config.dtype
                )
                activation = tf.keras.layers.Activation(tf.nn.relu)
                layer_norm = ipu.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i), dtype=config.dtype
                )
                dropout = ipu.keras.layers.Dropout(
                    config.variant_predictor_dropout_rate, dtype=config.dtype)

                self.conv_layers.append(
                    (conv1d, activation, layer_norm, dropout))
            self.output_layer = tf.keras.layers.Dense(1, dtype=config.dtype)

        self.config = config

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(
            tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype
        )

        # mask encoder hidden states
        outputs = encoder_hidden_states * attention_mask
        # pass though all layers
        for conv, act, ln, dp in self.conv_layers:
            outputs = conv(outputs)
            outputs = act(outputs)
            outputs = ln(outputs)
            outputs = dp(outputs)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs * attention_mask
        outputs = tf.squeeze(masked_outputs, -1)
        return outputs


class LengthRegulator(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs, training=False):
        """Call logic.
        Args:
            1. encoder_hidden_states, Tensor (float32) shape [batch_size, length, hidden_size]
            2. durations_gt, Tensor (float32/int32) shape [batch_size, length]
        """
        encoder_hidden_states, durations_gt = inputs
        outputs, encoder_masks = self._length_regulator(
            encoder_hidden_states, durations_gt
        )
        return outputs, encoder_masks

    def _synthetic_length_regulator(self, encoder_hidden_states, durations_gt):
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        repeats = durations_gt[0]
        real_length = tf.reduce_sum(repeats)
        encoder_masks = tf.sequence_mask(
            [real_length], self.config.max_wave_length)

        pad_num = self.config.max_wave_length - self.config.max_seq_length
        outputs = tf.pad(encoder_hidden_states, [[0, 0], [0, pad_num], [0, 0]])
        return outputs, encoder_masks

    def _length_regulator(self, encoder_hidden_states, durations_gt):
        batch_size = encoder_hidden_states.shape[0]
        outputs = {
            "output_types": [self.config.dtype],
            "output_shapes": [tf.TensorShape([batch_size, self.config.encoder_self_attention_params.hidden_size, self.config.max_wave_length])],
        }
        base_path = os.path.realpath(os.path.dirname(__file__))
        lib_path = os.path.join(
            base_path, r"custom_op/length_regulator/liblengthRegulator.so")
        gp_path = os.path.join(
            base_path, r"custom_op/length_regulator/lengthRegulator.gp")

        predict = ipu.custom_ops.precompiled_user_op(inputs=[encoder_hidden_states, durations_gt],
                                                     library_path=lib_path,
                                                     gp_path=gp_path,
                                                     outs=outputs)

        real_length = tf.reduce_sum(durations_gt, axis=1)
        encoder_masks = tf.sequence_mask(
            real_length, self.config.max_wave_length)
        encoder_masks = tf.cast(encoder_masks, self.config.dtype)
        return predict[0], encoder_masks


def build_n_encoder_block(config, inputs, num_encoder_layers, shift=0, training=True, prefix="encoder", **kwargs):
    hidden_states, attention_mask = inputs
    for layer_idx in range(num_encoder_layers):
        encoder_outputs = EncoderBlock(
            config,
            name=f"{prefix}/layer_._{layer_idx+shift}",
            **kwargs)([hidden_states, attention_mask], training=training)

        hidden_states = encoder_outputs[0]
    return hidden_states


def build_n_decoder_block(config, inputs, num_decoder_layers,  shift=0, training=True, is_compatible_encoder=True, **kwargs):
    hidden_states, encoder_mask, decoder_pos = inputs
    if not is_compatible_encoder:
        hidden_states = tf.keras.layers.Dense(
            units=config.hidden_size,
            name="project_compatible_decoder",
            dtype=config.dtype)(hidden_states)
    # Add decoder positional embedding before first encoder layer
    if shift == 0:
        decode_pos_embedding = keras.layers.Embedding(
            config.max_position_embeddings+1,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.Constant(
                sincos_embedding(
                    config.hidden_size, config.max_position_embeddings)),
            name="decoder/position_embeddings",
            trainable=False,
            dtype=config.dtype)(decoder_pos)
        hidden_states += tf.cast(decode_pos_embedding, hidden_states.dtype)

    return build_n_encoder_block(config, [hidden_states, encoder_mask], num_decoder_layers, shift, training, prefix="decoder", **kwargs)


def build_model(opts, training=True):
    data_type = tf.float16 if opts["precision"] == "16" else tf.float32
    config = FastSpeech2Config(dtype=data_type, **opts)
    batch_size = int(opts["batch_size"])

    # construct model
    input_ids = keras.Input(
        shape=(config.max_seq_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="input_ids")
    duration_gts = keras.Input(
        shape=(config.max_seq_length,),
        batch_size=batch_size,
        name="duration_gts")
    f0_gts = keras.Input(
        shape=(config.max_seq_length,),
        batch_size=batch_size,
        name="f0_gts")
    energy_gts = keras.Input(
        shape=(config.max_seq_length,),
        batch_size=batch_size,
        name="energy_gts")

    attention_mask = keras.layers.Lambda(
        lambda x: tf.math.not_equal(x, 0), name="attention_mask")(input_ids)
    phoneme_embedding = FastSpeechEmbeddings(
        config, name="embeddings")(input_ids, training=training)

    encoder_output = Encoder(config.encoder_self_attention_params, name="encoder")(
        [phoneme_embedding, attention_mask], training=training)
    last_encoder_hidden_states = keras.layers.Lambda(
        lambda x: x[0], name="last_encoder_hidden_states")(encoder_output)
    # energy predictor, here use last_encoder_hidden_states, you can use more hidden_states layers
    # rather than just use last_hidden_states of encoder for energy_predictor.
    # [batch_size, phoneme_length]
    duration_outputs = VariantPredictor(config, name="duration_predictor")(
        [last_encoder_hidden_states, attention_mask])

    # [barch_size, phoneme_length, feature]
    f0_outputs = VariantPredictor(config, name="f0_predictor")(
        [last_encoder_hidden_states, attention_mask], training=training)
    # [barch_size, phoneme_length, feature]
    energy_outputs = VariantPredictor(config, name="energy_predictor")(
        [last_encoder_hidden_states, attention_mask], training=training)

    if training:
        f0_embedding = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, 2), name="f0_expand")(f0_gts)
        energy_embedding = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, 2), name="energy_expand")(energy_gts)
    else:
        f0_embedding = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, 2), name="f0_expand")(f0_outputs)
        energy_embedding = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, 2), name="energy_expand")(energy_outputs)

    f0_embedding = keras.layers.Conv1D(
        filters=config.encoder_self_attention_params.hidden_size,
        kernel_size=9,
        padding="same",
        name="f0_embeddings",
        dtype=config.dtype)(f0_embedding)

    energy_embedding = keras.layers.Conv1D(
        filters=config.encoder_self_attention_params.hidden_size,
        kernel_size=9,
        padding="same",
        name="energy_embeddings",
        dtype=config.dtype)(energy_embedding)
    # apply dropout both training/inference
    f0_embedding = ipu.keras.layers.Dropout(
        config.variant_predictor_dropout_rate, name="f0_dropout")(f0_embedding, training=training)
    energy_embedding = ipu.keras.layers.Dropout(
        config.variant_predictor_dropout_rate, name="energy_dropout")(energy_embedding, training=training)
    # sum features
    last_encoder_hidden_states = keras.layers.Add(name="sum_feature")(
        [f0_embedding, energy_embedding, last_encoder_hidden_states])

    last_encoder_hidden_states = keras.layers.Lambda(
        lambda x: tf.transpose(x, [0, 2, 1]))(last_encoder_hidden_states)
    length_regulator_outputs, encoder_masks = LengthRegulator(
        config, name="length_regulator")(
        [last_encoder_hidden_states, duration_gts], training=training)
    # create decoder positional embedding
    # [batch_size, wav_len, hidden_size]
    length_regulator_outputs = keras.layers.Lambda(lambda x: tf.transpose(
        x, [0, 2, 1]), name="length_regulator_outputs")(length_regulator_outputs)

    decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[
        1] + 1, dtype=tf.int32)
    decoder_pos = keras.layers.Lambda(lambda x: tf.expand_dims(
        x, 0))(decoder_pos)

    masked_decoder_pos = tf.multiply(
        encoder_masks, tf.cast(decoder_pos, encoder_masks.dtype))

    decoder_output = Decoder(
        config.decoder_self_attention_params,
        is_compatible_encoder=config.encoder_self_attention_params.hidden_size ==
        config.decoder_self_attention_params.hidden_size,
        name="decoder")(
        [length_regulator_outputs,
         encoder_masks, masked_decoder_pos],
        training=training)

    last_decoder_hidden_states = keras.layers.Lambda(
        lambda x: x[0], name="last_decoder_hidden_states")(decoder_output)
    # here you can use sum or concat more than 1 hidden states layers from decoder.
    mels_before = keras.layers.Dense(
        units=config.num_mels,
        name="mel_before",
        dtype=config.dtype)(last_decoder_hidden_states)

    mels_after = TacotronPostnet(
        config=config,
        dtype=config.dtype,
        name="postnet")([mels_before, encoder_masks], training=training)
    mels_after = keras.layers.Add(
        name="mel_after")([mels_before, mels_after])

    inputs = (input_ids, duration_gts, f0_gts, energy_gts)
    outputs = outputs = (
        mels_before,
        mels_after,
        duration_outputs,
        f0_outputs,
        energy_outputs)

    return inputs, outputs


def build_pipeline_model(opts, training=True):
    data_type = tf.float16 if opts["precision"] == "16" else tf.float32
    config = FastSpeech2Config(dtype=data_type, **opts)
    batch_size = int(opts["batch_size"])

    # construct model
    with ipu.keras.PipelineStage(0):
        # Input layer
        input_ids = keras.Input(
            shape=(config.max_seq_length,),
            batch_size=batch_size,
            dtype=tf.int32,
            name="input_ids")
        duration_gts = keras.Input(
            shape=(config.max_seq_length,),
            batch_size=batch_size,
            name="duration_gts")
        f0_gts = keras.Input(
            shape=(config.max_seq_length,),
            batch_size=batch_size,
            name="f0_gts")
        energy_gts = keras.Input(
            shape=(config.max_seq_length,),
            batch_size=batch_size,
            name="energy_gts")

        attention_mask = keras.layers.Lambda(
            lambda x: tf.cast(tf.math.not_equal(x, 0), config.dtype), name="attention_mask")(input_ids)
        phoneme_embedding = FastSpeechEmbeddings(
            config, name="embeddings")(input_ids, training=training)

        encoder_output = Encoder(config.encoder_self_attention_params, name="encoder")(
            [phoneme_embedding, attention_mask], training=training)
        last_encoder_hidden_states = keras.layers.Lambda(
            lambda x: x[0], name="last_encoder_hidden_states")(encoder_output)

        # energy predictor, here use last_encoder_hidden_states, you can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        # [batch_size, phoneme_length]
        duration_outputs = VariantPredictor(config, name="duration_predictor")(
            [last_encoder_hidden_states, attention_mask])

        # [barch_size, phoneme_length, feature]
        f0_outputs = VariantPredictor(config, name="f0_predictor")(
            [last_encoder_hidden_states, attention_mask], training=training)
        # [barch_size, phoneme_length, feature]
        energy_outputs = VariantPredictor(config, name="energy_predictor")(
            [last_encoder_hidden_states, attention_mask], training=training)

        if training:
            f0_embedding = keras.layers.Lambda(
                lambda x: tf.expand_dims(x, 2))(f0_gts)
            energy_embedding = keras.layers.Lambda(
                lambda x: tf.expand_dims(x, 2))(energy_gts)
        else:
            f0_embedding = keras.layers.Lambda(
                lambda x: tf.expand_dims(x, 2))(f0_outputs)
            energy_embedding = keras.layers.Lambda(
                lambda x: tf.expand_dims(x, 2))(energy_outputs)

        f0_embedding = keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding="same",
            name="f0_embeddings",
            dtype=config.dtype)(f0_embedding)

        energy_embedding = keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding="same",
            name="energy_embeddings",
            dtype=config.dtype)(energy_embedding)
        # apply dropout both training/inference
        f0_embedding = ipu.keras.layers.Dropout(
            config.variant_predictor_dropout_rate)(f0_embedding, training=training)
        energy_embedding = ipu.keras.layers.Dropout(
            config.variant_predictor_dropout_rate)(energy_embedding, training=training)

        # sum features
        last_encoder_hidden_states = keras.layers.Add(name="sum_feature")(
            [f0_embedding, energy_embedding, last_encoder_hidden_states])

        last_encoder_hidden_states = keras.layers.Lambda(
            lambda x: tf.transpose(x, [0, 2, 1]))(last_encoder_hidden_states)
        length_regulator_outputs, encoder_masks = LengthRegulator(
            config, name="length_regulator")(
            [last_encoder_hidden_states, duration_gts], training=training)
        # create decoder positional embedding
        # [batch_size, wav_len, hidden_size]
        length_regulator_outputs = keras.layers.Lambda(lambda x: tf.transpose(
            x, [0, 2, 1]), name="length_regulator_outputs")(length_regulator_outputs)

        decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[
                               1] + 1, dtype=tf.int32)
        decoder_pos = keras.layers.Lambda(lambda x: tf.expand_dims(
            x, 0))(decoder_pos)

        masked_decoder_pos = tf.multiply(
            encoder_masks, tf.cast(decoder_pos, encoder_masks.dtype))
        first_decoder_output = build_n_decoder_block(
            config=config.decoder_self_attention_params,
            inputs=[length_regulator_outputs,
                    encoder_masks, masked_decoder_pos],
            num_decoder_layers=1,
            shift=0,
            training=training,
            is_compatible_encoder=True)

    with ipu.keras.PipelineStage(1):
        last_decoder_hidden_states = build_n_decoder_block(
            config=config.decoder_self_attention_params,
            inputs=[first_decoder_output,
                    encoder_masks, masked_decoder_pos],
            num_decoder_layers=config.decoder_num_hidden_layers - 1,
            shift=1,
            training=training,
            is_compatible_encoder=True
        )
        # here you can use sum or concat more than 1 hidden states layers from decoder.
        mels_before = keras.layers.Dense(
            units=config.num_mels,
            name="mel_before",
            dtype=config.dtype)(last_decoder_hidden_states)

        mels_after = TacotronPostnet(
            config=config,
            dtype=config.dtype,
            name="postnet")([mels_before, encoder_masks], training=training)
        mels_after = keras.layers.Add(
            name="mel_after")([mels_before, mels_after])

    inputs = (input_ids, duration_gts, f0_gts, energy_gts)
    outputs = (mels_before, mels_after, duration_outputs,
               f0_outputs, energy_outputs)
    return inputs, outputs


def build_inference_model(opts):
    data_type = tf.float16 if opts["precision"] == "16" else tf.float32
    config = FastSpeech2Config(dtype=data_type, **opts)
    batch_size = int(opts["batch_size"])
    # Input layer
    input_ids = keras.Input(
        shape=(config.max_seq_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="input_ids")

    # construct model
    attention_mask = keras.layers.Lambda(
        lambda x: tf.math.not_equal(x, 0), name="attention_mask")(input_ids)
    phoneme_embedding = FastSpeechEmbeddings(
        config, name="embeddings")(input_ids)
    encoder_output = Encoder(config.encoder_self_attention_params, name="encoder")(
        [phoneme_embedding, attention_mask])
    last_encoder_hidden_states = keras.layers.Lambda(
        lambda x: x[0], name="last_encoder_hidden_states")(encoder_output)
    # energy predictor, here use last_encoder_hidden_states, you can use more hidden_states layers
    # rather than just use last_hidden_states of encoder for energy_predictor.
    # [batch_size, phoneme_length]
    duration_outputs = VariantPredictor(config, name="duration_predictor")(
        [last_encoder_hidden_states, attention_mask])

    # duration_outputs = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
    duration_outputs = keras.layers.Activation(
        "relu")(tf.math.exp(duration_outputs) - 1.0)
    duration_outputs = tf.math.round(duration_outputs)
    # [batch_size, hidden_size, phn_len]

    # [barch_size, phoneme_length, feature]
    f0_outputs = VariantPredictor(config, name="f0_predictor")(
        [last_encoder_hidden_states, attention_mask])
    # [barch_size, phoneme_length, feature]
    energy_outputs = VariantPredictor(config, name="energy_predictor")(
        [last_encoder_hidden_states, attention_mask])

    f0_expand = keras.layers.Lambda(
        lambda x: tf.expand_dims(x, 2), name="f0_expand")(f0_outputs)
    energy_expand = keras.layers.Lambda(
        lambda x: tf.expand_dims(x, 2), name="energy_expand")(energy_outputs)

    f0_embedding = keras.layers.Conv1D(
        filters=config.encoder_self_attention_params.hidden_size,
        kernel_size=9,
        padding="same",
        name="f0_embeddings",
        dtype=config.dtype)(f0_expand)

    energy_embedding = keras.layers.Conv1D(
        filters=config.encoder_self_attention_params.hidden_size,
        kernel_size=9,
        padding="same",
        name="energy_embeddings",
        dtype=config.dtype)(energy_expand)
    # apply dropout both training/inference
    f0_embedding = ipu.keras.layers.Dropout(
        config.variant_predictor_dropout_rate, name="f0_dropout")(f0_embedding)
    energy_embedding = ipu.keras.layers.Dropout(
        config.variant_predictor_dropout_rate, name="energy_dropout")(energy_embedding)

    # sum features
    last_encoder_hidden_states = keras.layers.Add(name="sum_feature")(
        [f0_embedding, energy_embedding, last_encoder_hidden_states])

    last_encoder_hidden_states = keras.layers.Lambda(
        lambda x: tf.transpose(x, [0, 2, 1]))(last_encoder_hidden_states)

    length_regulator_outputs, encoder_masks = LengthRegulator(
        config, name="length_regulator")(
        [last_encoder_hidden_states, duration_outputs])
    # create decoder positional embedding
    # [batch_size, wav_len, hidden_size]
    length_regulator_outputs = keras.layers.Lambda(lambda x: tf.transpose(
        x, [0, 2, 1]), name="length_regulator_outputs")(length_regulator_outputs)

    decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[
                           1] + 1, dtype=tf.int32)
    decoder_pos = keras.layers.Lambda(lambda x: tf.expand_dims(
        x, 0))(decoder_pos)
    masked_decoder_pos = tf.multiply(
        encoder_masks, tf.cast(decoder_pos, encoder_masks.dtype))

    first_decoder_output = build_n_decoder_block(
        config=config.decoder_self_attention_params,
        inputs=[length_regulator_outputs,
                encoder_masks, masked_decoder_pos],
        num_decoder_layers=1,
        shift=0,
        training=False,
        is_compatible_encoder=True)
    last_decoder_hidden_states = build_n_decoder_block(
        config=config.decoder_self_attention_params,
        inputs=[first_decoder_output,
                encoder_masks, masked_decoder_pos],
        num_decoder_layers=config.decoder_num_hidden_layers - 1,
        shift=1,
        training=False,
        is_compatible_encoder=True
    )

    mels_before = keras.layers.Dense(
        units=config.num_mels,
        name="mel_before",
        dtype=config.dtype)(last_decoder_hidden_states)
    mels_after = TacotronPostnet(
        config=config,
        dtype=config.dtype,
        name="postnet")([mels_before, encoder_masks],
                        training=False)
    mels_after = keras.layers.Add(name="mel_after")([mels_before, mels_after])

    outputs = (mels_before, mels_after, duration_outputs, f0_outputs,
               energy_outputs)

    return input_ids, outputs
