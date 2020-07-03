# Copyright 2019 Graphcore Ltd.
import math
import pytest
import numpy as np
import tensorflow as tf
import popart
from bert_model import BertConfig, Bert

# Extracted from the official Transformer repo:
# https://github.com/tensorflow/models/blob/638ba7a407b455de91be7cf85a1ff3a7f86cc958/official/transformer/model/model_utils.py#L32


def get_position_encoding_tf(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):

    position = tf.cast(tf.range(length), tf.float32)

    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))

    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)

    scaled_time = tf.expand_dims(position, 1) * \
        tf.expand_dims(inv_timescales, 0)

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


@pytest.mark.parametrize("position_length, hidden_size", [(512, 768), (128, 768), (384, 768)])
def test_positional_encoding_data(position_length, hidden_size):
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
        assert(tf.executing_eagerly())

    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })
    config = BertConfig(vocab_length=9728,
                        batch_size=1,
                        hidden_size=hidden_size,
                        max_positional_length=position_length,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        positional_embedding_init_fn="TRANSFORMER",
                        inference=True)
    popart_model = Bert(config, builder=builder)

    shape = (config.max_positional_length, config.hidden_size)
    pos_pa = popart_model.generate_transformer_periodic_pos_data(
        config.dtype, shape)

    pos_tf = get_position_encoding_tf(shape[0], shape[1]).numpy()
    # Tensorflow broadcast multiplication seems to produce slightly different results
    # to numpy, hence the higher than expected error. The embeddings do correlate well
    # between the two despite this.
    assert(np.all(np.abs(pos_tf - pos_pa) < 5e-5))


def simplified_generator(pos_length, hidden_size):
    """
    Taken verbatim from the Enigma data generator implementation
    """
    scale = 4

    def value(x, y, siz):
        return .02/.707*np.cos(2*scale*np.pi*x*y/siz)

    data = []

    for x in range(pos_length):
        data.append([])
        for y in range(hidden_size):
            data[x].append(value(x, y, hidden_size))

    return np.asarray(data)


@pytest.mark.parametrize("position_length, hidden_size", [(512, 768), (128, 768), (384, 768)])
def test_simplified_position_encoding(position_length, hidden_size):

    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })
    config = BertConfig(vocab_length=9728,
                        batch_size=1,
                        hidden_size=hidden_size,
                        max_positional_length=position_length,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        positional_embedding_init_fn="SIMPLIFIED",
                        inference=True)
    popart_model = Bert(config, builder=builder)

    shape = (config.max_positional_length, config.hidden_size)
    pa_data = popart_model.generate_simplified_periodic_pos_data(
        config.dtype, shape)

    bb_data = simplified_generator(position_length, hidden_size)
    assert(np.all(np.abs(bb_data - pa_data) < 1e-8))
