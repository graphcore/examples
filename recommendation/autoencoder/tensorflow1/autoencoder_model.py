# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# Original paper:
# Training Deep AutoEncoders for Collaborative Filtering
# By Oleksii Kuchaiev and Boris Ginsburg
# https://arxiv.org/pdf/1708.01715.pdf

import tensorflow as tf
from collections import namedtuple


AutoencoderDefinition = namedtuple(
    'AutoencoderDefinition', [
        'layer_sizes',
        'dropout_rate'])


ARCHITECTURES = {
    128: AutoencoderDefinition([128, 256, 256], 0.65),
    256: AutoencoderDefinition([256, 256, 512], 0.8),
    512: AutoencoderDefinition([512, 512, 1024], 0.8)
}


class AutoencoderModel():
    def __init__(self, opts, *args, **kwargs):
        dtypes = opts.precision.split('.')
        self.dtype = tf.float16 if dtypes[0] == '16' else tf.float32
        self.master_dtype = tf.float16 if dtypes[1] == '16' else tf.float32

        definition = ARCHITECTURES[opts.size]

        # Apply layer sizes
        self.input_size = opts.input_size
        self.apply_selu = opts.apply_selu
        self.apply_dropout = opts.apply_dropout
        self.layer_sizes = definition.layer_sizes
        self.dropout_rate = definition.dropout_rate

    def activation(self, x, name='selu'):
        return tf.nn.selu(x, name=name)

    def fc(self, x, weights, biases, name='fc'):
        x = tf.nn.xw_plus_b(x, weights, biases, name=name)
        return x

    def _get_variable(self, name, shape, init):
        var = tf.get_variable(
            name,
            shape,
            initializer=init,
            dtype=self.master_dtype)
        if self.master_dtype != self.dtype:
            var = tf.cast(var, dtype=self.dtype)
        return var

    def _build_graph(self, x):
        weights_initializer = tf.contrib.layers.xavier_initializer(dtype=self.master_dtype)

        encoder_layer_sizes = [self.input_size] + self.layer_sizes
        encoder_W_sizes = zip(
            encoder_layer_sizes[:-1], encoder_layer_sizes[1:])

        # Build encoder graph
        encoder_weights = []
        with tf.variable_scope('encoder', use_resource=True, reuse=tf.AUTO_REUSE):
            for (layer_number, (num_units_in, num_units_out)) in enumerate(encoder_W_sizes):
                scope_name = "fc_{}".format(layer_number + 1)

                with tf.variable_scope(scope_name, use_resource=True, reuse=tf.AUTO_REUSE):
                    weights = self._get_variable(
                        'weights',
                        shape=[
                            num_units_in,
                            num_units_out],
                        init=weights_initializer)
                    biases = self._get_variable(
                        'biases', shape=[num_units_out], init=tf.constant_initializer(0.0))
                    encoder_weights.append(weights)

                    x = self.fc(x, weights, biases, name=scope_name + 'matmul')
                    if self.apply_selu:
                        x = self.activation(x, name=scope_name + 'selu')

            # Apply dropout only to the final layer of the encoder
            if self.apply_dropout:
                x = tf.nn.dropout(x, rate=self.dropout_rate, name='dropout')

        # Build decoder graph - constrained to mirror the encoder weights
        with tf.variable_scope('decoder', use_resource=True, reuse=tf.AUTO_REUSE):
            for (layer_number, weights) in enumerate(list(reversed(encoder_weights))):
                scope_name = "fc_{}".format(layer_number + len(encoder_weights))

                with tf.variable_scope(scope_name, use_resource=True, reuse=tf.AUTO_REUSE):
                    biases = self._get_variable(
                        'biases',
                        shape=[
                            weights.get_shape()[0]],
                        init=tf.constant_initializer(0.0))

                    x = self.fc(
                        x,
                        tf.transpose(weights),
                        biases,
                        name=scope_name +
                        'matmul')
                    if self.apply_selu:
                        x = self.activation(x, name=scope_name + 'selu')

        return x

    def __call__(self, x):
        return self._build_graph(x)

    @staticmethod
    def get_description_string(size=128):
        definition = ARCHITECTURES[size]
        architecture_description = "n, {}, dp({}), {}, n".format(
            ', '.join(str(s) for s in definition.layer_sizes),
            definition.dropout_rate,
            ', '.join(str(s) for s in list(reversed(definition.layer_sizes[:-1])))
        )
        return architecture_description
