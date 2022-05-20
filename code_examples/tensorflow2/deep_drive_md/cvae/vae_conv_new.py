# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Copyright (c) 2021, Cray Labs
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file has been modified by Graphcore Ltd.

'''
Convolutional variational autoencoder in Keras

Reference: "Auto-Encoding Variational Bayes" (https://arxiv.org/abs/1312.6114)
'''
import numpy
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Convolution2D, Conv2DTranspose, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.losses import binary_crossentropy
import warnings
import tensorflow as tf
import time


# save history from log
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, eps_mean=0.0, eps_std=1.0):
        super().__init__()
        self.eps_mean = eps_mean
        self.eps_std = eps_std

    def call(self, inputs):

        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=self.eps_mean, stddev=self.eps_std)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


def encoder_decoder(latent_dim, channels, image_size, feature_maps, filter_shapes, activation,
                    strides, conv_layers, dense_layers, dense_neurons, dense_dropouts, eps_mean,
                    eps_std):

    if K.image_data_format() == 'th' or K.image_data_format() == 'channels_first':
        encoder_inputs = Input(shape=(channels, image_size[0], image_size[1]))
    else:
        encoder_inputs = Input(shape=(image_size[0], image_size[1], channels))
    encode_conv = []
    layer = Convolution2D(
        feature_maps[0], filter_shapes[0], padding='same', activation=activation,
        strides=strides[0])(encoder_inputs)
    encode_conv.append(layer)
    for i in range(1, conv_layers):
        layer = Convolution2D(
            feature_maps[i], filter_shapes[i], padding='same', activation=activation,
            strides=strides[i])(encode_conv[i-1])
        encode_conv.append(layer)

    flat = Flatten()(encode_conv[-1])
    encode_dense = []
    layer = Dense(dense_neurons[0], activation=activation)(Dropout(dense_dropouts[0])(flat))
    encode_dense.append(layer)
    for i in range(1, dense_layers):
        layer = Dense(dense_neurons[i], activation=activation)(Dropout(dense_dropouts[i])(encode_dense[i-1]))
        encode_dense.append(layer)

    z_mean = Dense(latent_dim)(encode_dense[-1])
    z_log_var = Dense(latent_dim)(encode_dense[-1])
    z = Sampling(eps_mean, eps_std)([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


    decode_dense = []
    latent_inputs = Input(shape=(latent_dim,))
    layer = Dense(dense_neurons[-1], activation=activation)(latent_inputs)
    decode_dense.append(layer)
    for i in range(1, dense_layers):
        layer = Dense(dense_neurons[-i-1], activation=activation)(decode_dense[i-1])
        decode_dense.append(layer)
    if K.image_data_format() == 'th' or K.image_data_format() == 'channels_first':
        dummy_input = numpy.ones((1, channels, image_size[0], image_size[1]))
    else:
        dummy_input = numpy.ones((1, image_size[0], image_size[1], channels))
    dummy = Model(encoder_inputs, encode_conv[-1])
    conv_size = dummy.output_shape

    decode_conv = []
    layer = Dense(conv_size[1] * conv_size[2] * conv_size[3], activation=activation)(decode_dense[-1])
    decode_dense.append(layer)
    reshape = Reshape(conv_size[1:])(decode_dense[-1])
    decode_conv.append(reshape)

    # define deconvolutional decoding layers
    for i in range(1, conv_layers):
        if K.image_data_format() == 'th' or K.image_data_format() == 'channels_first':
            dummy_input = numpy.ones((1, channels, image_size[0], image_size[1]))
        else:
            dummy_input = numpy.ones((1, image_size[0], image_size[1], channels))
        dummy = Model(encoder_inputs, encode_conv[-i-1])
        conv_size = list(dummy.output_shape)

        if K.image_data_format() == 'th' or K.image_data_format() == 'channels_first':
            conv_size[1] = feature_maps[-i]
        else:
            conv_size[3] = feature_maps[-i]

        layer = Conv2DTranspose(
            feature_maps[-i-1], filter_shapes[-i], padding='same', activation=activation,
            strides=strides[-i])(decode_conv[i-1])
        decode_conv.append(layer)

    decoder_outputs = Conv2DTranspose(
        channels, filter_shapes[0], padding='same', activation='sigmoid',
        strides=strides[0])(decode_conv[-1])

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


class conv_variational_autoencoder(Model):
    '''
    variational autoencoder class

    parameters:
        - image_size: tuple
        height and width of images
        - channels: int
        number of channels in input images
        - conv_layers: int
        number of encoding/decoding convolutional layers
        - feature_maps: list of ints
        number of output feature maps for each convolutional layer
        - filter_shapes: list of tuples
        convolutional filter shape for each convolutional layer
        - strides: list of tuples
        convolutional stride for each convolutional layer
        - dense_layers: int
        number of encoding/decoding dense layers
        - dense_neurons: list of ints
        number of neurons for each dense layer
        - dense_dropouts: list of float
        fraction of neurons to drop in each dense layer (between 0 and 1)
        - latent_dim: int
        number of dimensions for latent embedding
        - activation: string (default='relu')
        activation function to use for layers
        - eps_mean: float (default = 0.0)
        mean to use for epsilon (target distribution for embedding)
        - eps_std: float (default = 1.0)
        standard dev to use for epsilon (target distribution for embedding)

    methods:
        - train(data,batch_size,epochs=1,checkpoint=False,filepath=None)
        train network on given data
        - save(filepath)
        save the model weights to a file
        - load(filepath)
        load model weights from a file
        - return_embeddings(data)
        return the embeddings for given data
        - generate(embedding)
        return a generated output given a latent embedding
    '''

    def __init__(self, image_size, channels, conv_layers, feature_maps, filter_shapes, strides,
                 dense_layers, dense_neurons, dense_dropouts, latent_dim, steps_per_exec,
                 activation='relu', eps_mean=0.0, eps_std=1.0, **kwargs):
        super(conv_variational_autoencoder, self).__init__(name="CVAE", **kwargs)

        self.history_call = LossHistory()
        self.time_history = TimeHistory()

        # check that arguments are proper length
        if len(filter_shapes) != conv_layers:
            raise Exception("number of convolutional layers must equal length of filter_shapes list")
        if len(strides) != conv_layers:
            raise Exception("number of convolutional layers must equal length of strides list")
        if len(feature_maps) != conv_layers:
            raise Exception("number of convolutional layers must equal length of feature_maps list")
        if len(dense_neurons) != dense_layers:
            raise Exception("number of dense layers must equal length of dense_neurons list")
        if len(dense_dropouts) != dense_layers:
            raise Exception("number of dense layers must equal length of dense_dropouts list")

        # even shaped filters may cause problems in theano backend
        even_filters = [f for pair in filter_shapes for f in pair if f % 2 == 0]
        if K.image_data_format() == 'th' and len(even_filters) > 0:
            warnings.warn('Even shaped filters may cause problems in Theano backend')
        if K.image_data_format() == 'channels_first' and len(even_filters) > 0:
            warnings.warn('Even shaped filters may cause problems in Theano backend')

        self.image_size = image_size

        (self.encoder, self.decoder) = encoder_decoder(
            latent_dim, channels, image_size, feature_maps, filter_shapes, activation,
            strides, conv_layers, dense_layers, dense_neurons, dense_dropouts, eps_mean,
            eps_std)
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.encoder.set_infeed_queue_options(prefetch_depth=16)
        self.decoder.set_infeed_queue_options(prefetch_depth=16)
        self.compile(optimizer=self.optimizer, steps_per_execution=steps_per_exec)
        self.inputs = self.encoder.inputs
        self.build(tf.TensorShape((1, image_size[0], image_size[1], channels)))
        self.summary()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }

    def save(self, filepath):
        '''
        save the model weights to a file

        parameters:
            - filepath: string
            path to save model weights

        outputs:
            None
        '''
        self.save_weights(filepath)

    def load(self, filepath):
        '''
        load model weights from a file

        parameters:
            - filepath: string
            path from which to load model weights

        outputs:
            None
        '''
        self.load_weights(filepath)

    def decode(self, data):
        '''
        return the decodings for given data

        parameters:
            - data: numpy array
            input data

        outputs:
            numpy array of decodings for input data
        '''
        _, _, z = self.encoder(data)
        return self.decoder(z)

    def return_embeddings(self, data):
        '''
        return the embeddings for given data

        parameters:
            - data: numpy array
            input data

        outputs:
            numpy array of embeddings for input data
        '''
        z_mean, _, _ = self.encoder(data)
        return z_mean

    def generate(self, embedding):
        '''
        return a generated output given a latent embedding

        parameters:
            - data: numpy array
            latent embedding

        outputs:
            numpy array of generated output
        '''
        return self.decoder(embedding)

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train(self, train_data, validation_data, batch_size, epochs, steps_per_epoch=1, validation_steps=1):
        self.fit(
            train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            callbacks=[self.history_call, self.time_history], shuffle=True, verbose=2)
