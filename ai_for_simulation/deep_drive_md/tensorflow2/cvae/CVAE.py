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


import numpy as np
from .vae_conv_new import conv_variational_autoencoder
import tensorflow as tf
import datetime


def CVAE(input_shape, steps_per_exec, latent_dim=3):
    image_size = input_shape[1:-1]
    channels = input_shape[-1]
    conv_layers = 4
    feature_maps = [64, 64, 64, 64]
    filter_shapes = [(3, 3), (3, 3), (3, 3), (3, 3)]
    strides = [(1, 1), (2, 2), (1, 1), (1, 1)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]
    feature_maps = feature_maps[0:conv_layers]
    filter_shapes = filter_shapes[0:conv_layers]
    strides = strides[0:conv_layers]
    autoencoder = conv_variational_autoencoder(
        image_size, channels, conv_layers, feature_maps,
        filter_shapes, strides, dense_layers, dense_neurons,
        dense_dropouts, latent_dim, steps_per_exec)

    return autoencoder


def create_datasets(x_train, x_val, batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    train_ds = train_ds.map(lambda x : (x, 0.)) # 0. is a dummy value that will be ignored
    train_ds = train_ds.batch(batch_size, drop_remainder=True).repeat().prefetch(16)
    val_ds = tf.data.Dataset.from_tensor_slices(x_val)
    val_ds = val_ds.map(lambda x : (x, 0.)) # 0. is a dummy value that will be ignored
    val_ds = val_ds.batch(batch_size, drop_remainder=True).repeat().prefetch(16)

    return train_ds, val_ds


def run_cvae(hyper_dim=3, epochs=10, batch_size=200, cm_data_input=None, validation=True):

    print("Train dataset size: ", cm_data_input.shape)

    # splitting data into train and validation
    train_val_split = int(0.8 * len(cm_data_input))
    cm_data_train, cm_data_val = cm_data_input[:train_val_split], cm_data_input[train_val_split:]
    input_shape = cm_data_train.shape

    steps_epoch = len(cm_data_train) // batch_size
    steps_val = len(cm_data_val) // batch_size if validation else None

    train_ds, val_ds = create_datasets(cm_data_train, cm_data_val, batch_size=batch_size)
    cm_data_train = train_ds
    cm_data_val = val_ds if validation else None

    cvae = CVAE(input_shape, steps_epoch, hyper_dim)

    start = datetime.datetime.now()
    cvae.train(
        cm_data_train, validation_data=cm_data_val, batch_size=batch_size,
        epochs=epochs, steps_per_epoch=steps_epoch,
        validation_steps=steps_val)
    exec_time = (datetime.datetime.now() - start).total_seconds()
    print("Elapsed Time:", exec_time, " seconds", "batch ", batch_size)

    epoch_times = cvae.time_history.times[1:]  # skip first epoch as there is always some warm-up or compilation
    data_size = batch_size * steps_epoch  # single epoch data size
    tputs = data_size / np.asarray(epoch_times)
    av_tput = sum(tputs) / len(tputs)
    print("Averaged from epochs throughput: ", av_tput, "img/sec")
    print("Average throughput (total data / total time): ", data_size * len(epoch_times) / sum(epoch_times), "img/sec")
    return cvae
