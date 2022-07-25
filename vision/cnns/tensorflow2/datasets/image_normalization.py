# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import tensorflow as tf
from tensorflow.python import ipu


def image_normalisation(image, mean, std, scale=255):
    mean = tf.cast(mean, dtype=image.dtype)
    std = tf.cast(std, dtype=image.dtype)
    mean = tf.broadcast_to(mean, tf.shape(image))
    std = tf.broadcast_to(std, tf.shape(image))
    return (image / scale - mean) / std


def fused_image_normalisation(image, mean, std, scale=1/255.):
    mean = tf.cast(mean, dtype=image.dtype)
    invstd = tf.cast([1./value for value in std], dtype=image.dtype)
    return ipu.image_ops.normalise_image(image, mean, invstd, scale)
