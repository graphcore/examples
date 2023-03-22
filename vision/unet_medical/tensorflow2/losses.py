# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# This file has been modified by Graphcore Ltd.
# The original file can be found here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/UNet_Medical/runtime/losses.py

import tensorflow as tf


def cast_flatten(y_true, y_pred):
    """Cast labels and logits to FP32 and then flatten them."""
    n_classes = y_pred.shape[-1]

    flat_labels = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(input=y_pred)[0], -1, n_classes])
    flat_logits = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(input=y_pred)[0], -1, n_classes])

    return flat_labels, flat_logits


def dice_coef_fn(y_true, y_pred, axis=1, eps=1e-6):
    """Calculate the Dice score."""
    intersection = tf.reduce_sum(input_tensor=y_pred * y_true, axis=axis)
    union = tf.reduce_sum(input_tensor=y_pred * y_pred + y_true * y_true, axis=axis)
    dice = (2.0 * intersection + eps) / (union + eps)
    return tf.reduce_mean(input_tensor=dice, axis=0)


def dice_coef_loss_fn(y_true, y_pred):
    """Calculate the Dice loss."""
    y_true, y_pred = cast_flatten(y_true, y_pred)
    dice_loss = tf.reduce_mean(
        input_tensor=1 - dice_coef_fn(y_true, tf.keras.activations.softmax(y_pred, axis=-1)), name="dice_loss"
    )
    return dice_loss


def ce_loss(y_true, y_pred):
    """Calculate the cross entropy loss."""
    y_true, y_pred = cast_flatten(y_true, y_pred)
    ce = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true), name="cross_loss"
    )
    return ce


def dice_ce_loss(y_true, y_pred):
    """Calculate the combined loss."""
    ce = ce_loss(y_true, y_pred)
    dice_loss = dice_coef_loss_fn(y_true, y_pred)
    return ce + dice_loss


def dice_coef_accuracy_fn(y_true, y_pred):
    return 1 - dice_coef_loss_fn(y_true, y_pred)
