# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import tensorflow as tf
from numpy import random, float32, maximum
from functools import partial


def assign_mixup_coefficients(data_dict, batch_size, alpha, rng_seed_for_testing=None):
    """
    This function generates mixup coefficients for a given minibatch and stores them in the 'data_dict'


    :param data_dict: data dictionary including 'image', 'label' and 'mixup_alpha'.
    :param batch_size: batch size for the minibatch that will be mixed.
    :param alpha: the value determining the beta distribution from which the mixing coefficient is drawn.
    :param rng_seed_for_testing: numpy seed (for testing)
    :return data_dict with new field 'mixup_coefficients', of shape [batch_size]
    """

    # numpy implementation of beta distribution is more numerically stable than TF.
    def numpy_mix_generator():
        if rng_seed_for_testing is not None:
            random.seed(rng_seed_for_testing)

        # symmetrical beta distribution.
        mix = random.beta(alpha, alpha, [batch_size]).astype(float32)
        # always choose the original image as the 'foreground' image
        mix = maximum(mix, 1. - mix)
        return mix

    assert batch_size > 1, "batch size MUST be > 1 within the mixup function"
    mixup_coefficients = tf.numpy_function(numpy_mix_generator, [], tf.float32)
    data_dict['mixup_coefficients'] = tf.reshape(tf.cast(mixup_coefficients, data_dict['image'].dtype), [batch_size])
    return data_dict


def mixup_image(data_dict):
    """
    function to perform mixup -- from https://arxiv.org/abs/1710.09412
    Note: batching must be performed before this operation because
    mixing happens between different members of the same minibatch

    :param data_dict: data dictionary with the 'image' and 'label' for the batch
    :return data_dict with new fields: 'label_mixed_up' (used in loss)
                                       'mixup_coefficients_2' (used if combining mixup and cutmix)
    """

    mix_coeffs = data_dict['mixup_coefficients']
    # mixing is done by 'rolling' each minibatch along by one place
    mixup_permute_batch_op = partial(tf.roll, shift=1, axis=0)

    data_dict['image'] = (mix_coeffs[:, None, None, None] * data_dict['image'] +
                          (1. - mix_coeffs[:, None, None, None]) * mixup_permute_batch_op(data_dict['image']))
    data_dict['label_mixed_up'] = mixup_permute_batch_op(data_dict['label'])

    # 'mixup_coefficients_2' will be used if we need to know the mixup coefficients for the SHUFFLED images
    # (used if mixing cutmix & mixup together)
    # it needs to use the same 'rolling' as in the cutmix, so that future cutmix will know the right mixed labels
    bs = int(data_dict['image'].shape[0])
    data_dict['mixup_coefficients_2'] = cutmix_permute_batch_op(bs)(data_dict['mixup_coefficients'])

    return data_dict


def cutmix_permute_batch_op(batch_size):
    """
    operation to permute the minibatch in preparation for 'cutmix'
    :param batch_size: mini-batch size
    :return: function to permute the batch to perform cutmix
    """
    # this is how we permute the batch to do cutmix
    return partial(tf.roll, shift=2 if batch_size > 2 else 1, axis=0)


def cutmix(data_dict, cutmix_lambda, cutmix_version=2, rng_seed_for_testing=None):
    """
    implementation of cutmix, https://arxiv.org/abs/1905.04899
    There is a major difference in our implementation. While the authors proposed sampling cutmix_lambda from a
    uniform distribution of [0, 1), we found that this over-regularised our models. Instead, we use a fixed
    value for lambda.
    We denote the images from the un-permuted minibatch as the 'base' images. We denote the patches we are pasting in
    as coming from the 'shuffled' images

    :param data_dict: data dictionary with the 'image' and 'label' for the batch
    :param cutmix_lambda: approximate proportion of the output image that the 'base' image makes up
    :param cutmix_version: int: which version of cutmix to use (v1 to repeat the results of [paper-url])
    :param rng_seed_for_testing: seed for testing purposes
    :return: data_dict with new fields 'cutmix_label', 'cutmix_lambda' and (if mixup) 'cutmix_label2', all of which
      are to be used in the loss function
    """
    assert 0. <= cutmix_lambda <= 1., "cutmix lambda must be between 0. and 1."

    input_images = data_dict['image']
    batch_size = int(input_images.shape[0])
    channels = int(input_images.shape[-1])
    assert tf.keras.backend.ndim(input_images) == 4
    assert batch_size > 1, "cutmix must have batch size > 1"

    cutmix_lambda = cutmix_v1_sample_lambda(cutmix_lambda) if cutmix_version == 1 else cutmix_lambda

    # do the shuffling by 'bumping' the array along by TWO. Random shuffling will end up with  many cases of the image
    # being shuffled with itself (because we normally have small batches).
    # note that we roll by 2 instead of 1, because if we are also using mixup, we do not want the base image to be
    # cut AND mixed with the same shuffled image

    permute_batch_op = cutmix_permute_batch_op(batch_size)
    shuffled_batch = permute_batch_op(input_images)
    cutmix_lambda = tf.cast(cutmix_lambda, tf.float32)

    # NHWC
    h = int(input_images.shape[1])
    w = int(input_images.shape[2])

    # coordinates for the centre of the cutout box
    r_x = tf.random.uniform([], maxval=w, seed=rng_seed_for_testing)
    r_y = tf.random.uniform([], maxval=h, seed=rng_seed_for_testing)

    # box will have same aspect ratio as the image itself
    r_w = w * tf.sqrt(1. - cutmix_lambda)
    r_h = h * tf.sqrt(1. - cutmix_lambda)

    # make sure that the corner is at least r_w / 2 away from any x edge, r_h / 2 away from a y edge
    r_x = tf.clip_by_value(r_x, r_w / 2, w - r_w / 2)
    r_y = tf.clip_by_value(r_y, r_h / 2, h - r_h / 2)

    # bounding box corner coordinates
    x1 = tf.cast(tf.round(r_x - r_w / 2.), tf.int32)
    x2 = tf.cast(tf.round(r_x + r_w / 2.), tf.int32)
    y1 = tf.cast(tf.round(r_y - r_h / 2.), tf.int32)
    y2 = tf.cast(tf.round(r_y + r_h / 2.), tf.int32)

    # creating the box of logicals (1=shuffled image, 0=original image)
    x = tf.range(w)
    y = tf.range(h)
    in_x_range = tf.cast(tf.logical_and(x >= x1, x < x2), tf.int32)
    in_y_range = tf.cast(tf.logical_and(y >= y1, y < y2), tf.int32)

    # y is related to the height, so shape is [None, len(in_y_range), len(in_x_range), None]
    mask = tf.cast(in_x_range * in_y_range[:, None], tf.bool)
    # proportion of the area that is 'zeros' (corresponding to original image)
    cutmix_value = 1. - tf.reduce_mean(tf.cast(mask, tf.float32))

    # get mask ready
    mask = tf.tile(mask[None, ..., None], [batch_size, 1, 1, channels])

    # same lambda value for each member of the minibatch
    cutmix_lambda = tf.fill([batch_size], value=tf.cast(cutmix_value, input_images.dtype))

    # where mask == 1, choose the shuffled batch. else unshuffled
    output_images = tf.where(mask, shuffled_batch, input_images)
    data_dict['image'] = output_images

    data_dict['cutmix_label'] = permute_batch_op(data_dict['label'])
    data_dict['cutmix_lambda'] = cutmix_lambda

    # mixup has been done
    if 'label_mixed_up' in data_dict:
        # need to store all the labels that will apply to a given image (that has already been mixed up)
        data_dict['cutmix_label2'] = permute_batch_op(data_dict['label_mixed_up'])

    return data_dict


def cutmix_v1_sample_lambda(cutmix_lambda):
    """
    draws a cutmix lambda from a distribution that:
    i) has a chance of cutmix_lambda := 1 (no cutmix)
    ii) draws non-unity values of cutmix lambda from the distribution below
    :param cutmix_lambda: original value of cutmix lambda
    :return: the new value of cutmix_lambda
    """
    print("\nUsing cutmix V1.\n")

    denom = 4.
    # sampling to determine (i) if cutmix should be set to 1. and (ii) if not, what it should be set to
    sample_0 = tf.random.uniform([], dtype=tf.float32)
    # ensure second term of cutmix_lambda where expression >= 0
    sample_1 = tf.random.uniform([], dtype=tf.float32, minval=tf.math.exp(-denom))
    cutmix_lambda = tf.where(sample_0 < (cutmix_lambda * 1.55 - 0.96), 1., 1. + tf.math.log(sample_1) / denom)
    return cutmix_lambda
