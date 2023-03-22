# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


def apply_categorical_feature_noise(features: tf.int32, vocab_sizes, noise_prob):
    # create random categories
    random_category = tf.random.uniform(features.get_shape().as_list(), maxval=1)
    random_category = tf.cast(tf.floor(random_category * vocab_sizes), features.dtype)

    sample_or_not = tf.random.uniform(features.get_shape().as_list(), maxval=1.0) < noise_prob
    noisy_features = tf.where(sample_or_not, random_category, features)
    return noisy_features


def weighted_sample(weights, n_samples):
    s = tf.random.uniform([n_samples], maxval=sum(weights))
    b = 0
    out = tf.zeros([n_samples], dtype=tf.int32)
    for i, w in enumerate(weights):
        out = tf.where((s >= b) & (s < b + w), i, out)
        b += w
    return out


def normalize_ogbBL(values_to_normalize, method):
    ogb_BL_mean = 3.581247
    ogb_BL_std = 0.6618104
    if method == "z_score":
        norm_values = (values_to_normalize - ogb_BL_mean) / ogb_BL_std
    elif method == "std_only":
        norm_values = values_to_normalize / (2 * ogb_BL_std)
    elif method == "mean_only":
        norm_values = values_to_normalize / (2 * ogb_BL_mean)
    else:
        norm_values = values_to_normalize
    return norm_values


def normalize_atom_distances(values_to_normalize, method):
    mean = 3.6966338
    std = 2.1563308
    if method == "z_score":
        norm_values = (values_to_normalize - mean) / std
    elif method == "std_only":
        norm_values = values_to_normalize / (2 * std)
    elif method == "mean_only":
        norm_values = values_to_normalize / (2 * mean)
    else:
        norm_values = values_to_normalize
    return norm_values
