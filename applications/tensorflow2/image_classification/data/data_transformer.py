# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python import ipu
import logging

from . import imagenet_processing
from custom_exceptions import UnsupportedFormat, DimensionError


IMAGENET_NORMALISATION_MEAN = [0.485, 0.456, 0.406]
IMAGENET_NORMALISATION_STD = [0.229, 0.224, 0.225]


class DataTransformer:

    logger = logging.getLogger('data_transformer')

    @staticmethod
    def normalization(ds, scale=1 / 255.0, img_type=tf.float32):
        # Applying normalization before `ds.cache()` to re-use it.
        # Note: Random transformations (e.g. images augmentations) should be applied
        # after both `ds.cache()` (to avoid caching randomness) and `ds.batch()`
        # (for vectorization https://www.tensorflow.org/guide/data_performance#vectorizing_mapping).
        if not isinstance(ds, tf.data.Dataset):
            raise UnsupportedFormat(
                f'Type of ds is not the one expected (tf.data.Dataset) {type(ds)}')
        if not hasattr(
                ds.element_spec, '__len__') or len(ds.element_spec) != 2:
            raise DimensionError(
                f'Data dimension is not the one supported (2) {ds.element_spec}')

        multiplier = tf.cast(scale, img_type)
        return ds.map(lambda x,
                      y: (multiplier * tf.cast(x, img_type), tf.cast(y, tf.int32)),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def cache_shuffle(ds: tf.data.Dataset, buffer_size: int = 1, shuffle: bool = True, seed: int = 42):

        if not isinstance(ds, tf.data.Dataset):
            raise UnsupportedFormat(
                f'Type of ds is not the one expected (tf.data.Dataset) {type(ds)}')
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size, seed=seed)

        return ds

    @staticmethod
    def cifar_preprocess(ds,
                         buffer_size,
                         img_type=tf.float32,
                         is_training=True,
                         accelerator_side_preprocess=False,
                         pipeline_num_parallel=48,
                         seed=42):

        if not isinstance(ds, tf.data.Dataset):
            raise UnsupportedFormat(
                f'Type of ds is not the one expected (tf.data.Dataset) {type(ds)}')
        if not hasattr(
                ds.element_spec, '__len__') or len(ds.element_spec) != 2:
            raise DimensionError(
                f'Data dimension is not the one supported (2) {ds.element_spec}')

        ds = DataTransformer.cache_shuffle(ds, buffer_size, is_training, seed)

        preprocess_fn = cifar_preprocess_training_fn if is_training else cifar_preprocess_inference_fn

        if accelerator_side_preprocess:
            host_side_preprocess_fn = None
            accelerator_side_preprocess_fn = preprocess_fn
        else:
            host_side_preprocess_fn = preprocess_fn
            accelerator_side_preprocess_fn = None

        def cifar_preprocess_map_func(x_image):
            assert(x_image.shape == (32, 32, 3))

            if host_side_preprocess_fn is not None:
                x_image = tf.cast(x_image, tf.float32)
                x_image = host_side_preprocess_fn(x_image)

            x_image = tf.cast(x_image, img_type)

            if is_training:
                shape = x_image.get_shape().as_list()
                padding = 4
                x_image = tf.pad(x_image, [[padding, padding], [padding, padding], [0, 0]], "CONSTANT")
                x_image = tf.image.random_crop(x_image, shape, seed=seed)

            return x_image

        ds = ds.map(lambda x, y: (cifar_preprocess_map_func(x), tf.cast(y, tf.int32)),
                    num_parallel_calls=pipeline_num_parallel)
        return ds, accelerator_side_preprocess_fn

    @staticmethod
    def imagenet_preprocessing(ds,
                               img_type,
                               is_training,
                               accelerator_side_preprocess=True,
                               pipeline_num_parallel=48,
                               fused_preprocessing=False,
                               seed=None):

        if fused_preprocessing is True and accelerator_side_preprocess is False:
            raise ValueError('Fused preprocessing can only be done on the IPU. Please enable preprocessing on the IPU.')

        if fused_preprocessing:
            preprocessing_fn = imagenet_fused_preprocess_training_fn if is_training else imagenet_fused_preprocess_inference_fn
        else:
            preprocessing_fn = imagenet_preprocess_training_fn if is_training else imagenet_preprocess_inference_fn

        if accelerator_side_preprocess:
            host_side_preprocess_fn = None
            accelerator_side_preprocess_fn = preprocessing_fn
        else:
            host_side_preprocess_fn = preprocessing_fn
            accelerator_side_preprocess_fn = None

        def processing_fn(raw_record): return imagenet_processing.parse_record(
            raw_record, is_training, img_type, host_side_preprocess_fn, seed=seed)

        return ds.map(processing_fn, num_parallel_calls=pipeline_num_parallel), accelerator_side_preprocess_fn


def _image_normalisation(image, mean, std, scale=255):
    mean = tf.cast(mean, dtype=image.dtype)
    std = tf.cast(std, dtype=image.dtype)
    mean = tf.broadcast_to(mean, tf.shape(image))
    std = tf.broadcast_to(std, tf.shape(image))
    return (image / scale - mean) / std


def _fused_image_normalisation(image, mean, std, scale=1/255.):
    mean = tf.cast(mean, dtype=image.dtype)
    invstd = tf.cast([1./value for value in std], dtype=image.dtype)
    return ipu.image_ops.normalise_image(image, mean, invstd, scale)


def _imagenet_normalize(image):
    return _image_normalisation(image,
                                IMAGENET_NORMALISATION_MEAN,
                                IMAGENET_NORMALISATION_STD)


def _imagenet_fused_normalize(image):
    return _fused_image_normalisation(image,
                                      IMAGENET_NORMALISATION_MEAN,
                                      IMAGENET_NORMALISATION_STD)


def _cifar_normalize(image):
    mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
    std = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
    return _image_normalisation(image, mean, std, scale=1)


def imagenet_preprocess_training_fn(image):
    return _imagenet_normalize(image)


def imagenet_fused_preprocess_training_fn(image):
    return _imagenet_fused_normalize(image)


def imagenet_preprocess_inference_fn(image):
    return _imagenet_normalize(image)


def imagenet_fused_preprocess_inference_fn(image):
    return _imagenet_fused_normalize(image)


def cifar_preprocess_training_fn(image):
    image = tf.image.random_flip_left_right(image)
    return _cifar_normalize(image)


def cifar_preprocess_inference_fn(image):
    return _cifar_normalize(image)
