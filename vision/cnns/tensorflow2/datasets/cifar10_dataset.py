# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from . import tfds_dataset
from typing import Optional, Callable
import tensorflow as tf
import functools
from tensorflow.python.ops import math_ops
from . import image_normalization


class CIFAR10Dataset(tfds_dataset.TFDSDataset):

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 split: str,
                 shuffle: bool = True,
                 deterministic: bool = False,
                 seed: Optional[int] = None,
                 img_datatype: tf.dtypes.DType = tf.float32,
                 accelerator_side_preprocess: bool = False):

        super().__init__(dataset_name=dataset_name,
                         dataset_path=dataset_path,
                         split=split,
                         shuffle=shuffle,
                         deterministic=deterministic,
                         seed=seed)

        self.img_datatype = img_datatype
        self.accelerator_side_preprocess = accelerator_side_preprocess

    def cpu_preprocessing_fn(self) -> Callable:

        return functools.partial(cifar_cpu_preprocessing_fn,
                                 split=self.split,
                                 img_datatype=self.img_datatype,
                                 accelerator_side_preprocess=self.accelerator_side_preprocess,
                                 seed=self.seed)

    def ipu_preprocessing_fn(self):

        if self.split == 'train':
            return functools.partial(_cifar10_flip_and_normalize, seed=self.seed)
        else:
            return _cifar10_normalize

    def post_preprocessing_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds


def cifar_cpu_preprocessing_fn(image, label, split, img_datatype, accelerator_side_preprocess, seed):

    assert(image.shape == (32, 32, 3))

    if accelerator_side_preprocess is False:
        image = tf.cast(image, tf.float32)
        if split == 'train':
            image = _cifar10_flip_and_normalize(image, seed)
        else:
            image = _cifar10_normalize(image)

    image = tf.cast(image, img_datatype)

    if split == 'train':
        shape = image.get_shape().as_list()
        padding = 4
        image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]], "CONSTANT")
        image = tf.image.random_crop(image, shape, seed=seed)

    label = tf.cast(label, tf.int32)

    return image, label


def _cifar10_flip_and_normalize(image, seed):
    image = tf.image.random_flip_left_right(image, seed=seed)
    return _cifar10_normalize(image)


def _cifar10_normalize(image):
    mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
    std = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
    return image_normalization.image_normalisation(image, mean, std, scale=1)
