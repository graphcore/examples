# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from . import tfds_dataset
from typing import Optional, Callable, Tuple
import tensorflow as tf
import functools


class MNISTDataset(tfds_dataset.TFDSDataset):

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 split: str,
                 shuffle: bool = True,
                 deterministic: bool = False,
                 seed: Optional[int] = None,
                 img_datatype: tf.dtypes.DType = tf.float32,
                 accelerator_side_preprocess: bool = False):

        if img_datatype == tf.uint8 and not accelerator_side_preprocess:
            raise ValueError('Transfering image in uint8 is not compatible with '
                             'preprocessing on the host. Move preprocessing to the IPU.')

        super().__init__(dataset_name=dataset_name,
                         dataset_path=dataset_path,
                         split=split,
                         shuffle=shuffle,
                         deterministic=deterministic,
                         seed=seed)

        self.img_datatype = img_datatype
        self.accelerator_side_preprocess = accelerator_side_preprocess

    def size(self) -> int:
        if self.split == 'train':
            return 60000
        else:
            return 10000

    def image_shape(self) -> Tuple:
        return (28, 28, 1)

    def num_classes(self) -> int:
        return 10

    def cpu_preprocessing_fn(self) -> Callable:

        return functools.partial(mnist_cpu_preprocessing_fn,
                                 scale_on_host=(self.accelerator_side_preprocess is None),
                                 img_datatype=self.img_datatype)

    def ipu_preprocessing_fn(self):

        if self.accelerator_side_preprocess:
            return scale_image
        else:
            return None


def mnist_cpu_preprocessing_fn(image, label, scale_on_host, img_datatype):

    image = tf.cast(image, img_datatype)
    if scale_on_host:
        scale_image(image)
    label = tf.cast(label, tf.int32)
    return image, label


def scale_image(image, scale=1 / 255.0):

    multiplier = tf.cast(scale, image.dtype)
    return multiplier * image
