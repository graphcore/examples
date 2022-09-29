# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import abc
from typing import Callable, Tuple
import tensorflow as tf


class AbstractDataset(abc.ABC):

    @abc.abstractmethod
    def read_single_image(self) -> tf.data.Dataset:
        raise NotImplementedError('Each dataset must provide an imlementation of read_single_image')

    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError('Each dataset must provide an implementation of size')

    @abc.abstractmethod
    def image_shape(self) -> Tuple:
        raise NotImplementedError('Each dataset must provide an implementation of image_shape')

    @abc.abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError('Each dataset must provide an implementation of num_classes')

    @abc.abstractmethod
    def cpu_preprocessing_fn(self) -> Callable:
        raise NotImplementedError('Each dataset must provie an implementation of process_image')

    @abc.abstractmethod
    def ipu_preprocessing_fn(self) -> Callable:
        raise NotImplementedError('Each dataset must provide an implementation of ipu_preprocessing_fn')
