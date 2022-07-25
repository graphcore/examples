# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import abc
from typing import Callable
from . import application_dataset
import tensorflow as tf


class AbstractDataset(abc.ABC):

    @abc.abstractmethod
    def read_single_image(self) -> application_dataset.ApplicationDataset:
        raise NotImplementedError('Each dataset must provide an imlementation of read_single_image')

    @abc.abstractmethod
    def cpu_preprocessing_fn(self) -> Callable:
        raise NotImplementedError('Each dataset must provie an implementation of process_image')

    @abc.abstractmethod
    def ipu_preprocessing_fn(self) -> Callable:
        raise NotImplementedError('Each dataset must provide an implementation of ipu_preprocessing_fn')

    def post_preprocessing_pipeline(self, ds: tf.data.Dataset)-> tf.data.Dataset:
        raise NotImplementedError('Each dataset must provide an implementation of post_processing_pipeline')
