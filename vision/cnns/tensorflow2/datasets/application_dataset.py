# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from attr import dataclass
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class ApplicationDataset:

    pipeline: tf.data.Dataset
    size: int
    image_shape: tuple
    num_classes: int
