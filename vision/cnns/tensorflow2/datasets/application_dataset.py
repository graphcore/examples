# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class ApplicationDataset:

    pipeline: tf.data.Dataset
    size: int
    image_shape: tuple
    num_classes: int
    padded_size: int = -1

    def evaluate_size(self, micro_batch_size) -> int:
        return self.pipeline.reduce(0, lambda x, _: x + 1) * micro_batch_size
