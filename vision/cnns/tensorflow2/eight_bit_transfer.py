# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

from custom_exceptions import InvalidPrecisionException


class EightBitTransfer:
    def __init__(self, compute_precision: tf.dtypes.DType):
        if compute_precision not in [tf.float16, tf.float32]:
            raise InvalidPrecisionException(f"invalid precision: {compute_precision}")
        self.tf_compute_precision = compute_precision

    # compression - host side
    def compress(self, image_tensors):
        return tf.cast(x=image_tensors, dtype=tf.uint8)

    # decompression - device side (to be added as layer in model)
    def decompress(self, image_tensors):
        return tf.cast(x=image_tensors, dtype=self.tf_compute_precision)
