# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import tensorflow as tf

from utilities.options import ALLOWED_PRECISION_TYPE


class Precision:

    def __init__(self, precision_str):
        if precision_str == "fp16":
            self.policy = tf.keras.mixed_precision.Policy("float16")
            self.cast_model_inputs_to_dtype = tf.float16
            self.metrics_precision = tf.float32
            self.optimizer_compute_precision = tf.float16
        elif precision_str == "fp32":
            self.policy = tf.keras.mixed_precision.Policy("float32")
            self.cast_model_inputs_to_dtype = tf.float32
            self.metrics_precision = tf.float32
            self.optimizer_compute_precision = tf.float32
        elif precision_str == "mixed":
            self.policy = tf.keras.mixed_precision.Policy("mixed_float16")
            self.cast_model_inputs_to_dtype = tf.float32
            self.metrics_precision = tf.float32
            self.optimizer_compute_precision = tf.float32
        else:
            raise ValueError(f"Unrecognised precision type: `{precision_str}`."
                             f" Choose one of {ALLOWED_PRECISION_TYPE}")
