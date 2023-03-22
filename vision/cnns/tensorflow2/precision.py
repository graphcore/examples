# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

import re
from typing import Tuple
import logging
from custom_exceptions import UnsupportedFormat


class Precision:

    logger = logging.getLogger("precision")
    supported_precisions = ["16.16", "16.32", "32.32"]
    type_mapping = {"16": tf.float16, "32": tf.float32}

    def __init__(self, precision: str):
        self.compute_precision, self.weight_update_precision = self.__validate_input(precision)

    def apply(self) -> None:
        if self.compute_precision == tf.float32:
            tf.keras.mixed_precision.set_global_policy("float32")
            Precision.logger.info("Setting precision 32.32")
        elif self.compute_precision == tf.float16:
            if self.weight_update_precision == tf.float16:
                tf.keras.mixed_precision.set_global_policy("float16")
                Precision.logger.info("Setting precision 16.16")
            else:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                Precision.logger.info("Setting precision 16.32")

    def __validate_input(self, precision: str) -> Tuple[tf.DType, tf.DType]:
        match = re.match(r"(\d+)\.(\d+)", precision)
        if not match:
            raise NameError(f"malformed precision format: {precision}.")

        if precision not in Precision.supported_precisions:
            raise UnsupportedFormat(
                f"precision {precision} is not supported." f"supported precisions: {Precision.supported_precisions}"
            )

        return self.type_mapping[match.group(1)], self.type_mapping[match.group(2)]
