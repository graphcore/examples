# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Any
import types
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


def wrap_in_loss_scaling(loss_class, scaling_factor):

    class LossScaler(loss_class):

        def __init__(self, *args,  **kwargs):
            super(LossScaler, self).__init__(*args, **kwargs)
            self.scaling_factor = scaling_factor

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return super().__call__(*args, **kwargs) * self.scaling_factor

    return LossScaler
