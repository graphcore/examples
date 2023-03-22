# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
from functools import partial

import popxl
import popxl.ops as ops
import popxl_addons as addons

__all__ = ["BatchNorm2D"]


class BatchNorm2D(addons.Module):
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9):
        """
        Implements Batch Normalization (only for inference)
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum  # Not used in inference; the default used is consistent with ONNX

    def build(self, x: popxl.Tensor) -> popxl.Tensor:

        shape = (x.shape[1],)

        self.weight = self.add_variable_input(
            "weight",
            partial(np.ones, shape),
            x.dtype,
        )

        self.bias = self.add_variable_input(
            "bias",
            partial(np.zeros, shape),
            x.dtype,
        )

        self.running_mean = self.add_variable_input(
            "running_mean",
            partial(np.zeros, shape),
            x.dtype,
        )

        self.running_var = self.add_variable_input(
            "running_var",
            partial(np.ones, shape),
            x.dtype,
        )

        y = ops.batch_norm_inference(
            x,
            scale=self.weight,
            bias=self.bias,
            mean=self.running_mean,
            var=self.running_var,
            epsilon=self.epsilon,
            momentum=self.momentum,
        )

        return y
