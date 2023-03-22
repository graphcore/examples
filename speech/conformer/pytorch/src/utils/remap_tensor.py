# pylint: disable=missing-function-docstring
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# This file has been modified by Graphcore Ltd.

import poptorch
import ctypes


def remap_tensor(
    x,
    debug_str="",
):
    ctypes.cdll.LoadLibrary("./custom_ops.so")

    return poptorch.custom_op(
        [x],
        "RemapCE",
        "ai.graphcore",
        1,
        example_outputs=[x],
        attributes={
            "fwd_grain_size": 16,
            "bwd_grain_size": 16,
            "fwd_clone_layout": 0,
            "bwd_clone_layout": 0,
            "fwd_after_matmul": 0,
            "bwd_after_matmul": 0,
            "debug_str": debug_str,
        },
    )[0]
