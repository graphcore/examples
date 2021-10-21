# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import ctypes
import os
import pathlib
from pathlib import Path

import torch

import poptorch


def load_custom_ops_lib(custom_op_path):
    so_path = os.path.join(os.path.dirname(__file__), custom_op_path)

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        print("Couldn't find file", so_path)
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


class CopyTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        load_custom_ops_lib("custom_ops/build/copy_tensor_custom_op.so")

    def forward(self, input_):
        return poptorch.custom_op(
            inputs=[input_],
            name="CopyTensor",
            domain="ai.graphcore",
            domain_version=1,
            example_outputs=[input_],
        )
