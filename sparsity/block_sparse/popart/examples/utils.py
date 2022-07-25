# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from subprocess import run


def build_custom_ops(so_path):
    """
    Build custom ops library.

    inputs:
    custom ops library path
    """
    build_path = os.path.dirname(so_path)
    run(['make', '-j'], cwd=build_path)
