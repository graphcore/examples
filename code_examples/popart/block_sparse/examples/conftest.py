# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from utils import build_custom_ops

so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../custom_ops.so")
build_custom_ops(so_path)
