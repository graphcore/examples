# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import utils

utils.build_custom_ops(utils.get_lib_path("block_sparse"))

utils.set_seed(1)
