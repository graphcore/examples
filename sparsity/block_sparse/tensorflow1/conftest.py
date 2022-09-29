# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import utils
import pytest


@pytest.fixture(autouse=True)
def create_libblock_sparse():
    utils.build_custom_ops(utils.get_lib_path("block_sparse"))
    utils.set_seed(1)
    return
