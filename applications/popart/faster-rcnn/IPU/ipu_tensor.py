# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di

import numpy as np
from config import cfg
import gcop

mappin_gc2npy = {
    'float': np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'bool': np.bool,
    'FLOAT': np.float32,
    'FLOAT32': np.float32,
    'FLOAT16': np.float16,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'BOOL': np.bool,
}
