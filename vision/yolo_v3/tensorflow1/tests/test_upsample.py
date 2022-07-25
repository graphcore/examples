# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from functools import partial as bind
from optparse import OptionParser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import tensorflow.compat.v1 as tf
from core.common import upsample
from tensorflow.python import ipu


tf.disable_v2_behavior()
tf.disable_eager_execution()


def test_upsample():
    input_value = np.random.random((2, 224, 224, 3)).astype(np.float16)
    # nearest_neighborhood function that run on cpu:
    input_data = tf.constant(input_value)
    input_shape = tf.shape(input_data)
    upsample_cpu = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    with tf.Session() as sess:
        cpu_output = sess.run(upsample_cpu)

    tf.reset_default_graph()

    # upsample that run on IPU:
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()
    with ipu.scopes.ipu_scope('/device:IPU:0'):
        input_data = tf.constant(input_value)
        upsample_ipu = upsample(input_data, "test_upsample", method="resize")
    with tf.Session(config=tf.ConfigProto()) as sess:
        ipu_output = sess.run(upsample_ipu)
    assert np.max(cpu_output - ipu_output) <= 0.0001
