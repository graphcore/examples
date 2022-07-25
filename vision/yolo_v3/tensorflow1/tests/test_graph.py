# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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


import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from core.yolov3 import YOLOV3 as current_YOLOV3
from ipu_utils import stages_constructor
from tensorflow.python import ipu
from tests.original_model.yolov3 import YOLOV3 as original_YOLOV3


tf.disable_eager_execution()
tf.disable_v2_behavior()


root_dir = os.path.dirname(__file__) + "/../"
tmp_weight_file = root_dir + "tests/tmp.ckpt"


def run_original_model(image_data):
    with tf.name_scope('define_input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    with tf.name_scope("define_loss"):
        model = original_YOLOV3(input_data, False)
    sess = tf.Session(config=tf.ConfigProto())
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    saver.save(sess, tmp_weight_file)
    return sess.run(
        [model.pred_sbbox, model.pred_mbbox, model.pred_lbbox],
        feed_dict={input_data: image_data})


def run_current_model(image_data):
    precision = tf.float32

    # Configure arguments for targeting the IPU
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    with open(root_dir + "config/config.json") as f:
        opts = json.load(f)
    opts["yolo"]["precision"] = "fp32" if precision == tf.float32 else "fp16"
    opts["test"]["batch_size"] = image_data.shape[0]
    opts["test"]["input_size"] = image_data.shape[1]
    opts["yolo"]["classes"] = root_dir + '/data/classes/voc.names'
    opts["yolo"]["anchors"] = root_dir + 'data/anchors/baseline_anchors.txt'
    with tf.name_scope("input"):
        # three channel images
        input_data = tf.placeholder(
            shape=image_data.shape, dtype=precision, name="input_data")
    model = current_YOLOV3(False, opts)
    # construct model
    # we will put whole network on one ipu
    layers = []
    # build layer functions for backbone and upsample
    layers.extend(model.build_backbone())
    # last layer of darknet53 is classification layer, so it have 52 conv layers
    assert len(layers) == 52
    layers.extend(model.build_upsample())
    # there is 25 conv layers if we count upsmaple as a conv layer
    assert len(layers) == 52+25
    # decoding layer and loss layer is always put on last IPU
    layers.append(model.decode_boxes)

    # reuse stages_constructor so we don't need to pass params by hand
    network_func = stages_constructor(
        [layers],
        ["input_data"],
        ["pred_sbbox", "pred_mbbox", "pred_lbbox"])[0]

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        output = ipu.ipu_compiler.compile(
            network_func, [input_data])

    sess = tf.Session(
        config=tf.ConfigProto())
    loader = tf.train.Saver()
    loader.restore(sess, tmp_weight_file)

    return sess.run(
        output,
        feed_dict={
            input_data: image_data,
        }
    )


def test_model_structure():
    image_data = np.random.random((1, 320, 320, 3))
    result_original = run_original_model(image_data)
    tf.reset_default_graph()
    result_current = run_current_model(image_data)
    for current_head, original_head in zip(result_current, result_original):
        if not np.allclose(current_head, original_head,
                           equal_nan=True, rtol=1e-3, atol=1e-3):
            raise RuntimeError("current model output do not match original model output")
