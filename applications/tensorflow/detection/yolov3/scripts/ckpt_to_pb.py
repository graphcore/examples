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

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from core.yolov3 import YOLOV3
from ipu_utils import stages_constructor
from tensorflow.python.framework import graph_util


def inference(input_data):

    model = YOLOV3(False, opts)
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
        ["input_data", "nums"],
        ["pred_sbbox", "pred_mbbox", "pred_lbbox", "nums"])[0]
    return network_func(input_data)


def ckpt2pb():
    input_size = opts["test"]["input_size"]
    use_moving_avg = opts["yolo"]["use_moving_avg"]
    moving_avg_decay = opts["yolo"]["moving_avg_decay"]
    precision = tf.float16 if opts["yolo"]["precision"] == "fp16" else tf.float32
    with tf.name_scope("input"):
        input_data = tf.placeholder(dtype=precision, shape=(1, input_size, input_size, 3), name="input_data")
    output = inference(input_data)
    output_names = []
    for tensor in output:
        output_names.append(tensor.name.split(":")[0])

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    if use_moving_avg:
        with tf.name_scope("ema"):
            ema_obj = tf.train.ExponentialMovingAverage(
                moving_avg_decay)
        saver = tf.train.Saver(ema_obj.variables_to_restore())
    else:
        saver = tf.train.Saver()
    saver.restore(sess, arguments.ckpt_path)

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
    constant_graph = graph_util.remove_training_nodes(constant_graph)

    with tf.gfile.GFile(arguments.pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluation in TensorFlow", add_help=False)
    parser.add_argument("--config", type=str, default="config/config_544_phase2.json",
                        help="json config file for yolov3.")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="checkpoint from training")
    parser.add_argument("--pb-path", type=str, required=True,
                        help="output pb file path")
    arguments = parser.parse_args()
    with open(arguments.config) as f:
        opts = json.load(f)
    ckpt2pb()
