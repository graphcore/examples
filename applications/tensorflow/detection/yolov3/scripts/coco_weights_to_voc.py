#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.



import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from core.yolov3 import YOLOV3
from ipu_utils import stages_constructor
from tensorflow.python import ipu


parser = argparse.ArgumentParser(
    description="Convert tf yolov3 weight train on coco to voc format")
parser.add_argument("--train-from-coco", action="store_false")
parser.add_argument("--original-weight", default="./ckpt_init/yolov3_coco.ckpt")
parser.add_argument("--converted-weight", default="./ckpt_init/yolov3_coco_converted.ckpt")
parser.add_argument("--config", default="./config/config.json")
flag = parser.parse_args()


with open(flag.config) as f:
    opts = json.load(f)
org_weights_path = flag.original_weight
cur_weights_path = flag.converted_weight
# predict head of yolov3
# we won't need this and they are not matched
preserve_cur_names = ["conv_sbbox", "conv_mbbox", "conv_lbbox"]
preserve_org_names = ["Conv_6", "Conv_14", "Conv_22"]


org_weights_mess = []
tf.Graph().as_default()
load = tf.train.import_meta_graph(org_weights_path + ".meta")
with tf.Session() as sess:
    load.restore(sess, org_weights_path)
    for var in tf.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split("/")
        var_shape = var.shape
        if flag.train_from_coco:
            if (var_name_mess[-1] not in ["weights", "gamma", "beta", "moving_mean", "moving_variance"]) or \
                    (var_name_mess[1] == "yolo-v3" and (var_name_mess[-2] in preserve_org_names)):
                continue
        org_weights_mess.append([var_name, var_shape])
        print("=> " + str(var_name).ljust(50), var_shape)
print()
tf.reset_default_graph()

cfg = ipu.config.IPUConfig()
cfg.configure_ipu_system()

cur_weights_mess = []
tf.Graph().as_default()
with tf.name_scope("input"):
    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3), name="input_data")

# converge to a ckpt with fp32
# then use another script to convert to fp16 if needed
opts["yolo"]["precision"] = "fp32"
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
# reuse stages_constructor so we don't need to pass params by hand
network_func = stages_constructor(
    [layers],
    ["input_data"],
    ["pred_sbbox", "pred_mbbox", "pred_lbbox"])[0]

network_func(input_data)

for var in tf.global_variables():
    var_name = var.op.name
    var_name_mess = str(var_name).split("/")
    var_shape = var.shape
    print(var_name_mess[0])
    if flag.train_from_coco:
        if var_name_mess[0] in preserve_cur_names:
            continue
    cur_weights_mess.append([var_name, var_shape])
    print("=> " + str(var_name).ljust(50), var_shape)

org_weights_num = len(org_weights_mess)
cur_weights_num = len(cur_weights_mess)
# maybe 4 more parameter because using conv2d_transpose
if cur_weights_num != org_weights_num and cur_weights_num != org_weights_num + 4:
    print("cur_weights_num:", cur_weights_num)
    print("org_weights_num:", org_weights_num)
    raise RuntimeError

print("=> Number of weights that will rename:\t%d" % cur_weights_num)
cur_to_org_dict = {}

# need to remove conv2d_transpose
cur_index_plus = 0
for index in range(cur_weights_num):
    org_name, org_shape = org_weights_mess[index-cur_index_plus]
    cur_name, cur_shape = cur_weights_mess[index]
    if "conv2d_transpose" in cur_name:
        # orginal weights are using nearest neighborhood
        # so it don't container wights for conv2d_transpose
        cur_index_plus += 1
        continue
    if cur_shape != org_shape:
        print(org_weights_mess[index-cur_index_plus])
        print(cur_weights_mess[index])
        raise RuntimeError
    cur_to_org_dict[cur_name] = org_name
    print("=> " + str(cur_name).ljust(50) + " : " + org_name)

with tf.name_scope("load_save"):
    name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
    restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
    load = tf.train.Saver(restore_dict)
    save = tf.train.Saver(tf.global_variables())
    for var in tf.global_variables():
        print("=> " + var.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("=> Restoring weights from:\t %s" % org_weights_path)
    load.restore(sess, org_weights_path)
    save.save(sess, cur_weights_path)
tf.reset_default_graph()
