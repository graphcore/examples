# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The logging code used in train.py
"""

import os
import csv
import random
import string
import json
import tensorflow as tf


def add_arguments(parser):
    group = parser.add_argument_group('logging')
    group.add_argument('--log-path', type=str, default="./logs/", help="log and weights save directory")

    return parser


def print_setting(opts, is_dien=True, is_training=False):
    if is_dien:
        log_str = ("CTR Model: \n"
                   " Max Sequence Length {max_seq_len}\n"
                   " Hidden Size {hidden_size}\n"
                   " Attension Size {attention_size}\n"
                   "CTR Dataset: \n"
                   " Use Synthetic Data {use_synthetic_data}\n"
                   " Epochs: {epochs}\n"
                   " Batches Per Step {batches_per_step}\n"
                   "CTR Training: \n"
                   " Batch Size {batch_size}\n"
                   " Learning Rate {learning_rate}\n"
                   " Replicas {replicas}\n")
    else:
        if is_training:
            log_str = (" Max Sequence Length {max_seq_len}\n"
                       " Hidden Size {hidden_size}\n"
                       " Attension Size {attention_size}\n"
                       " Precision {precision}\n"
                       " GRU Type {gru_type}\n"
                       " AUGRU Type {augru_type}\n"
                       " Epochs: {epochs}\n"
                       " Batches Per Step {batches_per_step}\n"
                       " Batch Size {batch_size}\n"
                       " Replicas {replicas}\n"
                       " Optimizer {optimizer}\n"
                       " LearningRate {learning_rate}\n"
                       " ModelPath {model_path}\n")
        else:
            log_str = (" Max Sequence Length {max_seq_len}\n"
                       " Hidden Size {hidden_size}\n"
                       " Attension Size {attention_size}\n"
                       " Precision {precision}\n"
                       " GRU Type {gru_type}\n"
                       " AUGRU Type {augru_type}\n"
                       " Use Synthetic Data {use_synthetic_data}\n"
                       " Epochs: {epochs}\n"
                       " Batches Per Step {batches_per_step}\n"
                       " Batch Size {batch_size}\n"
                       " Replicas {replicas}\n")

    print(log_str.format(**opts))


def print_to_file_and_screen(string, opts):
    print(string)
    if opts["log_path"]:
        if not os.path.exists(opts["log_path"]):
            os.makedirs(opts["log_path"])
        with open(os.path.join(opts["log_path"], 'log.txt'), "a+") as f:
            f.write(str(string) + '\n')



def get_trainable_var_count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        name = variable.name
        shape = variable.get_shape()
        print("var name: {}, shape: {}".format(name, shape))
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print("params: {}".format(variable_parameters))
        total_parameters += variable_parameters
    print("total params: {}".format(total_parameters))


def get_update_ops():
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if not update_ops:
        print("empty list")
    else:
        print("list is not empty")


# Total flops
def get_graph_complexity():
    g = tf.get_default_graph()
    run_meta = tf.RunMetadata()
    # Profile flops
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    print('total FLOPS with init: ', flops.total_float_ops)


# Grad or omit init flops
def get_graph_complexity_scope():
    g = tf.get_default_graph()

    run_meta = tf.RunMetadata()

    # Profile flops
    # In profile option builder:
    # use for grad flops            .with_node_names(show_name_regexes=['.*radient.*'])
    # use for omitting init flops   .with_node_names(hide_name_regexes=['.*Initializer.*'])
    opts = (tf.profiler.ProfileOptionBuilder(
            tf.profiler.ProfileOptionBuilder.float_operation())
            .order_by('name')
            .with_node_names(show_name_regexes=['.*radient.*'])
            # .with_node_names(hide_name_regexes=['.*Initializer.*'])
            .account_displayed_op_only(True)
            .with_file_output('omit-init_profile-bs-128.txt')
            .build())
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='scope', options=opts)
    print('total FLOPS: ', flops.total_float_ops)
