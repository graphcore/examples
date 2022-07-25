# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# 'Regression of 3D Sky Map to Cosmological Parameters (CosmoFlow)'
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.

# This file has been modified by Graphcore Ltd.

"""
Random dummy dataset specification.
"""

# System
import math

# Externals
import tensorflow as tf


def construct_dataset(sample_shape, target_shape, micro_batch_size=1, n_samples=32):

    datatype = tf.float32

    input_data = tf.random.uniform(
        sample_shape,
        dtype=datatype,
        name='synthetic_inputs')

    labels = tf.random.uniform(
        target_shape,
        dtype=datatype,
        name='synthetic_labels')

    data = tf.data.Dataset.from_tensors((input_data, labels))

    data = data.repeat(n_samples)
    data = data.batch(batch_size=micro_batch_size, drop_remainder=True)
    data = data.cache()
    data = data.repeat()
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def get_datasets(sample_shape, target_shape, micro_batch_size,
                 n_train, n_valid, n_epochs=None, shard=False,
                 rank=0, n_ranks=1):
    train_dataset = construct_dataset(sample_shape, target_shape, micro_batch_size=micro_batch_size)
    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, micro_batch_size=micro_batch_size)
    n_train_steps = n_train // micro_batch_size
    n_valid_steps = n_valid // micro_batch_size
    if shard:
        n_train_steps = n_train_steps // n_ranks
        n_valid_steps = n_valid_steps // n_ranks

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
