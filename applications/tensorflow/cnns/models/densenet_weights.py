# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
"""Helper utility to download imagenet weights for Densenet model."""

from pathlib import Path
from typing import List

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.framework.python.framework.checkpoint_utils import \
    list_variables, load_variable
from tensorflow.python.keras import backend as keras_backend
from tensorflow.python.keras.applications.densenet import DenseNet121

keras_backend.set_floatx('float16')


def get_densenet_weights(save_dir: Path = Path('densenet_121')) -> Path:
    """Download pre-trained imagenet weights for densenet model.

    Args:
        save_dir: Path to where checkpoint must be downloaded.

    Returns: Path to checkpoint file.

    """
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        keras_backend.set_session(sess)
        save_dir.mkdir(parents=True, exist_ok=True)
        _ = DenseNet121(weights='imagenet')
        saver = tf.train.Saver()
        return saver.save(sess, Path(save_dir, "densenet_model.ckpt").as_posix())


def load_fp32_weights_into_fp16_vars(checkpoint_path: Path) -> List:
    """Load fp32 weights from checkpoint path into fp16 variables.

    Assumes that caller has executed `tf.run(tf.global_variables_initializer())`

    Args:
        checkpoint_path: Checkpoint path

    Returns:
        Collection of ops to use to restore the weights in the graph.
    """
    checkpoint_variables = [var_name for var_name, _ in list_variables(checkpoint_path)]

    for graph_var in tf.global_variables():
        if graph_var.op.name in checkpoint_variables:
            var = load_variable(checkpoint_path, graph_var.op.name)
            weights = tf.cast(var, tf.float16) if var.dtype == np.float32 else var
            tf.add_to_collection('restore_ops', graph_var.assign(weights))
    return tf.get_collection('restore_ops')
