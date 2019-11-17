# Copyright 2019 Graphcore Ltd.
"""Helper utility to download imagenet weights for Densenet model."""

import numpy as np
import tensorflow as tf


from typing import List
from pathlib import Path
from tensorflow.python.keras import backend as keras_backend
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables, load_variable

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


if __name__ is "__main__":
    get_densenet_weights(Path("./densenet_weights_fp16"))
