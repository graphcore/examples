# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python import ipu
tf.disable_v2_behavior()

SIZE = 5


def add_op(x, y):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [tf.TensorShape([SIZE])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([x, y],
                                              lib_path,
                                              gp_path,
                                              outs=outputs)


if __name__ == '__main__':
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    with tf.device("cpu"):
        x_data = tf.placeholder(np.float32, [SIZE])
        y_data = tf.placeholder(np.float32, [SIZE])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(add_op, [x_data, y_data])

    with tf.Session() as sess:
        a = np.random.rand(SIZE)
        b = np.random.rand(SIZE)

        result = sess.run(xla_result, feed_dict={x_data: a, y_data: b})

    # Show result from the IPU:
    print("IPU:", result[0])
    # Same calculation on host for comparison:
    print("numpy:", a + b)
