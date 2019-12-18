# Copyright 2020 Graphcore Ltd.
import os
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

size = 5

with tf.device("cpu"):
    x_data = tf.placeholder(np.float32, [size])
    y_data = tf.placeholder(np.float32, [size])


def add_op(x, y):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [tf.TensorShape([size])],
    }

    base_path = os.getcwd()
    lib_path = os.path.join(base_path, "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([x, y],
                                              lib_path,
                                              gp_path,
                                              outs=outputs)


with ipu_scope("/device:IPU:0"):
    xla_result = ipu.ipu_compiler.compile(add_op, [x_data, y_data])

with tf.Session() as sess:
    a = np.random.rand(size)
    b = np.random.rand(size)

    result = sess.run(xla_result, feed_dict = {x_data: a, y_data: b})

# Show result from the IPU:
print("IPU:", result[0])

# Same calculation on host for comparison:
print("numpy:", a + b)
