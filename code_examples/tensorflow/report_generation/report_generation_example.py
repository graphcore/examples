# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils

datatype = tf.float16
num_units_in = 512
num_units_out = 1024


# example graph
def inference(x):
    with tf.variable_scope('fully_connected', use_resource=True):
        weights = tf.get_variable('weights', [num_units_in, num_units_out],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=datatype)
        biases = tf.get_variable('biases', [num_units_out], initializer=tf.constant_initializer(0.0), dtype=datatype)

        x = tf.nn.xw_plus_b(x, weights, biases)

    return x


# Main code

x = tf.placeholder(datatype, shape=[1, num_units_in])

with tf.device("/device:IPU:0"):
    logits = inference(x)

with tf.device('cpu'):
    # event trace
    report = gen_ipu_ops.ipu_event_trace()

# create a config with profiling on
opts = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
utils.configure_ipu_system(opts)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
# uncomment the following line if you're not interested in the report relative to the variable initializer graph
# sess.run(report)

training_data = np.zeros([1, num_units_in])
# run the graph
sess.run(logits, feed_dict={x: training_data})
# get the event trace
out = sess.run(report)
# extract the report
rep = utils.extract_all_strings_from_event_trace(out)
with open("report.txt", "w") as f:
    f.write(rep)

sess.close()
