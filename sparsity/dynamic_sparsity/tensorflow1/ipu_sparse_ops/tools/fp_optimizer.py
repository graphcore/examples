# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python import ipu
from functools import partial
import numpy as np
import argparse
import time
import os
from ipu_sparse_ops.fp_slot_opt import SelectableSlotFPFormatOptimizer


def parse_args():
    parser = argparse.ArgumentParser("Test program to make sure the custom optimiser designed "
                                     "to allow for choosing the dtype of slots independently "
                                     "from the model works fine")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"],
                        help="Sets the model variables dtype")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs to run.")
    parser.add_argument("--slots-fp-type", type=str, default="float32", choices=["float16", "float32"],
                        help="sets the slots dtype in the custom optimizer")
    parser.add_argument("--force-fp32-weight-update", action='store_true', help="Forces the weight "
                        "update to be performed in fp32, independentely from the model and slots dtype, "
                        "when the custom opt is used")
    parser.add_argument("--use-nesterov", action="store_true", help="Whether to use the nesterov update "
                        "method in the custom optimizer or not")
    return parser.parse_args()


def create_ipu_config():
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()


def create_dataset(x, y, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache().repeat()
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# parse args
opts = parse_args()

# create config
create_ipu_config()

# set a seed for reproducibility
tf.random.set_random_seed(0)
np.random.seed(0)

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), _ = fashion_mnist.load_data()

# Normalize the data
x_train = x_train.astype(opts.dtype) / 255
y_train = y_train.astype('int32')

batchsize = 32
train_device_iterations = len(x_train) // batchsize

# Create the infeeds and outfeeds
infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
    create_dataset(x_train, y_train, batchsize))
infeed_queue_2 = ipu.ipu_infeed_queue.IPUInfeedQueue(
    create_dataset(x_train, y_train, batchsize))
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
outfeed_queue_2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

# create weights initialisation matrix
init_hidden = np.random.normal(size=(28 * 28, 128)).astype(opts.dtype)
init_classifier = np.random.normal(size=(128, 10)).astype(opts.dtype)


def train_fn(infeed_queue, outfeed_queue, name, use_custom_op=False):
    def body(total_loss, x, y):
        with tf.variable_scope(name):
            x_flat = tf.layers.Flatten()(x)
            hidden = tf.layers.dense(x_flat, 128, name="hidden", kernel_initializer=tf.keras.initializers.Constant(init_hidden))
            logits = tf.layers.dense(hidden, 10, kernel_initializer=tf.keras.initializers.Constant(init_classifier))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
            opt_cls = tf.train.AdamOptimizer
            opt_kwargs = {}
            if use_custom_op:
                opt_cls = SelectableSlotFPFormatOptimizer(opt_cls)
                opt_kwargs['slots_dtype'] = opts.slots_fp_type
                opt_kwargs['force_fp32_weight_update'] = opts.force_fp32_weight_update
                opt_kwargs['use_nesterov'] = opts.use_nesterov
            optimizer = opt_cls(learning_rate=0.01, **opt_kwargs)
            train_op = optimizer.minimize(loss=loss)

            with tf.control_dependencies([train_op]):
                with tf.variable_scope("hidden", reuse=True):
                    w_hidden = tf.get_variable("kernel")
                out = outfeed_queue.enqueue({"weights": w_hidden,
                                             "m": optimizer.get_slot(w_hidden, "m"),
                                             "v": optimizer.get_slot(w_hidden, "v")})

            with tf.control_dependencies([out]):
                return loss + total_loss

    with tf.variable_scope(name):
        total_loss = 0.0
        total_loss = ipu.loops.repeat(train_device_iterations, body, [total_loss], infeed_queue)
        return total_loss / train_device_iterations


with ipu.scopes.ipu_scope('/device:IPU:0'):
    train = ipu.ipu_compiler.compile(partial(train_fn, infeed_queue=infeed_queue,
                                             outfeed_queue=outfeed_queue,
                                             use_custom_op=False,
                                             name="default"), [])
    train_with_custom_opt = ipu.ipu_compiler.compile(
        partial(train_fn, infeed_queue=infeed_queue_2,
                outfeed_queue=outfeed_queue_2,
                use_custom_op=True,
                name="custom"), [])

    dequeue = outfeed_queue.dequeue()
    dequeue_2 = outfeed_queue_2.dequeue()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(infeed_queue.initializer)
    sess.run(infeed_queue_2.initializer)

    for i in range(opts.epoch):
        print("epoch", i)

        loss = sess.run(train)
        out = sess.run(dequeue)
        loss_2 = sess.run(train_with_custom_opt)
        out_2 = sess.run(dequeue_2)

        print("Base loss: ", loss, "Loss with custom optimizer:", loss_2)

        v = out["v"]
        v_2 = out_2["v"]
        m = out["m"]
        m_2 = out_2["m"]
        w = out["weights"]
        w_2 = out_2["weights"]

        print(f"Last step max abs diff v: \n{np.max(np.abs(v[-1] - v_2[-1]))}")
        print(f"Last step max abs diff m: \n{np.max(np.abs(m[-1] - m_2[-1]))}")

    # Set tolerances appropriately as numpy is set for doubles by default:
    if opts.dtype == 'float16':
        rtol = 1e-03
        atol = 1e-04
    else:
        rtol = 1e-05
        atol = 1e-06

    if not np.allclose(m, m_2, rtol=rtol, atol=atol) or not np.allclose(v, v_2, rtol=rtol, atol=atol):
        raise Exception(f"Slots don't match. Max abs error for m=\n{np.max(np.abs(m - m_2))}"
                        f"\nMax abs error for v =\n{np.max(np.abs(v - v_2))}"
                        f"\nFinal abs error for m = \n{np.max(np.abs(m[-1] - m_2[-1]))}"
                        f"\nFinal abs error for v = \n{np.max(np.abs(v[-1] - v_2[-1]))}"
                        f"\nAbs diff v: \n{np.abs(v - v_2)}\n"
                        f"\nAbs diff m: \n{np.abs(m - m_2)}")

    if not np.allclose(w, w_2):
        raise Exception(f"Weights don't match. Max absolute error=\n{np.max(np.abs(w - w_2))}")

print("All asserts pass.")
