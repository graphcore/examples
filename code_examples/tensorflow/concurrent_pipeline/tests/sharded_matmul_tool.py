# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import jax_utils
from custom_ops import sharded
from utils import ClosureInitializer


def calc_reference_result_and_grad(a, b, labels):
    loss, logits = jax_utils.projection_softmax_cross_entropy(a, b, labels)
    grad_fns = jax_utils.projection_softmax_cross_entropy_grad_fns()
    return logits, loss, grad_fns[0](a, b, labels), grad_fns[1](a, b, labels)


def take_last_axis(logits, labels):
    batch_size = labels.get_shape().as_list()[0]
    batch_indices = tf.constant(np.arange(batch_size, dtype=np.int32))
    batch_indices = tf.reshape(batch_indices, [batch_size, 1])
    positions = tf.stack([batch_indices, labels], axis=-1)
    return tf.gather_nd(logits, positions, name="take_last_axis")


def ipu_computation(input, labels, weight_shape, weight_initialiser):
    opts = {
        "availableMemoryProportion": "0.3",
        "partialsType": "half"
    }
    weights = tf.get_variable('weights', shape=weight_shape, trainable=True, initializer=weight_initialiser)
    inputSharded = sharded.to_all(input, args.ipus)
    labelsSharded = sharded.to_all(labels, args.ipus)
    output = sharded.matmul(inputSharded, weights, opts, name="custom_matmul")
    ce = sharded.log_softmax_cross_entropy(output, labelsSharded)

    loss = tf.reduce_mean(ce, name="final_loss")
    grads = tf.gradients(loss, [input, weights])
    return output, loss, grads[0], grads[1]


def loss_fn(logits, loss, grads0, grads1):
    return logits, loss, grads0, grads1


def identity_stage(input, labels):
    return tf.identity(input), tf.identity(labels)


outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()


def pipelined_test(lhs, label_indices):

    def bound_ipu_computation(input, labels):
        rhs_init = ClosureInitializer(lambda: tf.constant(rhs))
        return ipu_computation(input, labels, weight_shape=rhs.shape, weight_initialiser=rhs_init)

    with tf.variable_scope("Test", use_resource=True):
        pipeline_op = ipu.pipelining_ops.pipeline(
            computational_stages=[identity_stage, bound_ipu_computation, loss_fn],
            device_mapping=[0, ipu.pipelining_ops._ALL_DEVICES, args.ipus - 1],  # Need to reference last IPU
            gradient_accumulation_count=1,
            repeat_count=1,
            inputs=[lhs, label_indices],
            infeed_queue=None,
            outfeed_queue=outfeed_queue,
            optimizer_function=None,
            pipeline_schedule=ipu.pipelining_ops.PipelineSchedule.Sequential,
            outfeed_loss=False,
            name="Pipeline")
        return pipeline_op


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32,
                        help="Rows in LHS matrix.")
    parser.add_argument("--cols", type=int, default=512,
                        help="Columns in LHS matrix.")
    parser.add_argument("--outsize", type=int, default=100,
                        help="Output size (columns in RHS matrix).")
    parser.add_argument("--ipus", type=int, default=2,
                        help="Number of IPUs (number of shards for matmul).")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    rng = np.random.default_rng(seed=101)

    lhs_shape = [args.rows, args.cols]
    rhs_shape = [args.cols, args.outsize]

    def glorot_limit(shape):
        return np.sqrt(6.0 / (shape[0] + shape[1]))

    # Test with a distribution of values likely to occur in practice:
    limit = glorot_limit(rhs_shape)
    lhs = rng.normal(0, 1, size=lhs_shape).astype(np.float32)
    rhs = rng.uniform(-limit, limit, size=rhs_shape).astype(np.float32)
    labels = rng.uniform(0, args.outsize, size=args.rows).astype(np.int32)
    labels = np.expand_dims(labels, axis=1)
    jax_logits, jax_loss, jax_dLhs, jax_dRhs = calc_reference_result_and_grad(lhs, rhs, labels)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = args.ipus
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        lhs_ph = tf.placeholder(np.float32, lhs_shape)
        labels_ph = tf.placeholder(np.int32, labels.shape)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        ipu_output = ipu.ipu_compiler.compile(pipelined_test, [lhs_ph, labels_ph])
        outfeed_op = outfeed_queue.dequeue()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(ipu_output, feed_dict={lhs_ph: lhs, labels_ph: labels})
        custom_logits, custom_loss, custom_dLhs, custom_dRhs = sess.run(outfeed_op)

    # Tolerances suitable for fp32:
    rtol = 1e-05
    atol = 1e-04

    # Check loss matches:
    print(f"JAX loss: {jax_loss}\nOur loss: {custom_loss}")
    if not np.allclose(custom_loss, jax_loss, rtol=rtol, atol=atol):
        raise RuntimeError("Loss does not match reference.")

    # Check output:
    if not np.allclose(custom_logits, jax_logits, rtol=rtol, atol=atol):
        print(f"ABS Max error: {np.max(np.abs(custom_logits - jax_logits))}")
        raise RuntimeError("Logits do not match reference.")

    # Check grads:
    if not np.allclose(custom_dLhs, jax_dLhs, rtol=rtol, atol=atol):
        print(f"Max ABS error: {np.max(np.abs(custom_dLhs - jax_dLhs))}")
        raise RuntimeError("dL/dLhs does not match reference.")
    if not np.allclose(custom_dRhs, jax_dRhs, rtol=rtol, atol=atol):
        print(f"Max ABS error: {np.max(np.abs(custom_dRhs - jax_dRhs))}")
        raise RuntimeError("dL/dRhs does not match reference.")

    print("Results match.")
