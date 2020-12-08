# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from ipu_sparse_ops import layers, optimizers
import os
import glob
import json
import tempfile
import argparse
import numpy as np
from functools import partial
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
os.sys.path.append("../../")  # dynamic_sparsity

tf.disable_v2_behavior()
tf.disable_eager_execution()


def model(opts, use_ipu_function, input_x):
    sparse_layers = []

    # The outer function is just a Python function.
    def sparseLinear(x, dense_length, opts):
        x_shape = x.shape.with_rank(2).as_list()
        limit = np.sqrt(6 / ((x_shape[-1] + dense_length) * opts.density))
        uniform_gen = partial(np.random.uniform, -limit, limit)
        indices_random_gen = np.random.default_rng(seed=0)

        sparse_layer = layers.SparseFcLayer.from_random_generator(
            hidden_size=dense_length,
            input_shape=x_shape,
            density=opts.density,
            block_size=1,
            values_initialiser_gen=uniform_gen,
            indices_initialiser_gen=indices_random_gen,
            name="sparse_layer",
            dtype=x.dtype,
            matmul_options=opts.sparse_matmul_options,
            use_bias=opts.use_bias,
            relu=True,
            disable_updating=opts.disable_updating,
            pooling_type="NONE")

        # Create placeholders on the host, outside XLA
        with tf.init_scope():  # escapes XLA
            with tf.device("cpu"):
                sparse_layer.create_placeholders()
        sparse_layers.append(sparse_layer)

        if use_ipu_function:
            @ipu.outlined_function
            def f(x):
                # Call the layer with the provided input
                x = sparse_layer(x, opts.compute_dense_grad)
                return x
            return f(x)
        else:
            return sparse_layer(x, opts.compute_dense_grad)

    x = input_x
    outputs = {}
    # Loop through n_layers which all use the same shape,
    # and therefore the same ipu_function
    for i in range(opts.n_layers):
        with tf.variable_scope(f"sparse_{i}", use_resource=True):
            x = sparseLinear(x, opts.hidden_length, opts)
            outputs[f"activation_{i}"] = x
    loss = tf.reduce_sum(x)
    # Construct a sparse optimizer as usual
    optimizer = optimizers.SparseOptimizer(tf.train.AdamOptimizer)
    optimizer = optimizer(sparse_layers=sparse_layers)

    g = optimizer.compute_gradients(loss)

    # Record the grads for comparison
    for grad, var in g:
        outputs[var.name + "_grad"] = grad
    if opts.compute_dense_grad:
        for i, layer in enumerate(sparse_layers):
            outputs["layer_{i}_gradW"] = layer.get_dense_grad_w(tf.reduce_sum(loss))
    outputs["input_grad"] = tf.gradients(loss, input_x)[0]

    return outputs


def run_without_ipu_functions(opts, input_values):
    g = tf.Graph()
    with g.as_default():
        with tf.device("cpu"):
            input_x = tf.placeholder(np.float32, input_values.shape, name="input_x")

        with ipu_scope("/device:IPU:0"):
            result = ipu.ipu_compiler.compile(partial(model, opts, False), [input_x])

        with tf.device("cpu"):
            variables = [var for var in tf.global_variables() if var.name.count("metainfo") == 1]
            variables += tf.trainable_variables()
            saver = tf.train.Saver(variables)
            initializer = tf.global_variables_initializer()

    g.finalize()
    with tf.Session(graph=g) as sess:
        sess.run(initializer)
        saver.save(sess, opts.checkpoint_path)  # save variables to use in second run
        results = sess.run(result, feed_dict={input_x: input_values})
    return results


def run_with_ipu_functions(opts, input_values):
    g = tf.Graph()
    with g.as_default():
        with tf.device("cpu"):
            input_x = tf.placeholder(np.float32, input_values.shape, name="input_x")

        with ipu_scope("/device:IPU:0"):
            result = ipu.ipu_compiler.compile(partial(model, opts, True), [input_x])

        with tf.device("cpu"):
            variables = [var for var in tf.global_variables() if var.name.count("metainfo") == 1]
            variables += tf.trainable_variables()
            loader = tf.train.Saver(variables)
            initializer = tf.global_variables_initializer()

    g.finalize()
    with tf.Session(graph=g) as sess:
        sess.run(initializer)
        loader.restore(sess, opts.checkpoint_path)  # restore the weights used in previous run
        results = sess.run(result, feed_dict={input_x: input_values})

    return results


def verify_outputs(results_with_functions, results_without_functions):
    # Make sure activations match
    rtol = 1e-05
    atol = 1e-05
    for key in results_with_functions:
        gold = results_without_functions[key]
        pred = results_with_functions[key]
    np.testing.assert_allclose(gold, pred, atol=atol, rtol=rtol, err_msg=f"Mismatch in value of {key}")

    print("All asserts pass.")


if __name__ == "__main__":
    # Program arguments
    parser = argparse.ArgumentParser()
    defaults = dict(
        density=0.4,
        n_layers=4,
        hidden_length=64,
        compute_dense_grad=True,
        use_bias=False,
        disable_updating=False,
        checkpoint_path=tempfile.mkdtemp(),
        report_path="./reports",
        sparse_matmul_options={"metaInfoBucketOversizeProportion": 0.2,
                               "availableMemoryProportion": 0.9}
    )
    parser.set_defaults(**defaults)
    opts = parser.parse_args()

    # Enable profiling in order to check if functions were used
    cfg = ipu.utils.create_ipu_config(profiling=True, profile_execution=True,
                                      enable_poplar_serialized_graph=True,
                                      report_directory=opts.report_path)
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    # Run once without functions and once without functions then compare
    input_values = np.random.randn(2, opts.hidden_length).astype(np.float32)
    results_without_functions = run_without_ipu_functions(opts, input_values)
    results_with_functions = run_with_ipu_functions(opts, input_values)
    verify_outputs(results_with_functions, results_without_functions)
