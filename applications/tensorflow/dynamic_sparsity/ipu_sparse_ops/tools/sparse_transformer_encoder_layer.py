# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tempfile
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from ipu_sparse_ops import sparse, optimizers
import os
import logging

os.sys.path.append("../../")  # dynamic_sparsity
from ipu_sparse_ops.model_baseclass import SparseModelOptions  # noqa: E402
from ipu_sparse_ops.transformer.transformer_baseclass import TransformerOptions   # noqa: E402
from ipu_sparse_ops.transformer.transformer_dense import DenseTransformer  # noqa: E402
from ipu_sparse_ops.transformer.transformer_dynsparse import DynsparseTransformer  # noqa: E402

# disable TF 2.0
tf.disable_eager_execution()
tf.disable_v2_behavior()


def get_program_arguments():
    transformer_parser = TransformerOptions()
    SparseModelOptions.add_all_arguments(transformer_parser)
    transformer_parser.add_argument("--profile", action="store_true",
                                    help="Enable profiling for mem profile")
    default_settings = dict(
        dtype=tf.float32,
        source_sequence_length=12,
        hidden_length=16,
        ff_length=64,
        attention_heads=1,
        qkv_length=16,
        sparsity=0.9,
        batch_size=1,
        random_seed=11,
        pooling_type='NONE',
        dropout_keep_prob=1
    )
    transformer_parser.set_defaults(**default_settings)
    return transformer_parser.parse_args()


def stream_dense_grads_from_device(transformer, loss, ops=None):
    # This will create tensorflow ops which have to be
    # run in a session to retrieve the result
    ops = {} if ops is None else ops
    for name, sparse_layer in transformer.sparse_layers.items():
        with tf.variable_scope(name, reuse=True):
            dense_grad_w = sparse_layer.get_dense_grad_w(loss)
            ops[name + '_grad_w'] = tf.convert_to_tensor(dense_grad_w)
    return ops


def sparse_transformer_fwd_and_grad(transformer, input_activation):
    transformer.compute_dense_grad = True
    output_activation = transformer.encoder_layer(input_activation, mask=None, compute_dense_grad=True, debug_name="layer_0")
    loss = tf.reduce_sum(output_activation)

    # Wrap the optimizer (this would help manage the slot variables)
    optimizer = optimizers.SparseOptimizer(tf.train.AdamOptimizer)
    optimizer = optimizer(learning_rate=1e-3, sparse_layers=transformer.sparse_layers.values())

    grads = optimizer.compute_gradients(loss)
    input_grad = tf.gradients(loss, input_activation)[0]
    with tf.control_dependencies([input_grad]):
        train_op = optimizer.apply_gradients(grads)

    with tf.control_dependencies([train_op]):
        streamOps = {"output_activation": output_activation}
        streamOps["input_grad"] = input_grad
        # Sparse grads
        for grad, var in grads:
            streamOps[var.op.name + "_grad"] = grad
        # Dense grads
        stream_dense_grads_from_device(transformer, loss, streamOps)
        return streamOps


def dense_transformer_fwd_and_grad(transformer, input_activation):
    output_activation = transformer.encoder_layer(input_activation, mask=None, debug_name="layer_0")
    loss = tf.reduce_sum(output_activation)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    grads = optimizer.compute_gradients(loss)
    input_grad = tf.gradients(loss, input_activation)[0]
    with tf.control_dependencies([input_grad]):
        train_op = optimizer.apply_gradients(grads)

    with tf.control_dependencies([train_op]):
        streamOps = {"output_activation": output_activation}
        streamOps["input_grad"] = input_grad
        for grad, var in grads:
            streamOps[var.op.name + "_grad"] = grad
        return streamOps


def main(args):
    tf.logging.set_verbosity(tf.logging.ERROR)
    np.set_printoptions(linewidth=200)
    random_seed = args.random_seed
    checkpoint_path = os.path.join(tempfile.mkdtemp(), "model.ckpt")

    # Input activations for the attention layer
    random_gen = np.random.default_rng(seed=random_seed)
    activations_np = random_gen.uniform(-0.1, 0.1, size=(args.batch_size, args.source_sequence_length, args.hidden_length))

    # Configure the IPU
    cfg = ipu.utils.create_ipu_config(profiling=args.profile, report_directory="./report/")
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    # Build IPU graphs
    sparse_decoder_graph = tf.Graph()
    sparse_transformer = DynsparseTransformer(args)
    with sparse_decoder_graph.as_default():
        with tf.device("cpu"):
            # placeholder for activations
            # weight placeholders are created inside sparse_transfomer
            inputs_ph = tf.placeholder(args.dtype, activations_np.shape)
        with ipu.scopes.ipu_scope("/device:IPU:0"):
            sparse_decoder = partial(sparse_transformer_fwd_and_grad, sparse_transformer)
            sparse_decoder_fetches = ipu.ipu_compiler.compile(sparse_decoder, [inputs_ph])
            ipu.utils.move_variable_initialization_to_cpu()

    # sparse-decoder
    with tf.Session(graph=sparse_decoder_graph) as sess:
        # initialize weights
        sess.run(tf.global_variables_initializer())

        # Save the sparse weights to checkpoint as dense
        sparse_transformer.checkpointAsDense(checkpoint_path)

        # run sparse decoder
        sparse_result = sess.run(sparse_decoder_fetches, feed_dict={inputs_ph: activations_np})

    # Create a dense transformer and initialize the weights to the values that
    # the sparse model was initialzed with originally
    dense_decoder_graph = tf.Graph()
    dense_transformer = DenseTransformer(args)
    with dense_decoder_graph.as_default():
        with tf.device("cpu"):
            # placeholder for activations
            # weights will get streamed from checkpoint
            inputs_ph = tf.placeholder(args.dtype, activations_np.shape)

        with ipu.scopes.ipu_scope("/device:IPU:0"):
            dense_decoder_fetches = partial(dense_transformer_fwd_and_grad, dense_transformer)
            dense_graph = ipu.ipu_compiler.compile(dense_decoder_fetches, [inputs_ph])
            ipu.utils.move_variable_initialization_to_cpu()

        with tf.device("cpu"):
            # We will only load the trainable variables, not momentum etc.
            loader = tf.train.Saver(tf.trainable_variables())

    # dense-decoder
    with tf.Session(graph=dense_decoder_graph) as sess:
        # Initialized momentums which are not part of the checkpoint
        sess.run(tf.global_variables_initializer())
        # Restore saved trainable variables
        loader.restore(sess, checkpoint_path)
        dense_result = sess.run(dense_graph, feed_dict={inputs_ph: activations_np})

    # TEST
    rtol = 1e-05
    atol = 1e-05
    if args.dtype == tf.float16:
        rtol = 1e-04
        atol = 1e-02
    # Compare model output activations (actual vs. desired) -> (sparse vs. dense)
    np.testing.assert_allclose(sparse_result["output_activation"], dense_result["output_activation"],
                               atol=atol, rtol=rtol, err_msg="Output activations do not match.")

    # Compate gradient of output wrt. input
    np.testing.assert_allclose(sparse_result["input_grad"], dense_result["input_grad"],
                               atol=atol, rtol=rtol, err_msg="Grads wrt. inputs do not match")

    # Compare the dense_w and sparse grads of every sparse layer
    for name, sparse_layer in sparse_transformer.sparse_layers.items():
        # Compate the dense grads
        dense_grad = dense_result[name + "/weight" + "_grad"]
        sparse_grad_w = sparse_result[name + "_grad_w"]
        np.testing.assert_allclose(sparse_grad_w, dense_grad, atol=atol, rtol=rtol,
                                   err_msg=f"Dense grads for layer {name} do not match")

        # Compare the sparse grads
        sparse_grad_padded = sparse_result[name + "/sparse_layer/nz_values_grad"]
        sparse_grad_data = sparse.SparseRepresentation(sparse_layer.weights.get_metainfo(), sparse_grad_padded)
        i, j, sparse_grad = sparse.triplets_from_representation(sparse_layer.weights.spec, sparse_grad_data, sparse_layer.weights.matmul_options)

        # Convert dense grads to blocks
        block_size, _ = sparse_layer.get_nonzero_blocks_shape()
        nx, ny = dense_grad.shape[0] // block_size, dense_grad.shape[1] // block_size
        strides = np.array(dense_grad.strides)  # strides are in bytes
        strides = tuple(strides * block_size) + tuple(strides)
        blocked_dense_grad = np.lib.stride_tricks.as_strided(dense_grad, (nx, ny, block_size, block_size), strides)
        blocked_dense_grad = np.squeeze(np.copy(blocked_dense_grad))  # this will squeeze out the special case block size 1
        np.testing.assert_allclose(sparse_grad, blocked_dense_grad[i, j], atol=atol, rtol=rtol,
                                   err_msg=f"Sparse grads for layer {name} do not match")

    print("All results match.")
    return sparse_result, dense_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.getLevelName("DEBUG"),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    args = get_program_arguments()
    a, b = main(args)
