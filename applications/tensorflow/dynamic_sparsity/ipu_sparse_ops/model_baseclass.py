# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# A baseclass for managing a model with multiple sparse layers
import time
import logging
import numpy as np
from . import layers
from . import sparse_training
from functools import partial
import tensorflow.compat.v1 as tf
from argparse import ArgumentParser
from tensorflow.python.ipu import scopes
from logging import getLogger
import os

logger = getLogger(os.path.basename(__file__))


class SparseModelOptions(ArgumentParser):
    """
    A subclass of ArgumentParser that defines which arguments must be specified
    for a SparseModel.
    This can be used either to construct a new argparse parse or simple to add
    arguments to an existing one through the use of the add_all_arguments staticmethod
    """

    def __init__(self):
        ArgumentParser.__init__(self)
        SparseModelOptions.add_all_arguments(self)

    @staticmethod
    def add_all_arguments(parser):
        parser.add_argument("--sparsity", type=float, help="Fraction of weights which are identically zero")
        parser.add_argument("--prune-ratio", default=0, help="Fraction of non-zero weights to target when pruning.")
        parser.add_argument("--regrow-type", choices=["rigl", "random"], default=None, help="The type of algorithm to use to re-grow pruned weights")
        parser.add_argument("--sparse-matmul-options", type=dict,
                            default={
                                "metaInfoBucketOversizeProportion": 0.2,
                                "availableMemoryProportion": 0.9
                            },
                            help="Options for all sparse matmul layers")
        parser.add_argument("--disable-updating", default=False, help="Make the sparsity patterns compile time constant.")


class SparseModel:
    def __init__(self, parser, *args, **kwargs):
        self.sparse_layers = {}
        # Populate all arguments which were declared in the SparseModelOptions class
        # if attribute was not provided in the parser, then use the default value
        for attribute, default in vars(SparseModelOptions().parse_args("")).items():
            setattr(self, attribute, getattr(parser, attribute, default))
        self.random = np.random.default_rng(self.random_seed)

    def sparseLinear(self, x, sparsity, dense_length, compute_dense_grad, use_bias):
        # The underlying API requires a 2D tensor, so collapse the batch dimensions
        *batch_dimensions, input_length = x.shape.as_list()
        x = tf.reshape(x, [-1, input_length])
        x_shape = x.shape.with_rank(2).as_list()

        # Each layer should have a unique scope name
        scope_name = tf.get_default_graph().get_name_scope()
        logger.info(f"Sparse layer with scope name: {scope_name}")

        # Construct the layer if it does not exist
        if scope_name not in self.sparse_layers:
            limit = np.sqrt(6 / ((x_shape[-1] + dense_length) * (1 - sparsity)))
            uniform_gen = partial(self.random.uniform, -limit, limit)
            indices_random_gen = np.random.default_rng(seed=self.random_seed)
            sparse_layer = layers.SparseFcLayer.from_random_generator(
                dense_length, x_shape, 1 - sparsity, uniform_gen,
                indices_random_gen, name="sparse_layer", dtype=x.dtype,
                matmul_options=self.sparse_matmul_options,
                bias=use_bias, relu=False, disable_updating=self.disable_updating)

            # Create placeholders on the host, outside XLA
            with tf.init_scope():  # escapes XLA
                with tf.device("cpu"):
                    sparse_layer.create_placeholders()
                self.sparse_layers[scope_name] = sparse_layer
        else:
            # Re-use a previously defined layer
            sparse_layer = self.sparse_layers[scope_name]

        # Call the layer with the provided input
        x = sparse_layer(x, compute_dense_grad)

        # Recover the original batch dimensions
        x = tf.reshape(x, batch_dimensions + [dense_length])
        return x

    def streamSparsityFromHost(self):
        # Handles updating the sparsity representation for all layers in the model
        ops = {}
        feed_dict = {}
        with scopes.ipu_scope("/device:IPU:0"):
            for name, sparse_layer in self.sparse_layers.items():
                with tf.variable_scope(name, reuse=True):
                    ops[name] = sparse_layer.update_sparsity_op()
                    if self.prune_ratio == 0 or self.prune_ratio is None:
                        # If a layer's sparsity pattern changed then its slot
                        # also need to be updated:
                        ops[name + '_slots'] = sparse_layer.update_slots_op()
                    feed_dict.update(sparse_layer.feed_dict())

        # Combine all layer updates into one update op and run it
        session = tf.get_default_session()
        if session is None:
            raise RuntimeError("TF default session not found")
        session.run(tf.group(ops.values()), feed_dict)
        logger.info("Done streaming sparsity from host to device.")
        return True

    def streamWeightsFromDevice(self, ops):
        # This will create tensorflow ops which have to be
        # run in a session to retrieve the result
        ops = {} if ops is None else ops
        for name, sparse_layer in self.sparse_layers.items():
            with tf.variable_scope(name, reuse=True):
                weights_tensor = tf.convert_to_tensor(sparse_layer.get_values_var())
                ops[name + '_non_zeros'] = weights_tensor
        return ops

    def streamDenseGradsFromDevice(self, loss, ops=None):
        # This will create tensorflow ops which have to be
        # run in a session to retrieve the result
        ops = {} if ops is None else ops
        for name, sparse_layer in self.sparse_layers.items():
            with tf.variable_scope(name, reuse=True):
                dense_grad_w = sparse_layer.get_dense_grad_w(loss)
                ops[name + '_grad_w'] = tf.convert_to_tensor(dense_grad_w)
        return ops

    def streamOptimizerSlotsFromDevice(self, optimizer, ops=None):
        # This will create tensorflow ops which have to be
        # run in a session to retrieve the result
        ops = {} if ops is None else ops
        for name, sparse_layer in self.sparse_layers.items():
            with tf.variable_scope(name, reuse=True):
                for slot_name in sparse_layer.sparse_slots:
                    slot = sparse_layer.sparse_slots[slot_name].tf_variable
                    ops[name + f'_{slot_name}'] = tf.convert_to_tensor(slot)
        return ops

    def updateSparsityInfoOnHost(self, session_outputs):
        # Typically this can only be done during training by using the
        # outputs of the training session
        for name, sparse_layer in self.sparse_layers.items():
            slots = {slot: session_outputs[name + f'_{slot}'][0] for slot in sparse_layer.sparse_slots}
            sparse_layer.sync_internal_representation(session_outputs[name + '_non_zeros'][0], slots)
        logger.info("Done streaming sparse weights from device to host")

    def checkpointAsDense(self, checkpoint_path):
        # Before calling this function make sure the weights on host have been updated
        # by calling updateSparsityInfoOnHost
        sess = tf.get_default_session()
        assert sess is not None, "Need to be inside a session to extract weights for checkpoint"
        trainable_vars = tf.trainable_variables()
        trainable_vars = set(trainable_vars)

        # Some of the trainable vars are from the sparse layers, those are
        # converted to dense here
        dense_values = {}
        for name, sparse_layer in self.sparse_layers.items():
            with tf.variable_scope(name):
                values_var = sparse_layer.get_values_var()
                dtype = values_var.dtype.as_numpy_dtype
                dense_values[name + "/weight"] = sparse_layer.extract_dense().astype(dtype)
                trainable_vars.discard(values_var)
            if sparse_layer.bias:
                dense_values[name + "/bias"] = sess.run(sparse_layer.b.value())
                trainable_vars.discard(sparse_layer.b)

        # Now handle the remaining (non-sparse) trainable variables
        for var in trainable_vars:
            dense_values[var.op.name] = sess.run(var.value())

        # Create a graph that will just hold the variables we wish to write
        checkpoint_graph = tf.Graph()
        with checkpoint_graph.as_default():
            variables = []
            for name, dense_value in dense_values.items():
                variables.append(tf.get_variable(name, initializer=dense_value))
            s = tf.train.Saver(variables)

        # Save the dense weights and biases
        with tf.Session(graph=checkpoint_graph) as sess:
            sess.run(tf.global_variables_initializer())
            s.save(sess, checkpoint_path)

        logger.info(f"Saved sparse model as a dense checkpoint to: {checkpoint_path}")
        return

    def syncPruneAndRegrowOnHost(self, step, total_steps, session_outputs):
        # Update weight information on host using session outputs
        self.updateSparsityInfoOnHost(session_outputs)

        # Pruning schedule
        def cosine_prune_schedule(t, T, max_pruned):
            return int(np.ceil(.5 * max_pruned * (1 + np.cos(t * (np.pi / T)))))

        if step == total_steps:
            return

        # Prune and grow each sparse layer
        for layer_name, sparse_layer in self.sparse_layers.items():
            t0 = time.perf_counter()
            updater = partial(sparse_training.prune_and_grow,
                              prune_schedule=partial(cosine_prune_schedule, t=step, T=total_steps),
                              prune_ratio=self.prune_ratio,
                              grad_w=np.array(session_outputs[layer_name + '_grad_w'][0]),
                              grow_method=self.regrow_type,
                              random_gen=self.random)

            sparse_layer.update_sparsity_pattern(updater)

            # Check how long the pruning took
            t1 = time.perf_counter()
            logger.info(f"Prune and grow for layer {layer_name} completed in {t1-t0:0.3f} seconds\n")

        logger.info(f"Prune and grow for step {step} complete")
        return True
