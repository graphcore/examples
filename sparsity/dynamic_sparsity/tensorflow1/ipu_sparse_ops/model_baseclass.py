# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# A baseclass for managing a model with multiple sparse layers
import time
import json
import numpy as np
from ipu_sparse_ops import layers
from ipu_sparse_ops import sparse_training
from functools import partial
import tensorflow.compat.v1 as tf
from argparse import ArgumentParser
from tensorflow.python.ipu import outlined_function
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
        parser.add_argument("--sparsity", type=float, default=0.9, help="Fraction of weights which are identically zero")
        parser.add_argument("--block-size", type=int, default=1, help="Set to 1 for element level sparsity, larger for block-level sparsity")
        parser.add_argument("--prune-ratio", type=float, default=0, help="Fraction of non-zero weights to target when pruning.")
        parser.add_argument("--regrow-type", type=str, choices=["rigl", "random"], default='rigl', help="The type of algorithm to use to re-grow pruned weights")
        parser.add_argument("--sparse-matmul-options", type=json.loads,
                            default={
                                "metaInfoBucketOversizeProportion": 0.2,
                                "availableMemoryProportion": 0.9
                            },
                            help="Options for all sparse matmul layers")
        parser.add_argument("--dense-grad-matmul-options", type=json.loads,
                            default={
                                "availableMemoryProportion": 0.1,
                                "partialsType": "float"
                            },
                            help="Options for all dense gradient matmuls in sparse layers")
        parser.add_argument("--disable-updating", action='store_true', help="Make the sparsity patterns compile time constant.")
        parser.add_argument('--pooling-type', type=str, default='NONE', choices=['NONE', 'SUM', 'AVG', 'MAX'],
                            help="Select dense gradients block pooling")
        parser.add_argument('--no-outline-sparse-layer', action='store_true', help="Disable per-sparse layer outlining.")
        parser.add_argument("--cosine-prune-schedule", type=json.loads,
                            default={
                                'zero_steps': 0,
                                'phase_delay': 0,
                                'period': 0.5
                                },
                            help="Fine grained control of the pruning schedule.")


class SparseModel:
    def __init__(self, parser, *args, **kwargs):
        self.sparse_layers = {}
        # Populate all arguments which were declared in the SparseModelOptions class
        # if attribute was not provided in the parser, then use the default value
        for attribute, default in vars(SparseModelOptions().parse_args("")).items():
            setattr(self, attribute, getattr(parser, attribute, default))
        self.random = np.random.default_rng(self.random_seed)

        # Piggy-back dense grad matmul options on the existing sparse opts
        self.sparse_matmul_options['dense_grad'] = self.dense_grad_matmul_options
        del self.dense_grad_matmul_options

    def getOrCreateSparseLinear(self, x_shape, x_dtype, sparsity, dense_length, block_size, use_bias, override_partials_type=None):
        x_dtype = tf.as_dtype(x_dtype)

        # Each layer should have a unique scope name
        scope_name = tf.get_default_graph().get_name_scope()
        logger.info(f"Sparse layer with scope name: {scope_name}")

        # Construct the layer if it does not exist
        if scope_name not in self.sparse_layers:
            layer_matmul_options = self.sparse_matmul_options
            if override_partials_type:
                layer_matmul_options['partialsType'] = override_partials_type
            limit = np.sqrt(6 / ((x_shape[-1] + dense_length) * (1 - sparsity)))
            uniform_gen = partial(self.random.uniform, -limit, limit)
            indices_random_gen = np.random.default_rng(seed=self.random_seed)
            sparse_layer = layers.SparseFcLayer.from_random_generator(
                dense_length, x_shape, 1 - sparsity,
                block_size=block_size,
                values_initialiser_gen=uniform_gen,
                indices_initialiser_gen=indices_random_gen,
                name="sparse_layer", dtype=x_dtype,
                matmul_options=layer_matmul_options,
                use_bias=use_bias, relu=False,
                disable_updating=self.disable_updating,
                pooling_type=self.pooling_type)

            # Create placeholders on the host, outside XLA
            with tf.init_scope():  # escapes XLA
                with tf.device("cpu"):
                    sparse_layer.create_placeholders()
                self.sparse_layers[scope_name] = sparse_layer
        else:
            # Re-use a previously defined layer
            sparse_layer = self.sparse_layers[scope_name]

        return sparse_layer

    def applySparseLinear(self, x, sparse_layer, dense_length, compute_dense_grad, disable_outlining=False):
        # The underlying API requires a 2D tensor, so collapse the batch dimensions
        *batch_dimensions, input_length = x.shape.as_list()
        x = tf.reshape(x, [-1, input_length])

        # Call the layer with the provided input
        if disable_outlining or self.no_outline_sparse_layer:
            x = sparse_layer(x, compute_dense_grad)
        else:
            @outlined_function
            def f(x):
                return sparse_layer(x, compute_dense_grad)
            x = f(x)
        # Recover the original batch dimensions
        x = tf.reshape(x, batch_dimensions + [dense_length])
        return x

    def sparseLinear(self, x, sparsity, dense_length, compute_dense_grad, use_bias, disable_outlining=False, block_size=None):
        if block_size is None:
            block_size = self.block_size

        x_shape = x.shape.as_list()
        batch_dims = np.prod(x_shape[0: -1])
        x_shape = [batch_dims, x_shape[-1]]

        sparse_layer = self.getOrCreateSparseLinear(x_shape, x.dtype, sparsity, dense_length,
                                                    block_size=block_size, use_bias=use_bias)

        return self.applySparseLinear(x, sparse_layer,
                                      dense_length, compute_dense_grad,
                                      disable_outlining)

    def buildSparsityUpdateOps(self):
        # Places ops into the graph that allow the sparsity of the variables to be updated
        # this should be called after the optimizer is created
        ops = {}
        for name, sparse_layer in self.sparse_layers.items():
            with tf.variable_scope(name, reuse=True):
                ops[name] = sparse_layer.update_sparsity_op()
                if self.prune_ratio == 0 or self.prune_ratio is None:
                    # If a layer's sparsity pattern changed then its slot
                    # also need to be updated:
                    ops[name + '_slots'] = sparse_layer.update_slots_op()
        self.sparsityUpdateOps = tf.group(ops.values())
        return self.sparsityUpdateOps

    def streamSparsityFromHostToDevice(self):
        # Use the previously defined sparsityUpdateOps to push new values
        # for the sparse layers onto device. This should be called inside a session
        feed_dict = {}
        for name, sparse_layer in self.sparse_layers.items():
            feed_dict.update(sparse_layer.feed_dict())

        session = tf.get_default_session()
        if session is None:
            raise RuntimeError("TF default session not found")
        t0 = time.perf_counter()
        session.run(self.sparsityUpdateOps, feed_dict)
        t1 = time.perf_counter()
        logger.info(f"Done streaming sparsity from host to device in {t1-t0:0.3f} seconds.")
        return True

    def updateSparsityInfoOnHost(self, session_outputs):
        # Typically this can only be done during training by using the
        # outputs of the training session
        for name, sparse_layer in self.sparse_layers.items():
            values_var = sparse_training.get_values_var()
            slots = {
                slot_name: session_outputs[slot.tf_variable.name]
                for slot_name, slot in sparse_layer.get_slot_var_dict().items()
            }
            sparse_layer.sync_internal_representation({"nz": session_outputs[values_var.name]}, slots)
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
                logger.debug(f"Checkpointing: {values_var.name}  as {name + '/weight'}")
                dense_values[name + "/weight"] = sparse_layer.extract_dense().astype(dtype)
                trainable_vars.discard(values_var)
            if sparse_layer.use_bias:
                logger.debug(f"Checkpointing: {sparse_layer.bias.name}  as {name + '/bias'}")
                dense_values[name + "/bias"] = sess.run(sparse_layer.bias.value())
                trainable_vars.discard(sparse_layer.bias)

        # Now handle the remaining (non-sparse) trainable variables
        for var in trainable_vars:
            logger.debug("Checkpointing:")
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

    def syncPruneAndRegrowOnHost(self, cosine_options, step, total_steps, session_outputs):
        # Pruning schedule
        def cosine_prune_schedule(t, T, max_pruned):
            return int(np.ceil(max_pruned * sparse_training.cosine_prune_function(t, T, cosine_options)))

        if step == total_steps:
            logger.debug("Final step: pruning will be skipped.")
            return None, self.prune_ratio * sparse_training.cosine_prune_function(step, total_steps, cosine_options)

        if sparse_training.cosine_prune_function(step, total_steps, cosine_options) == 0:
            sched = self.prune_ratio * sparse_training.cosine_prune_function(step, total_steps, cosine_options)
            logger.debug(f"Nothing to prune at step {step}/{total_steps}: schedule is {sched}")
            return None, sched

        t0 = time.perf_counter()
        # Prune and grow each sparse layer
        for layer_name, sparse_layer in self.sparse_layers.items():
            values_var_name = sparse_layer.get_values_var().name
            slots = {
                slot_name: session_outputs[slot.tf_variable.name]
                for slot_name, slot in sparse_layer.get_slot_var_dict().items()}
            nz = session_outputs[values_var_name]

            sparse_layer.sync_internal_representation({"nz": nz}, slots)

            # run prune and grow
            grow_results = sparse_training.prune_and_grow(
                    name=layer_name + "/" + sparse_layer.name,
                    triplets=sparse_layer.get_triplets(),
                    shape=sparse_layer.get_shape(),
                    spec=sparse_layer.weights.spec,
                    max_non_zeros=sparse_layer.get_max_non_zeros(),
                    slot_triplets=sparse_layer.extract_slot_triplets(),
                    prune_schedule=partial(cosine_prune_schedule, t=step, T=total_steps),
                    prune_ratio=self.prune_ratio,
                    grad_w=np.array(session_outputs[layer_name + "/sparse_layer/grad_w"]),
                    grow_method=self.regrow_type,
                    random_gen=self.random,
                    ipu_pooling_type=self.pooling_type)

            if grow_results is not None:
                sparse_layer.update_triplets(grow_results['gt'])
                sparse_layer.update_slots_from_triplets(grow_results['gs'])

        t1 = time.perf_counter()
        prune_and_grow_time = t1 - t0
        logger.info(f"Prune and grow for step {step} completed in "
                    f"{prune_and_grow_time:0.3f} seconds for {len(self.sparse_layers.keys())} layers")
        # return the time it took to performn the prune and grow as well as the current
        # factor of the cosine schedule for monitoring
        return prune_and_grow_time, self.prune_ratio * sparse_training.cosine_prune_function(step, total_steps, cosine_options)
