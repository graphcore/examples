# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes dynamic sparse layers through simple classes with a consistent API.
There are also dense layers wrapped in the same interface in order to make it easier
to build models where the layers can be switched from sparse to dense easily.

The sparse layers store a host side representation of the sparsity pattern
and utilities for manipulating and syncing this representation to and from TensorFlow
variables (on the device). It is the users responsibilty to manage the syncronisation of the
host and TensorFlow (device side) representation. This is achieved by calling the appropriate
methods in this sparse layer classes and arranging for the TensorFlow ops constructed by the layer
to be called at the right time in the TF compute graph. Careful orchestration of these can be used
to implement dynamic sparse optimisers such as RigL:
https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl
"""

import time
import os

import numpy as np
import tensorflow.compat.v1 as tf

from ipu_sparse_ops import sparse
from logging import getLogger
from typing import (
    Callable,
    List,
    Mapping,
    NamedTuple
)


logger = getLogger(os.path.basename(__file__))


class SparseMatrix:
    """
    Utility class to manage a sparse weight matrix in triplet (COO) format.
    :param spec: The parameters that specify the multiplication of inputs with this weight matrix.
    :param triplets: The sparse weight matrix specified in triplet/COO format.
                     E.g. as returned from 'ipu_sparse_ops.sparse.triplets_from_dense'.
    :param matmul_options: Poplib's matmul options specific for these weights. See popsparse
                           docs in Poplibs.
    """

    def __init__(self, spec: sparse.MatmulSpec, matmul_options: Mapping[str, str], triplets: sparse.Triplets):
        self.spec = spec
        self.matmul_options = matmul_options

        # Initialised by update_from_triplets/update_from_values
        self.representation: sparse.SparseRepresentation = None
        self.triplets: sparse.Triplets = None

        self.update_from_triplets(triplets=triplets)

    def update_from_triplets(self, triplets: sparse.Triplets):
        self.triplets = triplets
        self.representation = sparse.representation_from_triplets(
            self.spec, *self.triplets, self.matmul_options)

    def update_from_values(self, values: List[float]):
        np.copyto(self.representation.nz_values, values)
        self.triplets = sparse.triplets_from_representation(
            self.spec, self.representation, self.matmul_options)

    def extract_dense(self) -> np.ndarray:
        return sparse.dense_from_triplets(self.spec, *self.triplets)

    def extract_mask(self) -> np.ndarray:
        return sparse.mask_from_triplets(self.spec, *self.triplets)

    def get_metainfo(self) -> List[np.uint16]:
        return self.representation.metainfo_state

    def get_triplets(self) -> sparse.Triplets:
        return self.triplets

    def get_values(self) -> np.ndarray:
        return self.representation.nz_values

    def get_matmul_shape(self) -> List[int]:
        return (self.spec.input_size, self.spec.output_size)

    def get_data_type(self):
        return self.spec.data_type


class SparseSlot(NamedTuple):
    np_variable: np.ndarray
    tf_variable: tf.Variable
    placeholder: tf.Tensor = None


class SparseFcLayer:
    def __init__(
            self,
            weights: SparseMatrix,
            name: str,
            bias: bool = False,
            relu: bool = False,
            generator: Callable[..., np.ndarray] = None,
            disable_updating: bool = False):
        """
        Construct a new 'SparseFcLayer' object. It is recommended to create these layers
        using the factory functions: 'from_random_generator' or 'from_triplets'.

        This layer is for fully connected layers that are sparse and can have the sparsity pattern
        updated during training.
        :param weights: A SparseMatrix object describing the sparse weight matrix for this layer.
        :param name: Name string for the layer. This is not optional as it sets the variable namespace
                     used to access internal sparse variables.
        :bias: Flag to say whether a bias should be added to the layer.
        :relu: Flag to say whether a relu activation be added to the layer.
        :generator: A optional callable random number generator that can be used to
                    initialise the layer with random triplets.
        :disable_updating: A flag to disable updates of the sparsity pattern for this
                           layer. Non-zero values can still change.
        """
        self.weights = weights
        self.name = name

        # Each layer needs slot values that share the same sparse representation
        # as the non-zero value data (initialise slots to zero):
        self.sparse_slots: Mapping[str, SparseSlot] = {}  # Initialised by repeated calls to record_slot_var

        logger.debug(f"Created sparse values with shape {self.weights.get_values().shape}")
        logger.debug(f"Created sparse metainfo with shape {self.weights.get_metainfo().shape}")

        self.bias = bias
        self.relu = relu
        self.bias_init = tf.zeros_initializer()
        self.generator = generator
        self.disable_updating = disable_updating

        # Initialised by build
        self.built = False
        self.dense_dummy_var: tf.Variable = None
        self.metainfo_var: tf.Variable = None
        self.values_var: tf.Variable = None

    @classmethod
    def from_random_generator(
            cls, hidden_size: int, input_shape: List[int], density: float,
            values_initialiser_gen: Callable[..., np.ndarray],
            indices_initialiser_gen: Callable,
            matmul_options: Mapping[str, str],
            name: str,
            dtype: tf.DType = tf.float32,
            bias: bool = False, relu: bool = False, disable_updating: bool = False):
        """
        Utility factory function to build a 'SparseFcLayer' from a random sparsity
        pattern and random non zero values.
        """
        spec = sparse.matmul_spec_from_density(hidden_size, input_shape, density, dtype)
        t0 = time.perf_counter()
        triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
        t1 = time.perf_counter()
        weights = SparseMatrix(spec, matmul_options, triplets)
        logger.debug(f"Random triplets created in {(t1-t0):.3f} seconds")
        return cls(weights, name, bias, relu, values_initialiser_gen, disable_updating)

    @classmethod
    def from_triplets(cls, hidden_size: int, input_shape: List[int],
                      row_indices: List[int], col_indices: List[int],
                      values: List[float], matmul_options: Mapping[str, str], name: str,
                      dtype: tf.DType = tf.float32,
                      bias: bool = False, relu: bool = False, disable_updating: bool = False):
        """
        Utility factory function to build a 'SparseFcLayer' from a set of triplets (COO format).
        E.g. as returned from 'ipu_sparse_ops.sparse.triplets_from_dense'
        """
        spec = sparse.matmul_spec_from_max(hidden_size, input_shape, len(values), dtype)
        triplets = sparse.Triplets(row_indices, col_indices, values)
        weights = SparseMatrix(spec, matmul_options, triplets)
        return cls(weights, name, bias, relu, None, disable_updating)

    @classmethod
    def from_random_orthonormal_generator(cls, hidden_size: int, input_shape: List[int], density: float,
                                          matmul_options: Mapping[str, str], name: str, dtype: tf.DType = tf.float32,
                                          bias: bool = False, relu: bool = False, inference: bool = False):
        if (not input_shape[1] == hidden_size):
            raise ValueError(f"Dimensions must be square to use this generator: "
                             f"input_shape[1]={input_shape[1]}, hidden_size={hidden_size}")

        sparseMatrix, max_non_zeros = sparse.gen_sparse_rand_orthog_mat(hidden_size, density)

        # Get the matmul spec from the max non zeros
        spec = sparse.matmul_spec_from_max(hidden_size, input_shape, max_non_zeros, dtype)

        triplets = sparse.triplets_from_dense(sparseMatrix)

        # create Sparse matrix with triplets
        weights = SparseMatrix(spec, matmul_options, triplets)
        return cls(weights, name, bias, relu, None, inference)

    def get_shape(self) -> List[int]:
        return self.weights.get_matmul_shape()

    def get_data_type(self):
        return self.weights.get_data_type()

    def create_placeholders(self):
        self.metainfo_ph, self.nz_ph = self.weights.representation.makePlaceHolders(self.weights.spec.data_type)

    def feed_dict(self) -> Mapping[tf.Tensor, np.ndarray]:
        """
        Return a feed that can be used to initialise and update the sparsity
        pattern, values, and sparse slots.
        """
        if self.disable_updating:
            return {}

        feeds = {
            self.metainfo_ph: self.weights.representation.metaInfoFeed(),
            self.nz_ph: self.weights.representation.valuesFeed()
        }
        feeds.update({
            slot.placeholder: slot.np_variable
            for slot in self.sparse_slots.values()
        })

        return feeds

    def is_sparse(self) -> bool:
        """
        Used to distinguish sparse and dense layer objects within this module.
        """
        return True

    def update_sparsity_op(self) -> tf.Operation:
        """
        Return a TensorFlow op that can be used to update the sparsity pattern and values from a feed.
        This op must be fed from the result of calling this layer's feed_dict() method.
        If self.disable_updating is set then this will return a no-op.
        """
        if self.disable_updating:
            return tf.no_op()
        elif not self.built:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to create an "
                "update op to change the underlying sparsity pattern but has "
                "not yet been built. Either call the layer before using this "
                "method or build it explicitly by calling build().")
        else:
            return sparse.update_metainfo_op_with_vars(
                self.metainfo_ph, self.nz_ph,
                self.metainfo_var, self.values_var)

    def update_slots_op(self) -> tf.Operation:
        """
        Return a TensorFlow op that can be used to update the sparse slot (values only not pattern) from a feed.
        This op must be fed from the result of calling this layer's feed_dict() method. If you have not built the
        training op then this variable will not exist and an exception will be raised.
        """
        if not self.sparse_slots:
            logger.warning("update_slots_op called with no slots registered.")
            return tf.no_op()

        update_ops = [
            slot.tf_variable.assign(slot.placeholder)
            for slot in self.sparse_slots.values()
        ]
        return tf.group(update_ops)

    def get_values_var(self) -> tf.Variable:
        """
        Return the TensorFlow variable that is holding the non zero values for this layer.
        """
        return self.values_var

    def get_dense_grad_w(self, loss: tf.Tensor) -> List[tf.Tensor]:
        """
        Access the TensorFlow variable that is holding the dense gradient for this layer.
        The dense gradient is conditionally computed so may be stale.
        """
        if not self.built:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to return a dense gradient "
                "but has not yet been called.")

        logger.debug(f"Layer '{self.name}' grad dummy var: '{self.dense_dummy_var.name}'")
        dense_grad = tf.gradients(loss, self.dense_dummy_var)[0]

        if dense_grad is None:
            raise ValueError(
                f"This sparse layer '{self.name}' is being asked to return a dense gradient "
                "but the loss op does not depend on it. Make sure the loss op is dependent "
                "on the output of this layer.")

        return dense_grad

    def get_max_non_zeros(self) -> int:
        """
        The maximum number of non-zeros allowed in this layer.
        """
        return self.weights.spec.max_non_zeros

    def record_slot_var(self, slot_name: str, optimizer: tf.train.Optimizer):
        """
        Used by the optimiser to record a slot with this layer.
        """
        values_var = self.get_values_var()

        if values_var is None:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to record a "
                "slot variable but it has not yet been called! "
                "Make sure you call this layer upstream of the loss op or "
                "remove it from the sparse_layers list.")

        slot_var = optimizer.get_slot(values_var, slot_name)

        if slot_var is None:
            raise ValueError(
                f"This sparse layer '{self.name}' is being asked to record "
                f"a slot variable for '{self.values_var.name}' but no such "
                "slot exists! Make sure the loss op is actually dependent "
                "on this layer or remove it from the sparse_layers list.")

        if slot_var.shape != self.weights.get_values().shape:
            raise ValueError(
                f"Shape mismatch between variable {slot_var.shape} "
                f"and slot {self.weights.get_values().shape}")

        with tf.init_scope():  # escapes XLA, so placeholders can be created
            with tf.device("cpu"):
                placeholder = tf.placeholder(dtype=slot_var.dtype, shape=slot_var.shape)

        self.sparse_slots[slot_name] = SparseSlot(
            placeholder=placeholder,
            tf_variable=slot_var,
            np_variable=np.zeros_like(self.weights.get_values())
        )

    def update_triplets(self, new_triplets: sparse.Triplets):
        """
        Update the host side representation of the sparsity pattern with a new set of triplets.
        The on device representation will not be updated until you run the op returned from the
        layer's 'update_sparsity_op()' method.
        """
        self.weights.update_from_triplets(new_triplets)

    def extract_dense(self) -> np.ndarray:
        """
        Return a dense version of this layer's sparse weight matrix.
        """
        return self.weights.extract_dense()

    def extract_mask(self) -> np.ndarray:
        """
        Return a dense mask representation of this layer's weight matrix,
        """
        return self.weights.extract_mask()

    def get_triplets(self) -> sparse.Triplets:
        """
        Return a triplet version of this layer's sparse weight matrix.
        """
        return self.weights.get_triplets()

    def extract_slot_triplets(self) -> Mapping[str, sparse.Triplets]:
        slot_representations = {
            name: sparse.SparseRepresentation(self.weights.get_metainfo(), slot.np_variable)
            for name, slot in self.sparse_slots.items()
        }

        return {
            name: sparse.triplets_from_representation(
                self.weights.spec, representation, self.weights.matmul_options)
            for name, representation in slot_representations.items()
        }

    def update_slots_from_triplets(self, slot_triplets: Mapping[str, sparse.Triplets]):
        """
        Update the host side representation of the sparse slot with a new set of triplets.
        The row and column indices must be identical to those for the sparse weights.
        The on device representation will not be updated until you run the op returned from the
        layer's 'update_sparsity_op()' method.
        """
        slot_representations = {
            name: sparse.representation_from_triplets(
                self.weights.spec,
                *triplet,
                self.weights.matmul_options)
            for name, triplet in slot_triplets.items()
        }

        for name, representation in slot_representations.items():
            current_slot = self.sparse_slots[name]

            if current_slot.np_variable.shape != representation.nz_values.shape:
                raise RuntimeError(
                    "New slot shape is not compatible. "
                    f"Slot {name}: New: {representation.nz_values.shape} != old: {current_slot.shape}")

            self.sparse_slots[name] = SparseSlot(
                np_variable=representation.nz_values,
                tf_variable=current_slot.tf_variable,
                placeholder=current_slot.placeholder)

    def sync_internal_representation(self, values: List[float], slots: Mapping[str, List[float]]):
        """
        Used to store the values and slots returned from the device into the internal
        SparseRepresentation object (self.weights.representation). This will typically be called after each
        training step that you run on the device.
        """
        self.weights.update_from_values(values=values)
        if not self.disable_updating:
            for name, values in slots.items():
                np.copyto(self.sparse_slots[name].np_variable, values)

    def update_sparsity_pattern(self, updater, **kwargs):
        """
        Performs the pruning and growing of the weights to update the sparsity parttern, for the current fc layer
        :param updater: An update function for the sparsity pattern. This function must take as input the following
        key word arguments: name, triplets, shape, spec, max_non_zeros, slot_triplets
        """
        grow_results = updater(name=self.name,
                               triplets=self.weights.get_triplets(),
                               shape=self.get_shape(),
                               spec=self.weights.spec,
                               max_non_zeros=self.get_max_non_zeros(),
                               slot_triplets=self.extract_slot_triplets())

        # update the internal representation using the grown triplets and slots
        try:
            self.update_triplets(grow_results['gt'])
            self.update_slots_from_triplets(grow_results['gs'])
        except:
            print(f"Failed to update representation with triplets:\n{grow_results['gt'][0]}\n{grow_results['gt'][1]}\n{grow_results['gt'][2]}")
            print(f"Non-zeros: {len(grow_results['gt'][0])}")
            print(f"Layer spec: {self.weights.spec}")
            raise

    def build(self):
        """Generates the underlying variables once."""
        if self.built:
            return self

        self.values_var, self.metainfo_var, self.dense_dummy_var = \
            sparse.get_or_create_matmul_vars(
                self.weights.spec,
                self.weights.representation,
                self.weights.matmul_options,
                constant_metainfo=self.disable_updating)

        if self.bias:
            self.b = tf.get_variable(
                "bias", shape=[self.weights.spec.output_size],
                initializer=self.bias_init,
                dtype=self.weights.get_values().dtype)

        self.built = True
        return self

    def __call__(self, inputs: tf.Tensor, compute_dense_grad_w=tf.constant(False)) -> tf.Tensor:
        """
        Build and return the op to execute the layer. It will
        compute the matrix multiplication of input with the
        soarse weight matrix, then add bias and activation ops
        if these are enabled for this layer.
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, use_resource=True):
            self.build()

            z = sparse.matmul_with_vars(
                self.weights.spec,
                inputs,
                compute_dense_grad_w,
                self.weights.matmul_options,
                self.values_var,
                self.metainfo_var,
                self.dense_dummy_var)

            logger.debug(f"Layer '{self.name}' non-zeros var: '{self.values_var.name}'")

            if self.bias:
                z = z + self.b
            # Reshape z to remove group size of 1 (no other group sizes are supported
            # at the moment):
            z = tf.reshape(z, [self.weights.spec.batch_size, self.weights.spec.output_size])
            if self.relu:
                return tf.nn.relu(z)
            else:
                return z


class SparseGRULayer:
    def __init__(self, sequence_length: int, batch_size: int, input_size: int,
                 hidden_size: int, gru_layers: dict, variant: str, disable_updating: bool = False,
                 reset_after_fc: bool = False, name: str = 'gru', dtype: tf.DType = tf.float32):
        """
        Construct a new 'SparseGRULayer' object. It is recommended to create these layers
        using the factory functions: 'from_random_generator' or 'from_dense'.

        This layer is for GRU layers that are sparse and can have the sparsity pattern
        updated during training.
        :param sequence_length: The sequence length, should be the first dim of the input.
        :param batch_size: The batch size, should be the second dim of the input.
        :param input_size: The size of each input, should be the third and last dimension of the input.
        :param hidden_size: The hidden size for the GRU and its state.
        :param gru_layers: A dictionary that maps from name of the weight to its corresponding
                           instance of the SparseFcLayer class.
        :param variant: A string indicating the GRU cell variant. Possible choices are:
                        sequential, standard, parallel
        :disable_updating: A flag to disable updates of the sparsity pattern for this
                           layer. Non-zero values can still change.
        :reset_after_fc: A flag to have reset after FC. This technique exchanges the order of
                    r and U to calculate r*(Uxh_prev) instead of Ux(r*h_prev).
                    It can be used to have fully parallel implementation of GRU.
                    More details can be found here https://svail.github.io/diff_graphs/
        """
        # The possible GRU cell variants
        self.variants = ['sequential', 'standard', 'parallel']
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = gru_layers
        if variant.lower() not in self.variants:
            raise RuntimeError(f"Unrecognised GRU cell variant. Possible choices are: {self.variants}.")
        elif variant == 'parallel':
            raise NotImplementedError("Parallel variant of GRU cell not yet implemented.")
        self.variant = variant.lower()
        self.reset_after_fc = reset_after_fc
        self.disable_updating = disable_updating
        self.state = tf.zeros([batch_size, hidden_size], dtype)
        self.name = name

    @classmethod
    def from_dense(
            cls,
            hidden_size: int,
            input_shape: List[int],
            dense_weights: np.ndarray,
            matmul_options: Mapping[str, str] = {"metaInfoBucketOversizeProportion": 0.1},
            variant: str = 'sequential',
            dtype: tf.DType = tf.float32,
            disable_updating: bool = False,
            reset_after_fc: bool = False,
            name: str = 'gru'):
        """
        Utility factory function to build a 'SparseGRULayer' from a list of masked dense weights.
        """
        gru_layers = {}
        sequence_length, batch_size, input_size = input_shape
        if len(dense_weights) != 6:
            raise RuntimeError(
                f"6 weights are needed to build the GRU cell, {len(dense_weights)} provided.")

        if variant == 'sequential':
            subscript = {0: 'r', 1: 'z', 2: 'h'}
            # In a GRU cell there are:
            # 3 W matrices of size [input_size, hidden_size] and
            # 3 U matrices of size [hidden_size, hidden_size]
            for i, weight in enumerate(dense_weights):
                layer_name = 'W' if i < 3 else 'U'
                layer_name += subscript[i % 3]
                triplets = sparse.triplets_from_dense(weight)
                gru_layers[layer_name] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, input_size if i < 3 else hidden_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=(i < 3), relu=False, disable_updating=disable_updating, name=name + "_" + layer_name)

        if variant == 'standard':
            wr, wz, wh, ur, uz, uh = dense_weights

            wuz = np.concatenate([wz, uz], axis=0)
            wur = np.concatenate([wr, ur], axis=0)
            gate_weight = np.concatenate([wuz, wur], axis=1)

            triplets = sparse.triplets_from_dense(gate_weight)
            gru_layers['gate'] = SparseFcLayer.from_triplets(
                2 * hidden_size, [batch_size,
                                  input_size+hidden_size], *triplets,
                matmul_options=matmul_options,
                dtype=dtype,
                bias=True, relu=False, disable_updating=disable_updating, name=name + '_gate')

            if reset_after_fc:
                triplets = sparse.triplets_from_dense(wh)
                gru_layers['wh'] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, input_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=True, relu=False, disable_updating=disable_updating, name=name + '_wh')

                triplets = sparse.triplets_from_dense(uh)
                gru_layers['uh'] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, hidden_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=False, relu=False, disable_updating=disable_updating, name=name + '_uh')
            else:
                candidate_weight = np.concatenate([wh, uh], axis=0)
                triplets = sparse.triplets_from_dense(candidate_weight)
                gru_layers['candidate'] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, input_size + hidden_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=True, relu=False, disable_updating=disable_updating, name=name + '_candidate')

        if variant == 'parallel':
            raise NotImplementedError("Parallel variant of GRU cell not yet implemented.")

        return cls(sequence_length, batch_size, input_size, hidden_size, gru_layers,
                   variant, disable_updating, reset_after_fc, name, dtype)

    @classmethod
    def from_random_generator(
            cls, hidden_size: int,
            input_shape: List[int],
            density: float,
            values_initialiser_gen: Callable[..., np.ndarray],
            indices_initialiser_gen: Callable,
            matmul_options: Mapping[str, str] = {"metaInfoBucketOversizeProportion": 0.1},
            dtype: tf.DType = tf.float32,
            variant: str = 'sequential',
            disable_updating: bool = False,
            reset_after_fc: bool = False,
            name: str = 'gru'):
        """
        Utility factory function to build a 'SparseGRULayer' from a random sparsity
        pattern and random non zero values.
        """
        gru_layers = {}
        subscript = {0: 'r', 1: 'z', 2: 'h'}
        # In a GRU cell there are:
        # 3 W matrices of size [input_size, hidden_size] and
        # 3 U matrices of size [hidden_size, hidden_size]
        sequence_length, batch_size, input_size = input_shape
        if variant == 'sequential':
            for i in range(6):
                layer_name = 'W' if i < 3 else 'U'
                layer_name += subscript[i % 3]
                input_shape = [batch_size, input_size if i < 3 else hidden_size]
                spec = sparse.matmul_spec_from_density(hidden_size, input_shape, density, dtype)
                t0 = time.perf_counter()
                triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
                t1 = time.perf_counter()
                logger.info(
                    f"Random triplets created in {(t1-t0):.3f} seconds")
                gru_layers[layer_name] = SparseFcLayer.from_triplets(
                    hidden_size, input_shape, *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=(i < 3), relu=False, disable_updating=disable_updating, name=name + "_" + layer_name)

        if variant == 'standard':
            input_shape_new = [batch_size, input_size + hidden_size]
            spec = sparse.matmul_spec_from_density(2*hidden_size, input_shape_new, density, dtype)
            t0 = time.perf_counter()
            triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
            t1 = time.perf_counter()
            logger.info(f"Random triplets created in {(t1-t0):.3f} seconds")

            gru_layers['gate'] = SparseFcLayer.from_triplets(
                2 * hidden_size, input_shape_new, *triplets,
                matmul_options=matmul_options,
                dtype=dtype,
                bias=True, relu=False, disable_updating=disable_updating, name=name + '_gate')

            if reset_after_fc:
                spec = sparse.matmul_spec_from_density(hidden_size, [batch_size, input_size], density, dtype)
                triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
                gru_layers['wh'] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, input_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=True, relu=False, disable_updating=disable_updating, name=name + '_wh')
                spec = sparse.matmul_spec_from_density(hidden_size, [batch_size, hidden_size], density, dtype)
                triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
                gru_layers['uh'] = SparseFcLayer.from_triplets(
                    hidden_size, [batch_size, hidden_size], *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=False, relu=False, disable_updating=disable_updating, name=name + '_uh')
            else:
                spec = sparse.matmul_spec_from_density(hidden_size, input_shape_new, density, dtype)
                t0 = time.perf_counter()
                triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
                t1 = time.perf_counter()
                logger.info(f"Random triplets created in {(t1-t0):.3f} seconds")
                gru_layers['candidate'] = SparseFcLayer.from_triplets(
                    hidden_size, input_shape_new, *triplets,
                    matmul_options=matmul_options,
                    dtype=dtype,
                    bias=True, relu=False, disable_updating=disable_updating, name=name + '_candidate')

        if variant == 'parallel':
            raise NotImplementedError("Parallel variant of GRU cell not yet implemented.")

        return cls(sequence_length, batch_size, input_size, hidden_size, gru_layers,
                   variant, disable_updating, reset_after_fc, name, dtype)

    def create_placeholders(self):
        for name, layer in self.layers.items():
            layer.create_placeholders()

    def feed_dict(self) -> Mapping[tf.Tensor, np.ndarray]:
        """
        Return a feed that can be used to initialise and update the sparsity
        pattern, values, and sparse slots.
        """
        if self.disable_updating:
            return {}
        else:
            fd = {}
            for name, layer in self.layers.items():
                fd.update(layer.feed_dict())
            return fd

    def is_sparse(self) -> bool:
        """
        Used to distinguish sparse and dense layer objects within this module.
        """
        return True

    def update_sparsity_op(self) -> tf.Operation:
        """
        Return a TensorFlow op that can be used to update the sparsity pattern and values from a feed.
        This op must be fed from the result of calling this layer's feed_dict() method.
        Note: Unless disable_updating is set this must be called at least once before the layer can be used at all.
        If self.disable_updating is set then this will return a no-op.
        """
        if self.disable_updating:
            return tf.no_op()
        else:
            update_ops = []
            for name, layer in self.layers.items():
                update_ops.append(layer.update_sparsity_op())
            return tf.group(update_ops)

    def update_slots_op(self) -> tf.Operation:
        """
        Return a TensorFlow op that can be used to update the sparse slot (values only not pattern) from a feed.
        This op must be fed from the result of calling this layer's feed_dict() method. If you have not built the
        training op then these variables will not exist and an exception will be raised.
        """
        all_update_ops = [layer.update_slots_op() for layer in self.layers.values()]
        return tf.group(all_update_ops)

    def get_data_type(self):
        if not self.layers:
            raise RuntimeError("GRU cell has no layers.")
        return list(self.layers.values())[0].get_data_type()

    def get_sparse_grad_w(self, loss):
        """
        Calculate the sparse gradients w.r.t. weights
        :param loss: A tensor representing the loss.
        """
        # Convert the sparse gradient metainfo back to triplets and then use those row and col indices
        # to index the dense reference weight gradient:
        grad_w = {}
        for name, layer in self.layers.items():
            vars = layer.get_values_var()
            grad_w[name] = tf.gradients(loss, vars)[0]
        return grad_w

    def reconstruct_dense_gradients_w(self, sparse_weights_grad):
        for name, layer in self.layers.items():
            # convert the sparse gradient metainfo back to triplets and then to dense
            sparse_data = sparse.SparseRepresentation(layer.weights.get_metainfo(), sparse_weights_grad[name])
            triplets = sparse.triplets_from_representation(layer.weights.spec, sparse_data, layer.weights.matmul_options)
            grads = sparse.dense_from_triplets(layer.weights.spec, *triplets)

            # split the dense representation of sparse gradients
            if name == 'gate':
                wuz, wur = np.split(grads, 2, axis=1)
                wz, uz = np.split(wuz, [self.input_size], axis=0)
                wr, ur = np.split(wur, [self.input_size], axis=0)
            if name == 'wh':
                wh = grads
            if name == 'uh':
                uh = grads
            if name == 'candidate':
                wh, uh = np.split(grads, [self.input_size], axis=0)

        dense_grads_from_sparse = [wr, wz, wh, ur, uz, uh]
        return dense_grads_from_sparse

    def get_dense_grad_w(self, loss: tf.Tensor) -> List[List[tf.Tensor]]:
        """
        Access the TensorFlow variable that is holding the dense gradient
        with respect to all the weights.
        The dense gradients are conditionally computed so may be stale.
        :param loss: A tensor representing the loss.
        """
        if self.variant == 'sequential':
            return [layer.get_dense_grad_w(loss) for layer in self.layers.values()]

        if self.variant == 'standard':
            for name, layer in self.layers.items():
                dense_grad_w = layer.get_dense_grad_w(loss)
                if name == 'gate':
                    wuz, wur = tf.split(dense_grad_w, 2, axis=1)
                    wz, uz = tf.split(wuz, [self.input_size, self.hidden_size], axis=0)
                    wr, ur = tf.split(wur, [self.input_size, self.hidden_size], axis=0)
                if name == 'wh':
                    wh = dense_grad_w
                if name == 'uh':
                    uh = dense_grad_w
                if name == 'candidate':
                    wh, uh = tf.split(dense_grad_w, [self.input_size, self.hidden_size], axis=0)
            dense_grads_from_sparse = [wr, wz, wh, ur, uz, uh]
            return dense_grads_from_sparse

    def get_max_non_zeros(self, weight: str) -> int:
        """
        The maximum number of non-zeros allowed for the given weight.
        :param weight: A string indicating the name of the weight.
        """
        return self.layers[weight].weights.spec.max_non_zeros

    def record_slot_var(self, slot_name: str, optimizer: tf.train.Optimizer):
        """
        Used by the optimiser to record a slot with this layer.
        """
        for layer in self.layers.values():
            layer.record_slot_var(slot_name, optimizer)

    def update_triplets(self, new_triplets: List[sparse.Triplets]):
        """
        Update the host side representation of the sparsity pattern with a new set of triplets.
        The on device representation will not be updated until you run the op returned from the
        layer's 'update_sparsity_op()' method.
        :param new_triplets: A list of new triplets. There must be a new triplet for each sparse Fc layer
                             inside this GRU (i.e. 6)
        """
        if len(new_triplets) != 6:
            raise RuntimeError("There must be a set of 6 new triplets to update the existing ones within the GRU.")

        wr, wz, wh, ur, uz, uh = new_triplets
        if self.variant == 'standard':
            wur = sparse.concatenate_triplets(
                wr, ur, [self.input_size, self.hidden_size], axis=0)
            wuz = sparse.concatenate_triplets(
                wz, uz, [self.input_size, self.hidden_size], axis=0)
            gate_weight = sparse.concatenate_triplets(
                wuz, wur, [self.input_size + self.hidden_size, self.hidden_size], axis=1)
            self.layers['gate'].weights.update_from_triplets(gate_weight)

            if self.reset_after_fc:
                self.layers['wh'].weights.update_from_triplets(new_triplets[2])
                self.layers['uh'].weights.update_from_triplets(new_triplets[5])
            else:
                candidate_weight = sparse.concatenate_triplets(
                    wh, uh, [self.input_size, self.hidden_size], axis=0)
                self.layers["candidate"].weights.update_from_triplets(candidate_weight)

        if self.variant == 'sequential':
            for new_triplet, layer in zip(new_triplets, self.layers.values()):
                layer.weights.update_from_triplets(new_triplet)

    def extract_dense(self) -> List[np.ndarray]:
        """
        Return a dense version of all 6 sparse weight matrices.
        """
        dense_weights = {}
        if self.variant == 'sequential':
            for name, layer in self.layers.items():
                dense_weights[name] = layer.extract_dense()

        if self.variant == 'standard':
            gate = self.layers['gate'].extract_dense()
            wuz, wur = np.split(gate, 2, axis=1)
            wz, uz = np.split(wuz, [self.input_size], axis=0)
            wr, ur = np.split(wur, [self.input_size], axis=0)
            if self.reset_after_fc:
                wh = self.layers['wh'].extract_dense()
                uh = self.layers['uh'].extract_dense()
            else:
                candidate = self.layers['candidate'].extract_dense()
                wh, uh = np.split(candidate, [self.input_size], axis=0)

            dense_weights = {'wr': wr, 'wz': wz, 'wh': wh, 'ur': ur, 'uz': uz, 'uh': uh}
        return dense_weights

    def extract_mask(self, weight: str) -> np.ndarray:
        """
        Return a dense mask representation of the given sparse weight matrix.
        :param weight: A string indicating the name of the weight.
        """
        return self.layers[weight].extract_mask()

    def get_triplets(self) -> List[Mapping[str, sparse.Triplets]]:
        """
        Return a triplet version of all the sparse weight matrix.
        :param weight: A string indicating the name of the weight.
        """
        if self.variant == 'standard':
            all_triplets = {}
            wuz, wur = sparse.split_triplets(self.layers['gate'].get_triplets(), self.hidden_size, axis=1)
            all_triplets['Wz'], all_triplets['Uz'] = sparse.split_triplets(wuz, self.input_size, axis=0)
            all_triplets['Wr'], all_triplets['Ur'] = sparse.split_triplets(wur, self.input_size, axis=0)
            if self.reset_after_fc:
                all_triplets['Wh'] = self.layers['wh'].get_triplets()
                all_triplets['Uh'] = self.layers['uh'].get_triplets()
            else:
                all_triplets['Wh'], all_triplets['Uh'] = sparse.split_triplets(self.layers['candidate'].get_triplets(), self.input_size, axis=0)

        if self.variant == 'sequential':
            all_triplets = {name: layer.get_triplets() for name, layer in self.layers.items()}

        return all_triplets

    def extract_slot_triplets(self) -> List[Mapping[str, sparse.Triplets]]:
        """
        Return a triplet version of each weight's sparse slot matrix.
        """
        if self.variant == 'standard':
            all_slot_triplets = [{}, {}, {}, {}, {}, {}]
            gate_slot_triplets = self.layers['gate'].extract_slot_triplets()
            for name, triplets in gate_slot_triplets.items():
                wuz, wur = sparse.split_triplets(triplets, self.hidden_size, axis=1)
                wz, uz = sparse.split_triplets(wuz, self.input_size, axis=0)
                wr, ur = sparse.split_triplets(wur, self.input_size, axis=0)
                all_slot_triplets[0][name] = wr  # correspond to Wr
                all_slot_triplets[3][name] = ur  # correspond to Ur
                all_slot_triplets[1][name] = wz  # correspond to Wr
                all_slot_triplets[4][name] = uz  # correspond to Ur
            if self.reset_after_fc:
                all_slot_triplets[2] = self.layers['wh'].extract_slot_triplets()
                all_slot_triplets[5] = self.layers['uh'].extract_slot_triplets()
            else:
                candidate_slot_triplets = self.layers['candidate'].extract_slot_triplets()
                for name, triplets in candidate_slot_triplets.items():
                    wh, uh = sparse.split_triplets(triplets, self.input_size, axis=0)
                    all_slot_triplets[2][name] = wh
                    all_slot_triplets[5][name] = uh

        if self.variant == 'sequential':
            all_slot_triplets = [layer.extract_slot_triplets() for layer in self.layers.values()]

        return all_slot_triplets

    def update_slots_from_triplets(self, new_slots_triplets: List[Mapping[str, sparse.Triplets]]):
        """
        Update the host side representation of the sparse slots with a new set of triplets.
        The row and column indices must be identical to those for the sparse weights.
        The on device representation will not be updated until you run the op returned from the
        layer's 'update_sparsity_op()' method.
        :param new_slots_triplets: A list of new slot triplets. There must be a new slot
                                      triplet for each weight matrix inside this GRU (i.e. 6)
        """
        if len(new_slots_triplets) != 6:
            raise RuntimeError("There must be a set of 6 new slot triplets"
                               "to update the existing ones within the GRU.")
        if self.variant == 'standard':
            gate_slot_triplets = {}
            wr_dict, ur_dict, wz_dict, uz_dict = new_slots_triplets[0], new_slots_triplets[3], new_slots_triplets[1], new_slots_triplets[4]
            wh_dict, uh_dict = new_slots_triplets[2], new_slots_triplets[5]
            if not (wr_dict.keys() == ur_dict.keys() == wz_dict.keys() == uz_dict.keys() == wh_dict.keys() == uh_dict.keys()):
                raise RuntimeError("Names of the slots to update must be identical for each weight matrix")
            for name in wr_dict.keys():
                wur = sparse.concatenate_triplets(
                    wr_dict[name], ur_dict[name], [self.input_size, self.hidden_size], axis=0)
                wuz = sparse.concatenate_triplets(
                    wz_dict[name], uz_dict[name], [self.input_size, self.hidden_size], axis=0)
                gate_slot_triplets[name] = sparse.concatenate_triplets(
                    wuz, wur, [self.input_size + self.hidden_size, self.hidden_size], axis=1)
            self.layers['gate'].update_slots_from_triplets(gate_slot_triplets)
            if self.reset_after_fc:
                self.layers["wh"].update_slots_from_triplets(wh_dict)
                self.layers["uh"].update_slots_from_triplets(uh_dict)
            else:
                candidate_slot_triplets = {}
                for name in wh_dict.keys():
                    candidate_slot_triplets[name] = sparse.concatenate_triplets(
                        wh_dict[name], uh_dict[name], [self.input_size, self.hidden_size], axis=0)
                self.layers["candidate"].update_slots_from_triplets(candidate_slot_triplets)

        if self.variant == 'sequential':
            for new_slot_triplet, layer in zip(new_slots_triplets, self.layers.values()):
                layer.update_slots_from_triplets(new_slot_triplet)

    def sync_internal_representation(self, values: List[List[float]], slots: List[Mapping[str, List[float]]]):
        """
        Used to store the values and slots returned from the device into the internal
        SparseRepresentation object (self.weights.representation). This will typically be called after each
        training step that you run on the device.
        :param values: A list of nz values. There must be a new set of values for each
                       sparse Fc layer inside this GRU (i.e. 6)
        :param slots: A list of slot dicts. There must be a new set of slot values
                         for each sparse Fc layer inside this GRU (i.e. 6)
        """
        if len(values) != 6:
            raise RuntimeError("There must be a set of 6 new values "
                               "to update the existing ones within the GRU.")
        if not self.disable_updating and len(slots) != 6:
            raise RuntimeError("There must be a set of 6 new slot values "
                               "to update the existing ones within the GRU.")
        for new_values, new_slot, layer in zip(values, slots, self.layers.values()):
            layer.sync_internal_representation(new_values, new_slot)

    def __call__(self, input: tf.Tensor, compute_dense_grad_w):
        """
        Build and return the op to execute the layer. It will
        compute the result of applying the sparse GRU layer
        over the input.
        """
        # Check the input dimensions
        input_shape = input.get_shape().as_list()
        if len(input_shape) != 3:
            raise RuntimeError("The input to the GRU layer should be a rank 3 tensor,"
                               f" got rank {len(input_shape)} instead.")
        if input_shape[0] != self.sequence_length or \
           input_shape[1] != self.batch_size or \
           input_shape[2] != self.input_size:
            raise RuntimeError(
                "Unexpected input shape to the GRU layer. "
                f"Expected: {[self.sequence_length, self.batch_size, self.input_size]}"
                f"Got: {input_shape}")

        h0 = self.state

        def GRU_cell_sequential(ht, xt):
            logger.info("Sequential GRU..............")
            rt = tf.sigmoid(self.layers['Wr'](xt, compute_dense_grad_w) +
                            self.layers['Ur'](ht, compute_dense_grad_w))
            zt = tf.sigmoid(self.layers['Wz'](xt, compute_dense_grad_w) +
                            self.layers['Uz'](ht, compute_dense_grad_w))
            if self.reset_after_fc:
                # first matmul, then elementwise mul
                Urh = tf.multiply(rt, self.layers['Uh'](ht, compute_dense_grad_w))
            else:
                # first elementwise mul, then matmul
                Urh = self.layers['Uh'](tf.multiply(rt, ht), compute_dense_grad_w)
            h_candidate = tf.tanh(self.layers['Wh'](xt, compute_dense_grad_w) + Urh)

            h_out = tf.multiply(zt, ht) + tf.multiply(1 - zt, h_candidate)
            return h_out

        def GRU_cell_standard(h_prev, x):
            logger.info("Concatenated GRU..............")
            x_h_prev = tf.concat([x, h_prev], 1)
            gate_inputs = self.layers['gate'](x_h_prev, compute_dense_grad_w)
            value = tf.sigmoid(gate_inputs)
            z, r = tf.split(value=value, num_or_size_splits=2, axis=1)
            if self.reset_after_fc:
                candidate = self.layers['wh'](x, compute_dense_grad_w) + r * self.layers['uh'](h_prev, compute_dense_grad_w)
            else:
                r_state = r * h_prev
                candidate = self.layers['candidate'](tf.concat([x, r_state], 1), compute_dense_grad_w)

            c = tf.tanh(candidate)
            new_h = z * h_prev + (1 - z) * c
            return new_h

        # Repeat the cell sequence_length times (first dim of input)
        if self.variant == 'standard':
            out = tf.scan(GRU_cell_standard, input, initializer=h0)

        if self.variant == 'sequential':
            out = tf.scan(GRU_cell_sequential, input, initializer=h0)

        if self.variant == 'parallel':
            raise RuntimeError("The fully parallel GRU is not yet implemented.")
        return out


class DenseFcLayer:
    """
    This is a dense FC layer with the same call, placeholder and feed interface as SparseFcLayer.
    """

    def __init__(self, hidden_size, name, dtype=tf.float32, bias=False, relu=False):
        """
        :param hidden_size: Output size for the hidden layer.
        :param name: Name string for the layer. This is not optional as it
                     sets the variable namespace used to access internal variables.
        :bias: Flag to say whether a bias should be added to the layer.
        :relu: Flag to say whether a relu activation be added to the layer.
        """
        self.hidden_size = hidden_size
        self.name = name
        self.weight_init = tf.glorot_uniform_initializer()
        self.bias_init = tf.zeros_initializer()
        self.relu = relu
        self.bias = bias
        self.dtype = dtype

    def __call__(self, input, ignored=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, use_resource=True):
            self.w = tf.get_variable("weight", shape=[input.shape[-1], self.hidden_size],
                                     dtype=self.dtype, initializer=self.weight_init)
            self.b = tf.get_variable("bias", shape=[self.hidden_size],
                                     dtype=self.dtype, initializer=self.bias_init)
            if self.bias and self.relu:
                return tf.nn.relu_layer(input, self.w, self.b)
            else:
                z = tf.matmul(input, self.w)
                if self.bias:
                    z = z + self.bias
                if self.relu:
                    return tf.nn.relu(z)
                else:
                    return z

    def feed_dict(self):
        return {}

    def create_placeholders(self):
        None

    def is_sparse(self):
        return False

    def get_data_type(self):
        return self.dtype
