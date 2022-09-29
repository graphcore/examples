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
    NamedTuple,
    Union
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
    :param triplets: Triplets to initialise the sparse martix.
    """

    def __init__(self, spec: sparse.MatmulSpec, matmul_options: Mapping[str, str], triplets: sparse.Triplets, name: str = ""):
        self.spec = spec
        self.name = name
        self.matmul_options = matmul_options.copy()
        if 'partialsType' in matmul_options:
            if matmul_options['partialsType'] == "half" and spec.block_size == 1:
                raise ValueError("Half partials are not supported for 1x1 blocks.")

        self.dense_grad_matmul_options = self.matmul_options.pop('dense_grad', {})

        # Initialised by update_from_triplets/update_from_values
        self.representation: sparse.SparseRepresentation = None
        self.triplets: sparse.Triplets = None
        self.update_from_triplets(triplets=triplets)

    def update_from_triplets(self, triplets: sparse.Triplets):
        self.triplets = triplets
        self.representation = sparse.representation_from_triplets(
            self.spec, *self.triplets, self.matmul_options, debug_name=self.name)

    def update_from_values(self, values: List[float], metainfo: List[float] = None):
        np.copyto(self.representation.nz_values, values)
        if metainfo is not None:
            # Reinterpret cast the metainfo as uint16 rather than float16.
            metainfo_as_uint16 = np.frombuffer(metainfo.tobytes(),
                                               dtype=np.uint16)
            np.copyto(self.representation.metainfo_state, metainfo_as_uint16)
        self.triplets = sparse.triplets_from_representation(
            self.spec, self.representation, self.matmul_options, debug_name=self.name)

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
            use_bias: bool = False,
            relu: bool = False,
            disable_updating: bool = False,
            pooling_type: str = "NONE"):
        """
        Construct a new 'SparseFcLayer' object. It is recommended to create these layers
        using the factory functions e.g.: 'from_random_generator' or 'from_triplets'.

        This layer is for fully connected layers that are sparse and can have the sparsity pattern
        updated during training.
        :param weights: A SparseMatrix object describing the sparse weight matrix for this layer.
        :param name: Name string for the layer (used as a variable scope).
        :use_bias: Flag to say whether a bias should be added to the layer.
        :relu: Flag to say whether a relu activation be added to the layer.
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

        self.use_bias = use_bias
        self.relu = relu
        self.bias_init = tf.zeros_initializer()
        self.disable_updating = disable_updating
        self.pooling_type = pooling_type

        # Initialised by build
        self.built = False
        self.dense_dummy_var: tf.Variable = None
        self.metainfo_var: tf.Variable = None
        self.values_var: tf.Variable = None

    @classmethod
    def from_random_generator(
            cls, hidden_size: int, input_shape: List[int],
            density: float,
            block_size: int,
            values_initialiser_gen: Callable[..., np.ndarray],
            indices_initialiser_gen: Callable,
            matmul_options: Mapping[str, str],
            name: str,
            dtype: tf.DType = tf.float32,
            use_bias: bool = False,
            relu: bool = False,
            disable_updating: bool = False,
            pooling_type: str = 'NONE'):
        """
        Utility factory function to build a 'SparseFcLayer' from a random sparsity
        pattern and random non zero values.
        """
        spec = sparse.matmul_spec_from_density(hidden_size, input_shape, density,
                                               block_size, dtype, pooling_type)
        ns = tf.get_default_graph().get_name_scope()
        qualified_name = ns + "/" + name if ns else name
        logger.debug(f"Creating random sparse FC {qualified_name} with spec: {spec}")
        t0 = time.perf_counter()
        triplets = sparse.random_triplets(spec, indices_initialiser_gen, values_initialiser_gen)
        t1 = time.perf_counter()
        logger.debug(f"Generated triplets in {t1-t0:0.03f} seconds")
        weights = SparseMatrix(spec, matmul_options, triplets, name=qualified_name)
        logger.debug(f"Triplet stats for {qualified_name}: {sparse.triplet_stats(*triplets)}")
        return cls(weights, name, use_bias, relu, disable_updating, pooling_type=pooling_type)

    @classmethod
    def from_triplets(cls, hidden_size: int, input_shape: List[int],
                      row_indices: List[int], col_indices: List[int],
                      values: List[float], matmul_options: Mapping[str, str], name: str,
                      dtype: tf.DType = tf.float32,
                      use_bias: bool = False,
                      relu: bool = False,
                      disable_updating: bool = False,
                      pooling_type: str = 'NONE'):
        """
        Utility factory function to build a 'SparseFcLayer' from a set of triplets (COO format).
        E.g. as returned from 'ipu_sparse_ops.sparse.triplets_from_dense'
        """
        block_size = sparse.block_size_from_list(values)
        spec = sparse.matmul_spec_from_max(hidden_size, input_shape, len(values), block_size, dtype, pooling_type)
        ns = tf.get_default_graph().get_name_scope()
        qualified_name = ns + "/" + name if ns else name
        logger.debug(f"Creating random sparse FC {qualified_name} with spec: {spec}")
        triplets = sparse.Triplets(row_indices, col_indices, values)
        weights = SparseMatrix(spec, matmul_options, triplets, name=qualified_name)
        return cls(weights, name, use_bias, relu, disable_updating, pooling_type=pooling_type)

    def get_nonzero_blocks_shape(self) -> List[int]:
        return (self.weights.spec.block_size, self.weights.spec.block_size)

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
        if self.values_var is None:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to return the nonzero sparse "
                "values variable but has not yet been built. Call this layer or explicitly build it.")
        return self.values_var

    def get_metainfo_var(self) -> tf.Variable:
        """
        Return the TensorFlow variable that is holding the sparse metainfo values for this layer.
        """
        if self.metainfo_var is None:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to return the sparse "
                "metainfo variable but has not yet been built. Call this layer or explicitly build it.")
        return self.metainfo_var

    def get_dense_dummy_var(self) -> tf.Variable:
        """
        Return the TensorFlow dummy variable that is used to reference the dense gradient for this layer.
        """
        if self.dense_dummy_var is None:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to return the dense dummy "
                "variable but has not yet been built. Call this layer or explicitly build it.")
        return self.dense_dummy_var

    def get_dense_grad_w(self, loss: tf.Tensor) -> tf.Tensor:
        """
        Access the TensorFlow variable that is holding the dense gradient for this layer.
        The dense gradient is conditionally computed so may be stale.
        """
        dummy_var = self.get_dense_dummy_var()

        logger.debug(f"Layer '{self.name}' grad dummy var: '{dummy_var}'")
        dense_grad = tf.gradients(loss, dummy_var)[0]

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
        return self.weights.spec.max_non_zero_blocks

    def record_slot_var(self, slot_name: str, optimizer: tf.train.Optimizer):
        """
        Used by the optimiser to record a slot with this layer.
        Returns the TensorFlow slot_variable that was recorded.
        """
        if self.values_var is None:
            raise AttributeError(
                f"This sparse layer '{self.name}' is being asked to record a "
                "slot variable but it has not yet been called! "
                "Make sure you call this layer upstream of the loss op or "
                "remove it from the sparse_layers list.")

        slot_var = optimizer.get_slot(self.values_var, slot_name)

        if slot_var is None:
            raise ValueError(
                f"This sparse layer '{self.name}' is being asked to record "
                f"a slot variable for '{self.values_var.name}' but no such "
                "slot exists! Make sure the loss op is actually dependent "
                "on this layer or remove it from the sparse_layers list.")

        internal_name = slot_var.name
        logger.debug(f"Recording slot variable {slot_var.name} as {internal_name}")

        if slot_var.shape != self.weights.get_values().shape:
            raise ValueError(
                f"Shape mismatch between variable {slot_var.shape} "
                f"and slot {self.weights.get_values().shape}")

        with tf.init_scope():  # escapes XLA, so placeholders can be created
            with tf.device("cpu"):
                placeholder = tf.placeholder(dtype=slot_var.dtype, shape=slot_var.shape)

        self.sparse_slots[internal_name] = SparseSlot(
            placeholder=placeholder,
            tf_variable=slot_var,
            np_variable=np.zeros_like(self.weights.get_values())
        )
        return slot_var

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

    def get_slot_var_dict(self):
        """
        Return the dict holding the slots.
        """
        return self.sparse_slots

    def extract_slot_triplets(self) -> Mapping[str, sparse.Triplets]:
        slot_representations = {
            name: sparse.SparseRepresentation(self.weights.get_metainfo(), slot.np_variable)
            for name, slot in self.get_slot_var_dict().items()
        }
        return {
            name: sparse.triplets_from_representation(
                self.weights.spec, representation, self.weights.matmul_options, debug_name=name + "(slot)")
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
                self.weights.matmul_options, debug_name=name + "(slot)")
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

    def sync_internal_representation(self, values: List[Mapping[str, List[float]]], slots: Mapping[str, List[float]],
                                     metainfo: List[Mapping[str, List[int]]] = None):
        """
        Used to store the values and slots returned from the device into the internal
        SparseRepresentation object (self.weights.representation). This will typically be called after each
        training step that you run on the device.
        """
        values = [value for value in values.values()]
        if len(values) > 1:
            raise Exception("sync_internal_representation expects a single array of non-zero values")
        values = values[0]

        if metainfo is not None:
            metainfo = [metainf for metainf in metainfo.values()]
            if len(metainfo) > 1:
                raise Exception("sync_internal_representation expects a single set of metainfo or None")
            metainfo = metainfo[0]

        self.weights.update_from_values(values=values, metainfo=metainfo)
        if not self.disable_updating:
            for name, values in slots.items():
                np.copyto(self.sparse_slots[name].np_variable, values)

    def build(self):
        """Generates the underlying variables once."""
        if self.built:
            return

        self.values_var, self.metainfo_var, self.dense_dummy_var = \
            sparse.get_or_create_matmul_vars(
                self.weights.spec,
                self.weights.representation,
                self.weights.matmul_options,
                constant_metainfo=self.disable_updating)

        if self.use_bias:
            self.bias = tf.get_variable(
                "bias", shape=[self.weights.spec.output_size],
                initializer=self.bias_init,
                dtype=tf.dtypes.as_dtype(self.weights.get_values().dtype)
            )

        self.built = True
        return

    def __call__(
            self,
            inputs: tf.Tensor,
            compute_dense_grad_w: Union[bool, tf.Tensor] = False) -> tf.Tensor:
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
                self.dense_dummy_var,
                self.weights.dense_grad_matmul_options)

            logger.debug(f"Layer '{self.name}' non-zeros var: '{self.values_var.name}'")

            if self.use_bias:
                z = z + self.bias
            # Reshape z to remove group size of 1 (no other group sizes are supported
            # at the moment):
            z = tf.reshape(z, [self.weights.spec.batch_size, self.weights.spec.output_size])
            if self.relu:
                return tf.nn.relu(z)
            else:
                return z


class DenseFcLayer:
    """
    This is a dense FC layer with the same call, placeholder and feed interface as SparseFcLayer.
    """

    def __init__(self, hidden_size, name, dtype=tf.float32, use_bias=False, relu=False):
        """
        :param hidden_size: Output size for the hidden layer.
        :param name: Name string for the layer. This is not optional as it
                     sets the variable namespace used to access internal variables.
        :use_bias: Flag to say whether a bias should be added to the layer.
        :relu: Flag to say whether a relu activation be added to the layer.
        """
        self.hidden_size = hidden_size
        self.name = name
        self.weight_init = tf.glorot_uniform_initializer()
        self.bias_init = tf.zeros_initializer()
        self.relu = relu
        self.use_bias = use_bias
        self.dtype = dtype

    def __call__(self, input, ignored=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, use_resource=True):
            self.w = tf.get_variable("weight", shape=[input.shape[-1], self.hidden_size],
                                     dtype=self.dtype, initializer=self.weight_init)
            if self.use_bias:
                self.bias = tf.get_variable("bias", shape=[self.hidden_size],
                                            dtype=self.dtype, initializer=self.bias_init)
            if self.use_bias and self.relu:
                return tf.nn.relu_layer(input, self.w, self.bias)
            else:
                z = tf.matmul(input, self.w)
                if self.use_bias:
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
