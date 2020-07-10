# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from ipu_sparse_ops import sparse
import numpy as np
import tensorflow.compat.v1 as tf
from logging import getLogger
import time
import os

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = getLogger(os.path.basename(__file__))


class SparseFcLayer:
    def __init__(self, spec: sparse.MatmulSpec, triplets: list, bias=False, relu=False, generator=None):
        self.spec = spec
        t0 = time.perf_counter()
        self.data = sparse.representation_from_triplets(self.spec, *triplets)
        self.triplets = triplets
        t1 = time.perf_counter()
        logger.info(f"Random triplets created in {t1-t0:0.3f} seconds")
        # Each layer needs momentum data that shares the same sparse representation
        # as the non-zero value data (initialise momentum to zero):
        self.sparse_momentum = np.zeros_like(self.data.nz_values)
        logger.info(f"Created sparse values and momentum with shapes {self.data.nz_values.shape} {self.sparse_momentum.shape}")
        self.bias = bias
        self.relu = relu
        self.bias_init = tf.zeros_initializer()
        self.generator = generator

    @classmethod
    def from_random_generator(cls, hidden_size, input_shape, density, topk_ratio,
                              initialiser_gen, seed=None,
                              bias=False, relu=False):
        spec = matmul_spec_from_density(hidden_size, input_shape, density, topk_ratio)
        t0 = time.perf_counter()
        triplets = sparse.random_triplets(spec, seed, initialiser_gen)
        t1 = time.perf_counter()
        logger.info(f"Random triplets created in {(t1-t0):.3f} seconds")
        return cls(spec, triplets, bias, relu, initialiser_gen)

    @classmethod
    def from_triplets(cls, hidden_size, input_shape, topk_ratio,
                      row_indices, col_indices, values,
                      bias=False, relu=False):
        spec = matmul_spec_from_max(hidden_size, input_shape, len(values), len(values)*topk_ratio)
        triplets = [row_indices, col_indices, values]
        return cls(spec, triplets, bias, relu)

    def create_placeholders(self, data_type):
        self.metainfo_ph, self.nz_ph = self.data.makePlaceHolders(data_type)
        self.momentum_ph = tf.placeholder(data_type, self.data.valuesShape())

    def feed_dict(self):
        return {self.metainfo_ph: self.data.metaInfoFeed(),
                self.nz_ph: self.data.valuesFeed(),
                self.momentum_ph: self.sparse_momentum}

    def is_sparse(self):
        return True

    def update_sparsity_op(self):
        return sparse.update_metainfo_op(self.metainfo_ph, self.nz_ph)

    def update_momentum_op(self):
        if self.momentum_var is None:
            raise ValueError(f"Sparse FC Layer {name} has no momentum. (NOTE: You must build the"
                             "training op before building the update op).")
        return self.momentum_var.assign(self.momentum_ph)

    def get_values_var(self):
        return sparse.get_or_create_nz_values(
            self.data.nz_values.dtype, self.data.nz_values.shape)

    def get_dense_grad_w(self, loss):
        dummy_var = sparse.get_or_create_dense_grad_w(self.spec)
        return tf.gradients(loss, dummy_var)

    def get_max_prune_count(self):
        return self.spec.topk

    def record_momentum_var(self, optimiser, momentum_slot_name):
        self.momentum_var = optimiser.get_slot(self.get_values_var(), momentum_slot_name)

    def update_triplets(self, new_triplets):
        self.triplets = new_triplets
        self.data = sparse.representation_from_triplets(self.spec, *self.triplets)

    def extract_dense(self):
        return sparse.dense_from_triplets(self.spec, *self.triplets)

    def extract_mask(self):
        return sparse.mask_from_triplets(self.spec, *self.triplets)

    def extract_triplets(self):
        return sparse.triplets_from_representation(self.spec, self.data)

    def extract_momentum_triplets(self):
        momentum_data = sparse.SparseRepresentation(self.data.metainfo_state, self.sparse_momentum)
        return sparse.triplets_from_representation(self.spec, momentum_data)

    def update_momentum_from_triplets(self, new_momentum_triplets):
        momentum_data = sparse.representation_from_triplets(self.spec, *new_momentum_triplets)
        if self.sparse_momentum.shape != momentum_data.nz_values.shape:
            raise RuntimeError("New momentum shape is not compatible. "
                               f"New: {momentum_data.nz_values.shape} != old: {self.sparse_momentum.shape}")
        self.sparse_momentum = momentum_data.nz_values

    def sync_internal_data(self, values, momentum):
        np.copyto(self.data.nz_values, values)
        np.copyto(self.sparse_momentum, momentum)

    def __call__(self, input, compute_dense_grad_w=False):
        z = sparse.matmul(self.spec, input, compute_dense_grad_w)
        if self.bias:
            self.b = tf.get_variable("bias", shape=[self.spec.output_size],
                                     initializer=self.bias_init)
            z = z + self.b
        # Reshape z to remove group size of 1:
        z = tf.reshape(z, [self.spec.batch_size, self.spec.output_size])
        if self.relu:
            return tf.nn.relu(z)
        else:
            return z


def matmul_spec_from_density(hidden_size: int, input_shape, density: float, topk_ratio: float):
    max_non_zeros = int(np.ceil(density * input_shape[1] * hidden_size))
    return sparse.MatmulSpec(
        input_size=input_shape[1], output_size=hidden_size,
        num_groups=1, batch_size=input_shape[0],
        data_type=tf.float32,
        max_non_zeros=max_non_zeros,
        topk=int(np.ceil(topk_ratio * max_non_zeros)))


def matmul_spec_from_max(hidden_size: int, input_shape: list, max_non_zeros: int, topk: int):
    return sparse.MatmulSpec(
        input_size=input_shape[1], output_size=hidden_size,
        num_groups=1, batch_size=input_shape[0],
        data_type=tf.float32,
        max_non_zeros=max_non_zeros,
        topk=int(topk))


# This is not a sparse layer but it is useful to have
# a dense layer with the same interface:
class DenseFcReluLayer:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.weight_init = tf.glorot_uniform_initializer()
        self.bias_init = tf.zeros_initializer()

    def __call__(self, input, ignored=None):
        self.w = tf.get_variable("weight", shape=[input.shape[-1], self.hidden_size],
                                 initializer=self.weight_init)
        self.b = tf.get_variable("bias", shape=[self.hidden_size], initializer=self.bias_init)
        return tf.nn.relu_layer(input, self.w, self.b)

    def feed_dict(self):
        return {}

    def create_placeholders(self, data_type):
        None

    def is_sparse(self):
        return False
