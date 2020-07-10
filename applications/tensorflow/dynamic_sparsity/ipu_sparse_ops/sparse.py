# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import numpy as np
import collections
from ipu_sparse_ops import host_utils
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from logging import getLogger
tf.disable_eager_execution()
tf.disable_v2_behavior()


def get_lib_path(lib_name):
    base_path = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(base_path, "lib" + lib_name + ".so")


MatmulSpec = collections.namedtuple('MatmulSpec', 'max_non_zeros num_groups batch_size input_size output_size data_type topk')
logger = getLogger(os.path.basename(__file__))


class SparseRepresentation:
    def __init__(self, metainfo, nz):
        self.metainfo_state = metainfo
        self.nz_values = nz
        self.metainfo_state_fp16 = self.metainfo_state.view(dtype=np.float16)

    def makePlaceHolders(self, data_type):
        metainfo_ph = tf.placeholder(tf.float16, self.metaInfoShape())
        nz_ph = tf.placeholder(data_type, self.valuesShape())
        return metainfo_ph, nz_ph

    def metaInfoShape(self):
        return [self.metainfo_state.size]

    def valuesShape(self):
        return [self.nz_values.size]

    def metaInfoFeed(self):
        # XLA requires us to pass only floating point tensors to custom ops:
        return self.metainfo_state_fp16

    def valuesFeed(self):
        return self.nz_values

    def __str__(self):
        return f"metainfo: {self.metainfo_state} values:{self.nz_values}"


def get_or_create_args(spec: MatmulSpec):
    with tf.variable_scope("dummy", reuse=tf.AUTO_REUSE, use_resource=True):
        # Compile time args have to be passed in the tensor shape:
        args = [spec.output_size, spec.max_non_zeros, spec.num_groups]
        arg_dummy = tf.get_variable(
            name="args_hidden_in_shape",
            dtype=tf.float32,
            shape=args,
            trainable=False,
            initializer=tf.zeros_initializer())
        return arg_dummy


def get_or_create_nz_values(data_type, shape=None):
    with tf.variable_scope("sparse_weights", reuse=tf.AUTO_REUSE, use_resource=True):
        return tf.get_variable("values", dtype=data_type, shape=shape)


def get_or_create_metainfo(data_type, shape=None):
    with tf.variable_scope("sparse_weights", reuse=tf.AUTO_REUSE, use_resource=True):
        return tf.get_variable("metainfo", dtype=data_type, shape=shape)


def get_or_create_dense_grad_w(spec: MatmulSpec):
    # We need a dummy input that allows us to retrive the dense gradient:
    with tf.variable_scope("sparse_weights", reuse=tf.AUTO_REUSE, use_resource=True):
        return tf.get_variable("dense_gradW", shape=[spec.input_size, spec.output_size],
                               dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())


def allocate_matmul_inputs(lhs, spec: MatmulSpec):
    metainfo_size, nz_max_size = get_sparse_tensor_sizes(spec)
    arg_dummy = get_or_create_args(spec)
    outputs = {
        "output_types": [tf.float16, tf.float32, tf.float32],
        "output_shapes": [metainfo_size, nz_max_size, lhs.shape],
    }
    lib_path = get_lib_path("fc_allocate")
    return ipu.custom_ops.precompiled_user_op([arg_dummy, lhs],
                                              lib_path,
                                              outs=outputs,
                                              inputs_with_gradients=[])


def matmul(spec: MatmulSpec, lhs, return_dense_grad):
    metainfo, nz_values, dense_lhs = allocate_matmul_inputs(lhs, spec)

    dense_lhs = tf.identity(lhs)
    result_shape = tf.TensorShape([spec.batch_size, spec.output_size])

    # Make cars for the representaiton so we can update it from the host
    # and the weights are trainable:
    trainable_nz = get_or_create_nz_values(nz_values.dtype, nz_values.shape)
    metainfo = get_or_create_metainfo(metainfo.dtype, metainfo.shape)
    dense_grad_w = get_or_create_dense_grad_w(spec)

    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [result_shape],
    }
    arg_dummy = get_or_create_args(spec)
    inputs = [dense_lhs, metainfo, trainable_nz, arg_dummy, return_dense_grad, dense_grad_w]
    lib_path = get_lib_path("sparse_matmul")
    with_grads = [0, 2, 5]  # No grads for metainfo and dummy arg
    result = ipu.custom_ops.precompiled_user_op(inputs,
                                                lib_path,
                                                outs=outputs,
                                                inputs_with_gradients=with_grads)
    return result


def update_metainfo_op(metainfo_ph, nz_ph):
    # Returns an op that can be used to update the sparsity pattern:
    nz = get_or_create_nz_values(nz_ph.dtype)
    meta = get_or_create_metainfo(metainfo_ph.dtype)
    assign_nz = nz.assign(nz_ph)
    assign_meta = meta.assign(metainfo_ph)
    with tf.control_dependencies([assign_nz, assign_meta]):
        update_op = tf.no_op()
    return update_op


def representation_from_triplets(spec: MatmulSpec, row_indices, col_indices, values):
    # TODO: why is it necessary to sort by rows - popsparse claims it is not?
    sort_idx = np.argsort(row_indices)
    metainfo, nzvalues = host_utils.representation_from_triplets(
        spec.max_non_zeros, spec.num_groups, spec.batch_size, spec.input_size, spec.output_size,
        row_indices[sort_idx], col_indices[sort_idx], values[sort_idx])
    return SparseRepresentation(metainfo, nzvalues)


def triplets_from_representation(spec: MatmulSpec, sparse_data: SparseRepresentation):
    row_indices, col_indices, values = host_utils.triplets_from_representation(
        spec.max_non_zeros, spec.num_groups, spec.batch_size, spec.input_size, spec.output_size,
        sparse_data.metainfo_state, sparse_data.nz_values)
    return row_indices, col_indices, values


def get_sparse_tensor_sizes(spec: MatmulSpec):
    return host_utils.get_sparse_tensor_sizes(spec.max_non_zeros, spec.num_groups, spec.batch_size, spec.input_size, spec.output_size)


def triplets_from_dense(matrix: np.array):
    indices = np.nonzero(matrix)
    values = matrix[indices]
    return indices[0], indices[1], values


def dense_from_triplets(spec: MatmulSpec, row_indices, col_indices, values):
    # Input is multiplied on the left in popsparse:
    dense = np.zeros(shape=[spec.input_size, spec.output_size])
    dense[(row_indices, col_indices)] = values
    return dense


def mask_from_triplets(spec: MatmulSpec, row_indices, col_indices, values):
    # Input is multiplied on the left in popsparse:
    mask = np.zeros(shape=[spec.input_size, spec.output_size])
    mask[(row_indices, col_indices)] = 1
    return mask


def values_at_indices(row_indices, col_indices, matrix: np.array):
    grad_indices = (row_indices, col_indices)
    return matrix[grad_indices]


def random_triplets(spec: MatmulSpec, seed: int, value_generator, excluded_flat_indices=None, count=None):
    rng = np.random.default_rng(seed=seed)

    # Input is multiplied on the left in popsparse:
    rows = spec.input_size
    cols = spec.output_size
    number = count if count is not None else spec.max_non_zeros

    # Create a random sample of non-repeating flat indices
    # and then convert them to row, col:
    total_indices = rows * cols
    if total_indices < number:
        raise ValueError(f"Not enough indices (Attempting to draw {number} from set of {total_indices})")

    if excluded_flat_indices is None:
        flat_indices = rng.choice(total_indices, size=number, replace=False)
    else:
        # NOTE: Forming the total index list is a poor algorithm for very
        # large matrices:
        choose_from = np.delete(np.arange(total_indices), excluded_flat_indices)
        flat_indices = rng.choice(choose_from, size=number, replace=False)

    row_indices, col_indices = np.unravel_index(flat_indices, (rows, cols))
    values = value_generator(size=len(flat_indices))

    return row_indices, col_indices, values
