# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import json
import os

import math
import random
import numpy as np
import tensorflow.compat.v1 as tf

from ipu_sparse_ops import host_utils
from logging import getLogger
from tensorflow.python import ipu
from typing import (
    Callable,
    List,
    NamedTuple,
    Tuple,
    Union
)

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = getLogger(os.path.basename(__file__))


def get_lib_path(lib_name):
    base_path = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(base_path, "lib" + lib_name + ".so")


class MatmulSpec(NamedTuple):
    max_non_zeros: int
    num_groups: int
    batch_size: int
    input_size: int
    output_size: int
    data_type: tf.DType


class Triplets(NamedTuple):
    row_indices: List[int]
    col_indices: List[int]
    values: List[float]


def concatenate_triplets(a: Triplets, b: Triplets, dim_a: List[int], axis: int=0) -> Triplets:
    """
    Concatenate 2 sparse matrices from their triplets representation.
    Returns the triplets representation of the concatenated matrices.
    :param a: The first matrix to concatenate, as a triplets object
    :param b: The second matrix to concatenate, as a triplets object
    :param dim_a: The dimension of the first matrix. Must be an array of length 2
    :param axis: The axis along which to concatenate (0 means contatenate along rows)
    """
    if len(dim_a) != 2 or axis > 1 or axis < 0:
        raise Exception("Sparse triplets can only represent 2D matrices")

    b_rows = b.row_indices if axis == 1 else [index + dim_a[0] for index in b.row_indices]
    b_cols = b.col_indices if axis == 0 else [index + dim_a[1] for index in b.col_indices]


    rows = a.row_indices + b_rows
    cols = a.col_indices + b_cols
    values = a.values + b.values
    return Triplets(rows, cols, values)


def split_triplets(triplets: Triplets, split_index: List[int], axis=0) -> List[Triplets]:
    """
    Split a sparse matrix from its triplets representation into two
    (Also represented as triplets).
    :param triplets: The matrix to split, as a triplets object
    :param split_index: The matrix index where to split, i.e. the size of the left hand matrix along the split axis.
    Multiple indices can be passed in a list for multiple splits.
    :param axis: Axis along which to split. (0 means along the rows)
    e.g. a if split_index = 2, axis=0, [shape(4,2)] -> [shape(2,2)] , [shape(2,2)]
    """
    if (isinstance(split_index, int)):
        split_index = [split_index]
    elif (isinstance(split_index, list)):
        split_index.sort()
    else:
        raise Exception("split_index must be an integer or a list of integers")

    if axis == 0:
        index_list = np.array(triplets.row_indices)
    elif axis == 1:
        index_list = np.array(triplets.col_indices)
    else:
        raise Exception("Cannot split along required dimension (non-existant).")

    split = split_index[0]
    a_index = []
    b_index = []
    for k in range(index_list.size):
        if index_list[k] < split: a_index.append(k)
        else: b_index.append(k)

    a_rows = [triplets.row_indices[i] for i in a_index]
    a_cols = [triplets.col_indices[i] for i in a_index]
    a_vals = [triplets.values[i] for i in a_index]

    b_rows = [triplets.row_indices[i] for i in b_index]
    b_cols = [triplets.col_indices[i] for i in b_index]
    b_vals = [triplets.values[i] for i in b_index]

    # Remove the indices offset
    if axis == 0:
        b_rows = [i - split for i in b_rows]
    elif axis == 1:
        b_cols = [i - split for i in b_cols]
    split_index = [i - split for i in split_index]

    if len(split_index) == 1:
        return [Triplets(a_rows, a_cols, a_vals), Triplets(b_rows, b_cols, b_vals)]
    else:
        remaining_triplets = split_triplets(Triplets(b_rows, b_cols, b_vals), split_index[1:], axis)
        return [Triplets(a_rows, a_cols, a_vals)] + remaining_triplets


def matmul_spec_from_density(hidden_size: int, input_shape: List[int], density: float, dtype: tf.DType) -> MatmulSpec:
    """
    Utility to build a sparse matrix multiply specification object by specifying the density
    proportion of the layer (where density = 1 - sparsity).
    """
    max_non_zeros = int(np.ceil(density * input_shape[1] * hidden_size))
    return MatmulSpec(
        input_size=input_shape[1], output_size=hidden_size,
        num_groups=1, batch_size=input_shape[0],
        data_type=dtype,
        max_non_zeros=max_non_zeros)


def matmul_spec_from_max(hidden_size: int, input_shape: list, max_non_zeros: int, dtype: tf.DType) -> MatmulSpec:
    """
    Utility to build a sparse matrix multiply specification object by specifying the maxumim
    number of non zero entries.
    """
    return MatmulSpec(
        input_size=input_shape[1], output_size=hidden_size,
        num_groups=1, batch_size=input_shape[0],
        data_type=dtype,
        max_non_zeros=max_non_zeros)


class SparseRepresentation:
    """
    This class stores variables that hold the representation of
    a sparse weight matrix in the IPU's native format (metainfo
    and non-zero values).
    """
    def __init__(self, metainfo: List[np.uint16], nz_values: List[np.float]):
        self.metainfo_state = metainfo
        self.nz_values = nz_values
        self.metainfo_state_fp16 = self.metainfo_state.view(dtype=np.float16)

    def makePlaceHolders(self, data_type) -> Tuple[tf.Tensor, tf.Tensor]:
        metainfo_ph = tf.placeholder(tf.float16, self.metaInfoShape())
        nz_ph = tf.placeholder(data_type, self.valuesShape())
        logger.debug(f"Non zero values placeholder type: {nz_ph.dtype}")
        return metainfo_ph, nz_ph

    def metaInfoShape(self) -> List[int]:
        return [self.metainfo_state.size]

    def valuesShape(self) -> List[int]:
        return [self.nz_values.size]

    def metaInfoFeed(self) -> np.ndarray:
        # XLA requires us to pass only floating point tensors to custom ops:
        return self.metainfo_state_fp16

    def valuesFeed(self) -> np.ndarray:
        return self.nz_values

    def __str__(self):
        return f"metainfo: {self.metainfo_state} values:{self.nz_values}"


def nestable_json_from_dict(matmul_options: dict) -> str:
    json_options = json.dumps(matmul_options)
    return json_options.replace('"', '\\\"')


def get_json_args(spec: MatmulSpec, matmul_options: dict) -> str:
    # matmul_json is intended to be nested JSON that will
    # be parsed by Poplar hence has to be escaped:
    matmul_json = nestable_json_from_dict(matmul_options)
    return "{" \
           f"\"batch_size\":{spec.batch_size}, " \
           f"\"input_size\":{spec.input_size}, " \
           f"\"output_size\":{spec.output_size}, " \
           f"\"max_non_zeros\":{spec.max_non_zeros}, " \
           f"\"group_size\":{spec.num_groups}, " \
           f"\"data_type\":\"{str(spec.data_type)}\", " \
           f"\"matmul_options\":\"{matmul_json}\"" \
           "}"


def get_or_create_sparse_variable(name, data_type, shape, values, constant, trainable) -> Union[tf.Tensor, tf.Variable]:
    if constant:
        if values is None:
            raise ValueError("Must pass values if variable is constant.")
        return tf.constant(values, dtype=data_type, shape=shape)
    return tf.get_variable(name, dtype=data_type, initializer=values, trainable=trainable)


def get_or_create_nz_values(data_type, shape, values=None, constant=False) -> Union[tf.Tensor, tf.Variable]:
    return get_or_create_sparse_variable("nz_values", data_type, shape, values, constant, True)


def get_or_create_metainfo(data_type, shape, values=None, constant=False) -> Union[tf.Tensor, tf.Variable]:
    return get_or_create_sparse_variable("metainfo", data_type, shape, values, constant, False)


def get_or_create_dense_grad_w(spec: MatmulSpec) -> tf.Variable:
    # We need a dummy input that allows us to retrive the dense gradient:
    return tf.get_variable("dense_gradW", shape=[spec.input_size, spec.output_size],
                           dtype=spec.data_type, trainable=False, initializer=tf.zeros_initializer())


def get_or_create_matmul_vars(
        spec: MatmulSpec,
        sparse_data: SparseRepresentation,
        matmul_options: dict,
        constant_metainfo: bool):
    metainfo_size, nz_size, splits = get_sparse_tensor_sizes(spec, matmul_options)

    logger.info(f"Serialisation splits for dense grad W: {splits}")
    if any([s > 1 for s in splits]):
        logger.warn("Serialization of the dense grad W matmul is required to "
                    "respect memory budget but this is not implemented yet.")

    # Make vars for the representation so we can update it
    # from the host and the weights are trainable:
    trainable_nz = get_or_create_nz_values(spec.data_type, [nz_size],
                                           sparse_data.valuesFeed(), False)
    metainfo = get_or_create_metainfo(tf.float16, [metainfo_size],
                                      sparse_data.metaInfoFeed(), constant_metainfo)
    dense_grad_w = get_or_create_dense_grad_w(spec)

    return trainable_nz, metainfo, dense_grad_w


def matmul_with_vars(
        spec: MatmulSpec,
        lhs: tf.Tensor,
        return_dense_grad: bool,
        matmul_options: dict,
        trainable_nz: tf.Variable,
        metainfo: tf.Variable,
        dense_grad_w: tf.Variable) -> tf.Tensor:
    result_shape = tf.TensorShape([spec.batch_size, spec.output_size])

    outputs = {
        "output_types": [spec.data_type],
        "output_shapes": [result_shape],
    }

    json_args = get_json_args(spec, matmul_options)
    inputs = [lhs, metainfo, trainable_nz, return_dense_grad, dense_grad_w]
    lib_path = get_lib_path("sparse_matmul")
    with_grads = [0, 2, 4]  # No grads wanted for metainfo and scalar bool
    result = ipu.custom_ops.precompiled_user_op(
        inputs,
        lib_path,
        outs=outputs,
        inputs_with_gradients=with_grads,
        attributes=json_args,
        gradient_attributes=json_args)

    return result


def matmul(
        spec: MatmulSpec,
        lhs: tf.Tensor,
        return_dense_grad: bool,
        matmul_options: dict,
        sparse_data: SparseRepresentation,
        constant_metainfo: bool) -> tf.Tensor:
    (
        trainable_nz,
        metainfo,
        dense_grad_w
    ) = get_or_create_matmul_vars(
        spec, sparse_data, matmul_options, constant_metainfo)

    return matmul_with_vars(
        spec, lhs, return_dense_grad, matmul_options,
        trainable_nz, metainfo, dense_grad_w)


def update_metainfo_op(metainfo_ph: tf.Tensor, nz_ph: tf.Tensor) -> tf.Operation:
    # Returns an op that can be used to update the sparsity pattern:
    nz = get_or_create_nz_values(nz_ph.dtype, nz_ph.shape)
    meta = get_or_create_metainfo(metainfo_ph.dtype, metainfo_ph.shape)
    return update_metainfo_op_with_vars(metainfo_ph, nz_ph, meta, nz)


def update_metainfo_op_with_vars(
        metainfo_ph: tf.Tensor, nz_ph: tf.Tensor,
        metainfo_var: tf.Variable, nz_var: tf.Variable) -> tf.Operation:
    assign_nz = nz_var.assign(nz_ph)
    assign_meta = metainfo_var.assign(metainfo_ph)

    with tf.control_dependencies([assign_nz, assign_meta]):
        update_op = tf.no_op()

    return update_op


def representation_from_triplets(
        spec: MatmulSpec,
        row_indices: List[int],
        col_indices: List[int],
        values: List[float],
        matmul_options: dict,
        n_ipus: int = 1) -> SparseRepresentation:
    metainfo, nzvalues = host_utils.representation_from_triplets(
        n_ipus,
        spec.max_non_zeros, spec.num_groups, spec.batch_size, spec.input_size, spec.output_size,
        str(spec.data_type),
        row_indices, col_indices, values,
        json.dumps(matmul_options))
    return SparseRepresentation(metainfo, nzvalues.astype(spec.data_type.as_numpy_dtype()))


def triplets_from_representation(spec: MatmulSpec, sparse_data: SparseRepresentation,
                                 matmul_options: dict, n_ipus: int = 1) -> Triplets:
    row_indices, col_indices, values = host_utils.triplets_from_representation(
        n_ipus,
        spec.max_non_zeros, spec.num_groups, spec.batch_size, spec.input_size, spec.output_size,
        str(spec.data_type),
        sparse_data.metainfo_state, sparse_data.nz_values, json.dumps(matmul_options))
    return Triplets(row_indices, col_indices, values)


def get_sparse_tensor_sizes(spec: MatmulSpec, matmul_options: dict, n_ipus: int = 1):
    return host_utils.get_sparse_tensor_sizes(n_ipus,
                                              spec.max_non_zeros, spec.num_groups,
                                              spec.batch_size, spec.input_size, spec.output_size,
                                              str(spec.data_type),
                                              json.dumps(matmul_options))


def triplets_from_dense(matrix: np.ndarray) -> Triplets:
    indices = np.nonzero(matrix)
    values = matrix[indices]
    return Triplets(indices[0].tolist(), indices[1].tolist(), values.tolist())


def dense_from_triplets(spec: MatmulSpec, row_indices: List[int], col_indices: List[int], values: List[float]) -> np.ndarray:
    # Input is multiplied on the left in popsparse:
    dense = np.zeros(shape=[spec.input_size, spec.output_size])
    dense[(row_indices, col_indices)] = values
    return dense


def mask_from_triplets(spec: MatmulSpec, row_indices: List[int], col_indices: List[int], values: List[float]) -> np.ndarray:
    # Input is multiplied on the left in popsparse:
    mask = np.zeros(shape=[spec.input_size, spec.output_size])
    mask[(row_indices, col_indices)] = 1
    return mask


def values_at_indices(row_indices: List[int], col_indices: List[int], matrix: np.ndarray) -> np.ndarray:
    grad_indices = (row_indices, col_indices)
    return matrix[grad_indices]


def random_indices(
        spec: MatmulSpec,
        indices_initialiser_gen: Callable,
        excluded_indices: Tuple[List[int], List[int]] = None,
        count: int = None) -> np.ndarray:
    """
    Generate a random set of row and column indices according to the given matmul spec.
    :param spec: Specification for the matrix multiplication in which the sparse indices will be used.
    :param indices_initialiser_gen: Random number generator (if None, a new one is created with no seed).
    :param excluded_indices: Tuple of lists containing row and column indices that should not be
                             in the generated set.
    :param count: Number of indices to generate. Overrides the max non zeros in spec which is used
                  instead if count is None.
    """

    if indices_initialiser_gen is None:
        indices_initialiser_gen = np.random.default_rng()

    # Input is multiplied on the left in popsparse:
    rows = spec.input_size
    cols = spec.output_size
    number = count if count is not None else spec.max_non_zeros

    # Create a random sample of non-repeating flat indices
    # and then convert them to row, col:
    total_indices = rows * cols
    if total_indices < number:
        raise ValueError(f"Not enough indices (Attempting to draw {number} from set of {total_indices})")

    if excluded_indices is None:
        flat_indices = indices_initialiser_gen.choice(total_indices, size=number, replace=False)
    else:
        fc_shape = (spec.input_size, spec.output_size)
        excluded_flat_indices = np.ravel_multi_index(excluded_indices, fc_shape)

        # NOTE: Forming the total index list is a poor algorithm for very
        # large matrices:
        choose_from = np.delete(np.arange(total_indices), excluded_flat_indices)
        flat_indices = indices_initialiser_gen.choice(choose_from, size=number, replace=False)

    return np.unravel_index(flat_indices, (rows, cols))


def random_triplets(
        spec: MatmulSpec,
        indices_initialiser_gen: Callable,
        value_generator: Callable[..., np.ndarray],
        excluded_indices: Tuple[List[int], List[int]] = None,
        count: int = None) -> Triplets:
    """
    Generate a random set of row and column indices and values according to the given matmul spec.
    :param spec: Specification for the matrix multiplication in which the sparse indices will be used.
    :param seed: Seed for the random number generator.
    :param value_generator: Callable generator to generate the values.
    :param excluded_indices: Tuple of lists containing row and column indices that should not be
                             in the generated set.
    :param count: Number of indices to generate. Overrides the max non zeros in spec which is used
                  instead if count is None.
    """
    row_indices, col_indices = random_indices(spec, indices_initialiser_gen, excluded_indices, count)
    values = value_generator(size=len(row_indices))
    return Triplets(row_indices, col_indices, values)


def disjoint_random_indices(spec: MatmulSpec, size_a: int, size_b: int, indices_initialiser_gen: Callable) -> Tuple[List[int], List[int]]:
    """
    Return two disjoint sets of indices (A and B) for the given matrix spec
    and given set sizes. If size_a + size_b exceeds the total number of indices
    for the matrix given by spec then an exception will be raised.
    :param spec: Specification for the matrix multiplication in which the sparse indices will be used.
    :param size_a: Number of indices in first set.
    :param size_b: Number of indices in second set.
    """
    indices_a = random_indices(spec, indices_initialiser_gen, None, size_a)
    indices_b = random_indices(spec, indices_initialiser_gen, indices_a, size_b)
    return indices_a, indices_b


def gen_sparse_rand_orthog_mat(dimension, sparse_density):
    N = dimension
    max_nonzero = int(np.ceil(sparse_density*N*N))

    #  initialize the sparse matrix to an identity matrix
    Q = np.identity(N)
    Q_nonZero = N
    #  loop and generate the random, orthogonal matrices and use them
    #  to construct the overall random, orthogonal matrix we wish to return
    #  until the maximum desired density is reached
    while Q_nonZero < max_nonzero:
        # sample a random angle
        theta = random.random()*2*math.pi

        # sample a random pair of indices
        index = random.sample(range(N), 2)
        index.sort()

        i, j = index

        # specify the indices and values to initialize the
        # basic random, orthogonal matrix
        I = [i, i, j, j]
        J = [i, j, i, j]
        V = [math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]
        Qk = np.identity(N)
        Qk[I, J] = V
        # update the overall random, orthogonal matrix
        Q = np.matmul(Qk, Q)

        Q[np.isclose(Q, np.zeros(shape=(dimension, dimension)))] = 0
        Q_nonZero = np.count_nonzero(Q)

    # Confirm algorithm has produced an orthogonal matrix
    QTQ = Q.T @ Q
    QQT = Q @ Q.T
    I = np.identity(N)
    rtol = 1e-08
    atol = 1e-07
    assert np.allclose(QTQ, I, atol=atol, rtol=rtol), "Failed orthogonal check, Q.T @ Q does not equal identity"
    assert np.allclose(QQT, I, atol=atol, rtol=rtol), "Failed orthogonal check, Q @ Q.T does not equal identity"

    return Q, Q_nonZero
