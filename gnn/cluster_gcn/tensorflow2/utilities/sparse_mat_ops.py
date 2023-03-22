# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor, SparseKerasTensor

from utilities.constants import AdjacencyForm

TENSOR_TYPES = (tf.Tensor, KerasTensor)
SPARSE_TENSOR_TYPES = (tf.SparseTensor, SparseKerasTensor)


def assert_sparse_tensor(x):
    assert isinstance(x, tf.sparse.SparseTensor), f"Expected type to be SparseTensor, but provided {type(x)}."


def assert_sparse_tuple(x):
    assert isinstance(x, tuple), f"Expected type to be a tuple, but provided {type(x)}."
    assert len(x) == 3, f"Expected 3 elements, but provided {len(x)}."


def dtype_(x):
    """
    Helper function to return the dtype of the different ways of expressing
    a sparse tensor, namely Tensor, SparseTensor, or tuple.
    """
    if isinstance(x, tuple):
        return x[1].dtype
    else:
        return x.dtype


def indices_(x):
    if isinstance(x, tuple):
        return x[0]
    else:
        return x.indices


def shape_(x):
    if isinstance(x, tuple):
        return x[2]
    else:
        return x.shape


def values_(x):
    if isinstance(x, tuple):
        return x[1]
    else:
        return x.values


# Ops that operate when the inputs include a sparse matrix in
# tuple or tf.SparseTensor form and whose result is a
# sparse matrix represented as a tuple, which will be converted
# to tf.SparseTensor by consumer function if needed.
# ===========================================================


def sparse_diag(diag_vec):
    """
    Returns sparse diagonal matrix in tuple form, with entries given by the input vector.
    The output tuple will be converted to SparseTensor or SparseTuple by the consumer function.
    """
    size = diag_vec.shape[0]
    nodes = tf.expand_dims(tf.range(size), axis=-1)
    indices = tf.concat((nodes, nodes), axis=1)
    return indices, diag_vec, (size, size)


def sparse_diag_matmul(a_diag, sp_b):
    """
    Computes a_diag_mat @ sp_b, where a_diag_mat is a dense diagonal matrix with diagonal entries
    given by the input vector a_diag, and sp_b is a sparse matrix. The result is a sparse matrix
    represented as a tuple, whose rows are scaled by the diagonal entries of a_diag_mat.
    The output tuple will be converted to SparseTensor or SparseTuple by the consumer function.
    """
    scaled_values = values_(sp_b) * tf.gather(a_diag, indices_(sp_b)[:, 0])
    return indices_(sp_b), scaled_values, shape_(sp_b)


def sparse_diag_part(sp_x):
    """Returns dense vector whose entries are the diagonal elements of the sparse input matrix."""
    mask = sp_x.indices[:, 0] == sp_x.indices[:, 1]
    diag_indices = sp_x.indices[mask][:, 0]
    diag_values = sp_x.values[mask]
    return tf.scatter_nd(tf.reshape(diag_indices, (-1, 1)), diag_values, (sp_x.shape[0],))


def sparse_tuple_diag_part(sp_x):
    """Returns dense vector whose entries are the diagonal elements of the sparse input matrix."""
    num_edges = indices_(sp_x).shape[0]
    # Generate a mask with True for self-edges.
    mask = tf.equal(indices_(sp_x)[:, 0], indices_(sp_x)[:, 1])
    # Replace True values in mask with the index of the corresponding
    # self-edges, and False values with the index of the fake node.
    diag_indices = tf.where(mask, tf.range(num_edges), num_edges - 1)
    # Since fake node is the largest one, we can sort to get the
    # indices of self-edges first.
    diag_indices = tf.sort(diag_indices)
    # Get the indices of the self-edges and dismiss the rest.
    diag_indices = tf.slice(diag_indices, (0,), (shape_(sp_x)[0],))
    # Gather the values corresponding to the self-edges.
    diag_values = tf.gather(values_(sp_x), diag_indices)
    return diag_values


# Ops that operate on inputs that include a sparse matrix
# represented as a SparseTuple.
# ===========================================================


def sparse_tuple_add_diag(sp_a, sp_diag):
    """
    Returns sp_a + sp_diag, where sp_a is a sparse matrix and
    sp_diag is a diagonal matrix, both represented in tuple form.
    """
    assert_sparse_tuple(sp_a)
    assert_sparse_tuple(sp_diag)

    # In order to sum values, scatter the diagonal values so that
    # they match the position of the diagonal elements in sp_a.
    num_edges = indices_(sp_a).shape[0]

    # Generate a mask with True for self-edges.
    mask = tf.equal(indices_(sp_a)[:, 0], indices_(sp_a)[:, 1])
    # Replace True values in mask with the index of the corresponding
    # self-edges, and False values with the index of the fake node.
    diag_indices = tf.where(mask, tf.range(num_edges), num_edges - 1)
    # Since fake node is the largest one, we can sort to get the
    # indices of self-edges first.
    diag_indices = tf.sort(diag_indices)
    # Scatter the values of the diagonal matrix in the positions of
    # the self-edges.
    diagvalues_ = tf.scatter_nd(
        tf.expand_dims(diag_indices[: shape_(sp_diag)[0]], axis=1), values_(sp_diag), (num_edges,)
    )
    # Add the values of the diagonal matrix.
    new_values = values_(sp_a) + diagvalues_
    return indices_(sp_a), new_values, shape_(sp_a)


def sparse_tuple_dense_matmul(sp_a, b):
    """
    We assume the adjacency has no zero rows/columns, otherwise we would have to scatter.
    """
    y = tf.expand_dims(values_(sp_a), axis=-1) * tf.gather(b, indices_(sp_a)[:, 1])
    z = tf.math.unsorted_segment_sum(y, indices_(sp_a)[:, 0], num_segments=shape_(sp_a)[0])
    return z


def sparse_tuple_eye(size, dtype):
    """
    Return the identity matrix represented as a SparseTuple.
    """
    nodes = tf.expand_dims(tf.range(size), axis=-1)
    indices = tf.concat((nodes, nodes), axis=-1)
    values = tf.ones((size,), dtype=dtype)
    return indices, values, (size, size)


def sparse_tuple_scale_by_constant(sp_a, c):
    """
    Returns a sparse tuple where the values are scaled by a constant.
    """
    assert_sparse_tuple(sp_a)
    return indices_(sp_a), values_(sp_a) * c, shape_(sp_a)


def sparse_tuple_sum_rows(sp_a):
    """
    Returns dense tensor whose entries are the sum of the rows of a sparse matrix expressed
    in tuple form.
    """
    assert_sparse_tuple(sp_a)
    return tf.math.unsorted_segment_sum(values_(sp_a), indices_(sp_a)[:, 0], num_segments=shape_(sp_a)[0])


# High level ops that switch between dense and sparse ops.
# =================================================================


def add(a, b):
    """Computes the sum of one matrix and a diagonal one."""
    if isinstance(a, SPARSE_TENSOR_TYPES) and isinstance(b, SPARSE_TENSOR_TYPES):
        return tf.sparse.add(a, b)
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return sparse_tuple_add_diag(a, b)
    else:
        return a + b


def eye(size, dtype, adjacency_form):
    """Returns dense or sparse the identity matrix of a given size and type."""
    if adjacency_form == AdjacencyForm.DENSE:
        return tf.eye(size, dtype=dtype)

    if adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        return tf.sparse.eye(size, dtype=dtype)

    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        return sparse_tuple_eye(size, dtype)


def diag(diag_vec, adjacency_form):
    """Returns a dense or sparse diagonal matrix with diagonal elements given by the input
    vector diag_vec."""
    if adjacency_form == AdjacencyForm.DENSE:
        return tf.linalg.diag(diag_vec)
    if adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        indices, values, shape = sparse_diag(diag_vec)
        indices = tf.cast(indices, tf.int64)
        shape = tf.cast(shape, tf.int64)
        return tf.SparseTensor(indices, values, shape)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        return sparse_diag(diag_vec)


def diag_matmul(a_diag_vec, sp_b):
    """Computes the matrix product of a diagonal matrix, with values given by the entries of the
    input vector a_diag_vec, times a sparse matrix. The result can be a sparse or dense matrix."""
    if isinstance(sp_b, TENSOR_TYPES):
        a_diag_mat = tf.linalg.diag(a_diag_vec)
        return tf.linalg.matmul(a_diag_mat, sp_b, a_is_sparse=True, b_is_sparse=True)
    if isinstance(sp_b, SPARSE_TENSOR_TYPES):
        return tf.SparseTensor(*sparse_diag_matmul(a_diag_vec, sp_b))
    if isinstance(sp_b, tuple):
        return sparse_diag_matmul(a_diag_vec, sp_b)


def diag_part(x):
    """Returns a dense vector with entries given by the diagonal of the input matrix."""
    if isinstance(x, TENSOR_TYPES):
        return tf.linalg.diag_part(x)
    if isinstance(x, SPARSE_TENSOR_TYPES):
        return sparse_diag_part(x)
    if isinstance(x, tuple):
        return sparse_tuple_diag_part(x)


def reduce_sum_rows(a):
    """Returns a dense vector whose entries are the sum of elements across the rows of a matrix."""
    if isinstance(a, TENSOR_TYPES):
        return tf.math.reduce_sum(a, axis=1)
    if isinstance(a, SPARSE_TENSOR_TYPES):
        return tf.sparse.reduce_sum(a, axis=1)
    if isinstance(a, tuple):
        return sparse_tuple_sum_rows(a)


def scale_by_constant(a, c):
    """Computes the product of a matrix, a, times a scalar value, c. The output can be a
    dense or sparse matrix."""
    if isinstance(a, TENSOR_TYPES):
        return a * c
    if isinstance(a, SPARSE_TENSOR_TYPES):
        return tf.sparse.map_values(tf.multiply, a, c)
    if isinstance(a, tuple):
        return sparse_tuple_scale_by_constant(a, c)


def sp_dense_matmul(sp_a, b):
    """Computes a dense matrix that results from the product of a sparse matrix and a dense one."""
    if isinstance(sp_a, TENSOR_TYPES):
        return tf.linalg.matmul(sp_a, b, a_is_sparse=True)
    if isinstance(sp_a, SPARSE_TENSOR_TYPES):
        return tf.sparse.sparse_dense_matmul(sp_a, b)
    if isinstance(sp_a, tuple):
        return sparse_tuple_dense_matmul(sp_a, b)
