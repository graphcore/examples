# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import numpy as np
from functools import reduce
from operator import mul
from operator import add
from subprocess import run

g_random_sparse_mask = np.random.RandomState()
g_random_data = np.random.RandomState()
g_random_labels = np.random.RandomState()

g_high_value = 10.0
g_low_value = -10.0


def build_custom_ops(so_path):
    """
    Build custom ops library.

    inputs:
    custom ops library path
    """
    build_path = os.path.dirname(so_path)
    run(['make', '-j'], cwd=build_path)


def set_seed(seed):
    """
    Set seeds of random generatorts.

    inputs:
    seed: seed value
    """
    g_random_sparse_mask.seed(seed)
    g_random_data.seed(seed)
    g_random_labels.seed(seed)


def create_random_sparse_mask(sparsity, rows, cols):
    """
    Create a random sparse mask.

    inputs:
    sparsity: sparsity level (0.4 means 40% of blocks are empty)
    rows: number of rows
    cols: number of columns

    returns: mask as 2D array of [0|1] values
    """

    assert(sparsity >= 0.0)
    assert(sparsity < 1.0)
    proportion = [sparsity, 1 - sparsity]
    mask = g_random_sparse_mask.choice([0, 1], size=(rows, cols), p=proportion)

    # don't want mask to be all zeros
    while np.all(mask == 0):
        mask = g_random_sparse_mask.choice([0, 1], size=(rows, cols), p=proportion)
    return mask


def create_block_sparse_matrix(dims, block_size, sparsity_mask, initial_value=0):
    """
    Create sparse matrix in dense and block-sparse form

    inputs:
    dims: dimensions of the sparse matrix
    block_size: size of a block (8x8, 16x16 etc)
    sparsity_mask: block-sparsity mask as 2D array (1 for non-zero and 0 for zero block)
    initial_value: the value of masked elements

    returns:
    block_sparse_matrix: np.array of num_blocks * block_sz
    dense_matrix: np.array of size dims with the dense representation of the matrix.
                 i.e. explcit zeros for a zero block. Used to perform
                 dense MM's for a reference output

    e.g for a sparse matrix of size (6, 6) with block size 2x2
    Matrix contains 9 blocks of size 2x2, some sparse and some dense
    If the 6x6 matrices has 2 non zero blocks, then ..

    Inputs:
    dims = [6, 6]
    block_size = [2,2]
    sparsity_mask = [[1, 1, 0],[1, 0, 0],[0, 1, 1]] (say)

    Outputs:
    block_sparse_matrix = 5 x 4 array
    dense_matrix = 6x6 array
    """

    assert(len(dims) == 2)
    assert(len(block_size) == 2)
    assert(dims[0] % block_size[0] == 0)
    assert(dims[1] % block_size[1] == 0)

    rows = dims[0]
    cols = dims[1]

    block_size_row = block_size[0]
    block_size_col = block_size[1]

    num_block_rows = rows // block_size_row
    num_block_cols = cols // block_size_col

    if initial_value == 0:
        dense_matrix = np.zeros((rows, cols))
    else:
        dense_matrix = np.empty((rows, cols))
        dense_matrix.fill(initial_value)

    block_sparse_matrix = []
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            if sparsity_mask[block_row][block_col]:
                block_data = g_random_data.randint(low=g_low_value,
                                                   high=g_high_value,
                                                   size=block_size_row * block_size_col).astype("float32")
                block_sparse_matrix.append(block_data)
                dense_matrix[block_row * block_size_row: (block_row+1) * block_size_row,
                             block_col * block_size_col: (block_col+1) * block_size_col] = block_data.reshape(block_size_row, block_size_col)

    # At this point mask is a 2D array, flatten it into 1D list and return, bsr_rhs is already a list (so convert to array)
    return np.array(block_sparse_matrix), dense_matrix


def create_dense_tensor(dims, data_type="float32"):
    """
    Create a tensor with random elements

    inputs:
    dims: list of tensor dimensions

    returns: created tensor
    """
    return g_random_data.randint(low=g_low_value, high=g_high_value, size=dims).astype(data_type)


def create_block_sparse_tensor(nominal_shape, block_size, sparsity_or_mask, data_type="float32", initial_value=0):
    """
    Create sparse tensor in dense and block-sparse form

    inputs:
    nominal_shape: list of dimensions of the sparse tensor e.g (2, 3, 4, 4)
    block_size : size of each block (e.g. [8, 8])
    sparsity_or_mask : block sparsity level (0.4 means, 40% of blocks are zeros)
                       or existing block sparsity mask as a flattened 2D array
                       (1 for non-zero and 0 for zero block)

    returns:
    sparse_matrix : sparse representation of matrix (nnz_blocks * block size)
    dense_matrix: dense representation of the matrix (for ref calc)
    sparsity_mask : generated or provided block sparsity mask
    """

    assert(len(nominal_shape) >= 2)
    assert(len(block_size) == 2)

    # skip last two dimensions
    # last 2 dims enter the MM, others form the group
    num_grouped_dims = reduce(mul, nominal_shape[:-2], 1)
    rows = nominal_shape[-2]
    cols = nominal_shape[-1]

    assert(rows % block_size[0] == 0)
    assert(cols % block_size[1] == 0)
    block_size_row = block_size[0]
    block_size_col = block_size[1]

    num_block_rows = rows // block_size_row
    num_block_rows_total = num_block_rows * num_grouped_dims
    num_block_cols = cols // block_size_col

    if not isinstance(sparsity_or_mask, list):
        sparsity = sparsity_or_mask
        generate_mask = True
        sparsity_mask_1d = []
    else:
        generate_mask = False
        sparsity_mask_1d = sparsity_or_mask
        assert(len(sparsity_mask_1d) == num_block_rows_total * num_block_cols)
        sparsity_mask = np.reshape(sparsity_mask_1d, (num_block_rows_total, num_block_cols))

    # Create dense matrix of nominal dims
    if initial_value == 0:
        dense_matrix = np.zeros(nominal_shape).astype(data_type)
    else:
        dense_matrix = np.empty(nominal_shape).astype(data_type)
        dense_matrix.fill(initial_value)

    dense_matrix = dense_matrix.reshape((num_grouped_dims, rows, cols))

    dims = [rows, cols]

    sparse_matrix = []
    for g in range(num_grouped_dims):
        if not generate_mask:
            sparsity_mask_1g = sparsity_mask[num_block_rows * g: num_block_rows * (g + 1)]
        else:
            sparsity_mask_1g = create_random_sparse_mask(sparsity, num_block_rows, num_block_cols)
            sparsity_mask_1d.extend(sparsity_mask_1g.flatten())
        assert(reduce(add, sparsity_mask_1g.flatten(), 0) > 0)
        _bsr, dense_matrix[g] = create_block_sparse_matrix(dims, block_size, sparsity_mask_1g, initial_value)
        # _bsr comes as array

        sparse_matrix.extend(_bsr)

    dense_matrix = dense_matrix.reshape(nominal_shape)

    # all parameters are returned as numpy arrays
    return np.array(sparse_matrix, dtype=data_type), dense_matrix, sparsity_mask_1d


def create_random_labels(dims):
    """
    Create random labels tensor.
    Every row contains 1 at random place, the rest elements are 0

    inputs:
    dims: dimensions for the tensor

    returns: generated labels
    """

    assert(len(dims) >= 2)

    lbs = np.zeros(dims, np.int)
    cols = dims[-1]
    lbs = np.reshape(lbs, (-1, cols))
    for r in range(0, lbs.shape[0]):
            idx = g_random_data.randint(0, cols)
            lbs[r][idx] = 1
    lbs = np.reshape(lbs, dims)
    return lbs


def create_random_sparse_labels(dims, sparsity_mask, block_size):
    """
    Create random sparse labels tensor
    Every row contains 1 at a random, but non-masked place, the rest elements are 0

    inputs:
    dims: dimensions for the tensor
    sparsity_mask: block-sparsity mask
    block_size : size of each block

    returns: generated labels
    """

    assert(len(dims) >= 2)
    assert(len(block_size) == 2)
    assert(dims[-1] % block_size[-1] == 0)

    b_cols = dims[-1] // block_size[-1]
    sparsity_mask = np.reshape(sparsity_mask, (-1, b_cols))

    b_rows = sparsity_mask.shape[0]
    b_lbs = np.zeros(sparsity_mask.shape, np.int)
    for b_r in range(0, b_rows):
        nzr = np.sum(sparsity_mask[b_r])
        if nzr > 0:
            b_nzc = g_random_data.randint(1, nzr + 1)
            nzc = 0
            for b_c in range(0, b_cols):
                nzc = nzc + sparsity_mask[b_r][b_c]
                if nzc == b_nzc:
                    b_lbs[b_r][b_c] = 1
                    break
    lbs_1b = np.zeros(block_size, np.int)
    b_row, b_col = (block_size[0], block_size[1])
    for r_b in range(0, b_row):
        idx_b = g_random_data.randint(0, b_col)
        lbs_1b[r_b][idx_b] = 1
    lbs = np.kron(b_lbs, lbs_1b)
    lbs = np.reshape(lbs, dims)
    return lbs


def create_empty_rows_mask(dims, sparsity_mask, block_size, extra_mask=None):
    """
    Create mask of empty rows
    Every row contains all 0 elements if all elements in this row are masked,
    otherwise all elements are 1

    inputs:
    dims: dimensions for the tensor
    sparsity_mask: block-sparsity mask
    block_size : size of each block
    extra_mask: extra mask in elementwise form

    return: Tensor with masked rows

    Example:
    dims = [6, 4]
    sparsity_mask = [[1, 1],[0, 0],[1, 0]]
    block_size = [2, 2]
    extra_mask = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]

    Output = [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]

    Explanation:
    Block-sparsity mask in elemetwise form:
    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    Extra mask:
    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    Combined as AND mask:
    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    Empty rows mask:
    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
    """

    assert(len(dims) >= 2)
    assert(len(block_size) == 2)
    assert(dims[-1] % block_size[-1] == 0)

    b_cols = dims[-1] // block_size[-1]
    sparsity_mask = np.reshape(sparsity_mask, (-1, b_cols))

    cols = dims[-1]
    er_msk = np.ones(dims)
    er_msk = np.reshape(er_msk, (-1, cols))
    rows = er_msk.shape[0]
    msk_1b = np.ones(block_size)
    msk = np.kron(sparsity_mask, msk_1b)
    if extra_mask is not None:
        extra_mask = np.reshape(extra_mask, (-1, cols))
        assert(extra_mask.shape == msk.shape)
        msk = msk * extra_mask

    for r in range(0, rows):
        nzr = np.sum(msk[r])
        if nzr == 0:
            er_msk[r] = np.zeros(cols)
    er_msk = np.reshape(er_msk, dims)
    return er_msk


def create_diagonal_mask(dims, mask_types):
    """
    Create a tensor, containing diagional triangular mask

    inputs:
    dims: dimensions for the tensor
    mask_types: list of mask types for each 2D slice
    0 = no mask
    1 = zero upper triangle
    2 = zero lower triangle

    returns: created tensor

    Example:
    dims = [2, 3, 3]
    mask_types = [1, 2]

    Output:
    [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]]]
    """

    assert(len(dims) >= 2)

    rows = dims[-2]
    cols = dims[-1]
    num_grouped_dims = reduce(mul, dims[:-2], 1)
    assert(num_grouped_dims == len(mask_types))
    dims2d = (num_grouped_dims, rows, cols)
    mask = np.ones(dims2d)
    for g in range(0, num_grouped_dims):
        if mask_types[g] == 1:
            for r in range(0, rows):
                for c in range(r + 1, cols):
                    mask[g][r][c] = 0
        elif mask_types[g] == 2:
            for r in range(0, rows):
                for c in range(0, min(r, cols)):
                    mask[g][r][c] = 0
    mask = np.reshape(mask, dims)
    return mask


def to_block_sparse(dense_tensor, block_size, sparsity_mask, data_type="float32"):
    """
    Convert a dense tensor into blok-sparse format.

    inputs:
    dense_tensor: input tensor as numpy array, can have 2 dimensions or more
    block_size: block size as a tuple (block row length, block column length)
    sparsity_mask: sparsity matrix as a flattened 2D array.
        If dense tensor has more that 2 dimensions, the sparsity mask must cover all 2D slices
    data_type: data type for the output tensor
    returns: tensor in block-sparse format: [total number of non-zero blocks, block row * block col]
    """

    nominal_shape = dense_tensor.shape
    assert(len(nominal_shape) >= 2)
    num_grouped_dims = reduce(mul, nominal_shape[:-2], 1)

    rows = nominal_shape[-2]
    cols = nominal_shape[-1]
    block_size_row = block_size[0]
    block_size_col = block_size[1]
    assert(rows % block_size_row == 0)
    assert(cols % block_size_col == 0)
    num_block_rows = rows // block_size_row
    num_block_cols = cols // block_size_col

    dense_tensor = dense_tensor.reshape((num_grouped_dims, rows, cols))
    block_sparse_matrix = []
    idx_sparse = 0
    for g in range(num_grouped_dims):
        r = 0
        for br in range(num_block_rows):
            r1 = r + block_size_row
            c = 0
            for bc in range(num_block_cols):
                c1 = c + block_size_col
                if sparsity_mask[idx_sparse] == 1:
                    block_sparse_matrix.append(dense_tensor[g, r:r1, c:c1].flatten())
                c = c1
                idx_sparse = idx_sparse + 1
            r = r1
    return np.array(block_sparse_matrix, dtype=data_type)


def get_lib_path(lib_name):
    """
    Get full library path,
    assuming library file is located in the same directory as this file

    inputs: library name (without lib prefix)
    returns: full path of a library
    """

    base_path = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(base_path, "lib" + lib_name + ".so")
