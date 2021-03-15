# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import numpy as np
from functools import reduce
from operator import mul
import popart
import pytest
import ctypes


so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

# range for filling blocks
MATRIX_LOW_VALUE = -10
MATRIX_HIGH_VALUE = 10

# library provides support for 3 kinds of sparse MM's
# from set of 2 inputs and one output, 2 of them will be dense
# and the third will be sparse
g_sparseMatMulTypeLookup = {
    'DENSE_LHS_SPARSE_RHS_DENSE_OUT': 0,
    'DENSE_LHS_DENSE_RHS_SPARSE_OUT': 1,
    'SPARSE_LHS_SPARSE_RHS_SPARSE_OUT': 2
}

# g_ -> global
#
g_input_data_type = "float32"
g_output_data_type = "float32"
g_pp_data_type = "float32"

np.set_printoptions(linewidth=500)

g_random_sparse_mask = np.random.RandomState()
g_random_data = np.random.RandomState()
g_random_labels = np.random.RandomState()


"""
Set seeds of random generatorts.
"""
g_random_sparse_mask.seed(1)
g_random_data.seed(1)
g_random_labels.seed(1)


def create_sparse_list(dims, block_size, sparsity, initial_value=0):
    """
    dims: dimensions of the sparse matrix
    block_size: size of a block (8x8, 16x16 etc)
    sparsity: sparsity level (0.4 means 40% of blocks are empty)

    returns--
    block_sparse_matrix: np.array of num_blocks * block_sz
    dense_matrix: np.array if size dim with a dense representation of the matrix.
                 i.e. explcit zeros for a zero block. Used to perform
                 dense MM's for a reference output
    mask: list with sparsity pattern. size = num_blocks (both dense and sparse)

    e.g for a sparse matrix of size (6, 6) with block size 2x2
    Matrix contains 9 blocks of size 2x2, some sparse and some dense
    If the 6x6 matrices has 2 non zero blocks, then ..

    Inputs:
    dims = [6, 6]
    block_size = [2,2]
    sparsity = 0.4 (say)

    Outputs:
    block_sparse_matrix = 2 x 4 array
    dense_matrix = 6x6 array
    sparsity = 9x1 list
    """
    block_size_row = block_size[0]
    block_size_col = block_size[1]

    num_block_rows = dims[0] // block_size_row
    num_block_cols = dims[1] // block_size_col

    assert(sparsity < 1.0)
    proportion = [sparsity, 1 - sparsity]
    mask = g_random_sparse_mask.choice([0, 1], size=(num_block_rows, num_block_cols), p=proportion)

    # dont want mask to be all zeros
    while np.all(mask == 0):
        mask = g_random_sparse_mask.choice([0, 1], size=(num_block_rows, num_block_cols), p=proportion)

    if initial_value == 0:
        dense_matrix = np.zeros((num_block_rows * block_size_row,
                                 num_block_cols * block_size_col))
    else:
        dense_matrix = np.empty((num_block_rows * block_size_row,
                                 num_block_cols * block_size_col))
        dense_matrix.fill(initial_value)

    block_sparse_matrix = []
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            if mask[block_row][block_col]:
                block_data = g_random_data.randint(low=MATRIX_LOW_VALUE,
                                                   high=MATRIX_HIGH_VALUE,
                                                   size=block_size_row * block_size_col).astype("float32")
                block_sparse_matrix.append(block_data)
                dense_matrix[block_row * block_size_row: (block_row+1) * block_size_row,
                             block_col * block_size_col: (block_col+1) * block_size_col] = block_data.reshape(block_size_row, block_size_col)

    # At this point mask is a 2D array, flatten it into 1D list and return, bsr_rhs is already a list (so convert to array)
    return np.array(block_sparse_matrix), dense_matrix, mask.flatten().tolist()


def create_dense_matrix(dims):
    return g_random_data.randint(low=MATRIX_LOW_VALUE, high=MATRIX_HIGH_VALUE, size=dims).astype(g_input_data_type)


def create_sparse_matrix(nominal_shape, block_size, sparsity, initial_value=0):
    """
    Create a sparse_matrix.

    Inputs:
    nominal_shape: List of dimensions of the sparse tensor e.g (2, 3, 4, 4)
    block_size   : size of each block (e.g. [8, 8, 8])
    sparsity     : block sparsity level (0.4 means, 40% of blocks are zeros)

    Outputs:
    bsr : sparse representation of matrix (nnz_blocks * block size)
    lengths_per_2d_plane: List with num-non-zero blocks per group dim.
    i.e. for a (2, 3, 4, 4) tensor with 2 nnz_blocks in each 4x4 matrix,
    this will have shape of 6x1 and each row storing 2
    dense_matrix: dense representation of the matrix (for ref calc)
    mask : list of num_blocks (1 for non-zero blocks and 0 for others)
    """
    # skip last two dimensions
    # last 2 dims enter the MM, others form the group
    num_grouped_dims = reduce(mul, nominal_shape[:-2], 1)
    rows = nominal_shape[-2]
    cols = nominal_shape[-1]

    # Create dense matrix of nominal dims
    if initial_value == 0:
        dense_matrix = np.zeros(nominal_shape).astype(g_input_data_type)
    else:
        dense_matrix = np.empty(nominal_shape).astype(g_input_data_type)
        dense_matrix.fill(initial_value)

    dense_matrix = dense_matrix.reshape((num_grouped_dims, rows, cols))

    dims = [nominal_shape[-2], nominal_shape[-1]]

    bsr = []
    bsr_lengths_per_2d_plane = []
    mask = []
    for dim in range(num_grouped_dims):
        _bsr, dense_matrix[dim], _mask = create_sparse_list(dims, block_size, sparsity, initial_value)
        # _bsr comes as array
        # _mask comes as list

        bsr.extend(_bsr)
        mask.extend(_mask)

        bsr_lengths_per_2d_plane.append(_bsr.shape[0])

    dense_matrix = dense_matrix.reshape(nominal_shape)
    mask = np.array(mask)

    block_size_row = block_size[0]
    block_size_col = block_size[1]

    num_block_rows = dims[0] // block_size_row
    num_block_cols = dims[1] // block_size_col

    # all parameters are returned as numpy arrays
    return np.array(bsr), np.array(bsr_lengths_per_2d_plane), dense_matrix, mask


def mm(lhs, rhs):
    return np.matmul(lhs, rhs)


# Stable softmax numpy implementation
def softmax(x):
    x_max = np.max(x, axis = -1)
    x = x - np.expand_dims(x_max, axis=-1)
    x = np.exp(x)
    x_sum = np.sum(x, axis=-1)
    x = x / np.expand_dims(x_sum, axis=-1)
    return x


def sparse_mm_infer(sparse_mm_type, lhs_dims, vanilla_rhs_dims, block_size, sparsity_level, transpose_rhs, memory_cycle_ratio, inner_group_size):
    """ """
    if transpose_rhs:
        matmul_dims = [lhs_dims[-2], vanilla_rhs_dims[-1], vanilla_rhs_dims[-2]]
    else:
        matmul_dims = [lhs_dims[-2], vanilla_rhs_dims[-2], vanilla_rhs_dims[-1]]

    lhs = create_dense_matrix(lhs_dims)
    if sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT']:
        bsr_rhs, lengths_per_2d_plane, vanilla_rhs, sparsity_mask = create_sparse_matrix(vanilla_rhs_dims, block_size[1:], sparsity_level)

        rhs = bsr_rhs
        rhs_dims = bsr_rhs.shape
    elif sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT']:
        output_dims = lhs_dims[:-1]
        output_dims.append(vanilla_rhs_dims[-1])
        output_block_size = [block_size[0], block_size[2]]

        bsr_output, lengths_per_2d_plane, _, sparsity_mask = create_sparse_matrix(output_dims, output_block_size, sparsity_level)

        rhs_dims = vanilla_rhs_dims
        rhs = create_dense_matrix(rhs_dims)

    # Create a builder and construct a graph
    builder = popart.Builder()

    lhs_tensorInfo = popart.TensorInfo("FLOAT", lhs_dims)
    rhs_tensorInfo = popart.TensorInfo("FLOAT", rhs_dims)

    lhsTensor = builder.addInputTensor(lhs_tensorInfo)
    rhsTensor = builder.addInputTensor(rhs_tensorInfo)

    outTensor = builder.customOp(opName = "BSMatMul",
                                 opVersion=1,
                                 domain = "ai.graphcore",
                                 inputs = [lhsTensor, rhsTensor],
                                 attributes = {
                                  "bsr_rhs_lengths_per_2d_plane": lengths_per_2d_plane.tolist(),
                                  "matrix_dims": matmul_dims,
                                  "block_size": block_size,
                                  "sparsity_mask": sparsity_mask.tolist(),
                                  "bsmatmul_type": sparse_mm_type,
                                  "transpose_rhs": transpose_rhs,
                                  "memory_cycle_ratio": memory_cycle_ratio,
                                  "inner_group_size": inner_group_size,
                                  "in_type": g_input_data_type,
                                  "out_type": g_output_data_type,
                                  "pp_type": g_pp_data_type
                                 })[0]

    builder.addOutputTensor(outTensor)

    proto = builder.getModelProto()

    # Describe how to run the model
    dataFlow = popart.DataFlow(1, {outTensor: popart.AnchorReturnType("ALL")})

    # Create a session to compile and execute the graph
    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(1))

    # Compile graph
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    rhs = np.array(rhs, dtype=g_input_data_type)

    stepio = popart.PyStepIO({lhsTensor: lhs, rhsTensor: rhs}, anchors)
    session.run(stepio)

    ipuOutput = anchors[outTensor]

    if sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT']:
        if transpose_rhs:
            transpose_indices = list(range(len(vanilla_rhs_dims)))
            transpose_indices[-2], transpose_indices[-1] = transpose_indices[-1], transpose_indices[-2]

            vanilla_rhs = vanilla_rhs.transpose(tuple(transpose_indices))
            goldOutput = mm(lhs, vanilla_rhs)
        else:
            goldOutput = mm(lhs, vanilla_rhs)
    else:
        assert len(lhs.shape) == len(rhs.shape)
        if(len(lhs.shape) == 2):
            lhs = np.expand_dims(lhs, 0)
            rhs = np.expand_dims(rhs, 0)

        mmOutput = mm(lhs, rhs)

        totalGroupDims = int(np.prod(lhs_dims[:-2]))

        num_rows_sparsity_mask_2d = output_dims[-2] // block_size[0]
        num_cols_sparsity_mask_2d = output_dims[-1] // block_size[2]

        assert sparsity_mask.shape == (totalGroupDims * num_rows_sparsity_mask_2d * num_cols_sparsity_mask_2d,)
        mmOutput = mmOutput.reshape((totalGroupDims, lhs_dims[-2], rhs_dims[-1]))

        goldOutput = []
        for dim in range(totalGroupDims):
            offset = num_rows_sparsity_mask_2d * num_cols_sparsity_mask_2d
            mmOutput_2d = mmOutput[dim]
            sliced_sparsity_mask = sparsity_mask[dim * offset: dim * offset + offset]

            for sparsity_mask_idx in range(len(sliced_sparsity_mask)):
                if sliced_sparsity_mask[sparsity_mask_idx]:
                    mmOutput_2d_row_start = (sparsity_mask_idx // num_cols_sparsity_mask_2d) * block_size[0]
                    mmOutput_2d_row_end = mmOutput_2d_row_start + block_size[0]

                    mmOutput_2d_col_start = (sparsity_mask_idx % num_cols_sparsity_mask_2d) * block_size[2]
                    mmOutput_2d_col_end = mmOutput_2d_col_start + block_size[2]

                    mmOutput_2d_sliced = mmOutput_2d[mmOutput_2d_row_start: mmOutput_2d_row_end, mmOutput_2d_col_start: mmOutput_2d_col_end]
                    goldOutput.append(mmOutput_2d_sliced.reshape(block_size[0] * block_size[2]))

        goldOutput = np.array(goldOutput)

    return ipuOutput, goldOutput


def sparse_mm_train(sparse_mm_type, lhs_dims, vanilla_rhs_dims, block_size, sparsity_level, transpose_rhs, memory_cycle_ratio, inner_group_size):
    if transpose_rhs:
        matmul_dims = [lhs_dims[-2], vanilla_rhs_dims[-1], vanilla_rhs_dims[-2]]
    else:
        matmul_dims = [lhs_dims[-2], vanilla_rhs_dims[-2], vanilla_rhs_dims[-1]]

    lhs = create_dense_matrix(lhs_dims)
    if sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT']:
        bsr_rhs, lengths_per_2d_plane, vanilla_rhs, sparsity_mask = create_sparse_matrix(vanilla_rhs_dims, block_size[1:], sparsity_level)

        rhs = bsr_rhs
        rhs_dims = bsr_rhs.shape

    elif sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT']:
        output_dims = lhs_dims[:-1]
        output_dims.append(vanilla_rhs_dims[-1])
        output_block_size = [block_size[0], block_size[2]]

        bsr_output, lengths_per_2d_plane, vanilla_output, sparsity_mask = create_sparse_matrix(output_dims, output_block_size, sparsity_level)

        lhs_inv = np.linalg.inv(lhs)

        rhs = np.matmul(lhs_inv, vanilla_output)
        rhs_dims = vanilla_rhs_dims

    # MODEL CREATION
    builder = popart.Builder()

    lhs_tensorInfo = popart.TensorInfo("FLOAT", lhs_dims)
    lhsTensor = builder.addInputTensor(lhs_tensorInfo)
    rhsTensor = builder.addInitializedInputTensor(rhs)

    outTensor = builder.customOp(opName = "BSMatMul",
                                 opVersion=1,
                                 domain = "ai.graphcore",
                                 inputs = [lhsTensor, rhsTensor],
                                 attributes = {
                                  "bsr_rhs_lengths_per_2d_plane": lengths_per_2d_plane.tolist(),
                                  "matrix_dims": matmul_dims,
                                  "block_size": block_size,
                                  "sparsity_mask": sparsity_mask.tolist(),
                                  "bsmatmul_type": sparse_mm_type,
                                  "transpose_rhs": transpose_rhs,
                                  "memory_cycle_ratio": memory_cycle_ratio,
                                  "inner_group_size": inner_group_size,
                                  "in_type": g_input_data_type,
                                  "out_type": g_output_data_type,
                                  "pp_type": g_pp_data_type
                                 })[0]

    builder.addOutputTensor(outTensor)

    probs = builder.aiOnnx.softmax([outTensor], axis=1)

    if sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT']:
        labels_shape = lhs_dims[:-1]
    elif sparse_mm_type == g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT']:
        labels_shape = [np.sum(sparsity_mask)]

    label_tensorInfo = popart.TensorInfo("INT32", labels_shape)
    labelTensor = builder.addInputTensor(label_tensorInfo)

    loss = builder.aiGraphcore.nllloss([probs, labelTensor], debugPrefix = "nllLossVal")

    proto = builder.getModelProto()
    #######################

    # Describe how to run the model
    anchor_desc = {
        outTensor: popart.AnchorReturnType("ALL"),
        loss: popart.AnchorReturnType("ALL")
    }

    dataFlow = popart.DataFlow(1, anchor_desc)

    label_data = g_random_labels.choice(9, labels_shape)

    session = popart.TrainingSession(fnModel=proto,
                                     loss=loss,
                                     deviceInfo=popart.DeviceManager().acquireAvailableDevice(1),
                                     optimizer=popart.ConstSGD(0.01),
                                     dataFlow=dataFlow)

    # Compile graph
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    # TRAINING
    session.weightsFromHost()

    stepio = popart.PyStepIO({
            lhsTensor: lhs,
            labelTensor: label_data}, anchors)

    session.run(stepio)


def sparse_softmax(dims, block_size, sparsity_level, inner_group_size):
    """ """

    sparse_input, lengths_per_2d_plane, dense_input, sparsity_mask = create_sparse_matrix(dims, block_size, sparsity_level, -1000)

    # Create a builder and construct a graph
    builder = popart.Builder()

    tensor_info = popart.TensorInfo("FLOAT", sparse_input.shape)
    input_tensor = builder.addInputTensor(tensor_info)

    output_tensor = builder.customOp(opName = "BsSoftmax",
                                     opVersion = 1,
                                     domain = "ai.graphcore",
                                     inputs = [input_tensor],
                                     attributes = {
                                      "matrixDims": dims,
                                      "blockSize": block_size,
                                      "sparsity": sparsity_mask.tolist(),
                                      "groupSizes": lengths_per_2d_plane.tolist(),
                                      "innerGroupSize": inner_group_size,
                                      "subBlockMaskPerGroup": "None" * len(lengths_per_2d_plane)
                                     })[0]
    builder.addOutputTensor(output_tensor)

    proto = builder.getModelProto()

    # Describe how to run the model
    dataFlow = popart.DataFlow(1, {output_tensor: popart.AnchorReturnType("ALL")})

    # Create a session to compile and execute the graph
    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(1))

    # Compile graph
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    sparse_input = np.array(sparse_input, dtype=g_input_data_type)
    stepio = popart.PyStepIO({input_tensor: sparse_input}, anchors)
    session.run(stepio)

    ipu_output = anchors[output_tensor]

    group_dims = dims[:-2]
    mat_dims = dims[-2:]
    blocks_2d = [mat_dims[0] // block_size[0], mat_dims[1] // block_size[1]]
    num_blocks_2d = blocks_2d[0] * blocks_2d[1]
    block_area = block_size[0] * block_size[1]

    total_group_dims = int(np.prod(group_dims))
    assert sparsity_mask.shape == (total_group_dims * num_blocks_2d,)

    cpu_output = softmax(dense_input)

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    cpu_output = cpu_output.reshape([total_group_dims, blocks_2d[0], block_size[0], blocks_2d[1], block_size[1]])
    cpu_output = np.transpose(cpu_output, [0, 1, 3, 2, 4])
    cpu_output = cpu_output.reshape(total_group_dims, num_blocks_2d, block_area)

    gold_output = []
    offset = 0
    for g in range(total_group_dims):
        cpu_output_2d = cpu_output[g]

        sliced_sparsity_mask = sparsity_mask[offset: offset + num_blocks_2d]
        offset = offset + num_blocks_2d
        for sparsity_mask_idx in range(num_blocks_2d):
            if sliced_sparsity_mask[sparsity_mask_idx]:
                gold_output.append(cpu_output_2d[sparsity_mask_idx])

    gold_output = np.array(gold_output)
    assert ipu_output.shape == gold_output.shape

    return ipu_output, gold_output


#
# INFERENCE TEST
#

# test_data_infer tuple --> (matMulType, lhs_dims, rhs_dims, block_size, sparsity, transpose_rhs, inner_group_size)
test_data_infer = [
    # 2D
    ("tag_inf_0", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_1", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, False, 1),
    ("tag_inf_2", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [32, 32], [32, 32], [16, 8, 8], 0.8, False, 1),
    ("tag_inf_3", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [64, 64], [64, 256], [64, 8, 64], 0.9, False, 1),
    ("tag_inf_4", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [128, 128], [128, 128], [32, 8, 16], 0.2, False, 1),

    # 3D, False
    ("tag_inf_5", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 8, 8], [2, 8, 8], [8, 8, 8], 0.1, False, 1),
    ("tag_inf_6", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [5, 16, 16], [5, 16, 16], [8, 8, 8], 0.3, False, 1),
    ("tag_inf_7", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [7, 32, 32], [7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_8", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [11, 64, 64], [11, 64, 64], [64, 8, 64], 0.6, False, 1),
    ("tag_inf_9", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [12, 128, 128], [12, 128, 128], [32, 8, 16], 0.8, False, 1),

    # 4D, False
    ("tag_inf_10", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 8, 8], [1, 1, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_11", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 16, 16], [1, 1, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_12", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 32, 32], [1, 1, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_13", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 64, 128], [1, 1, 128, 256], [64, 8, 64], 0.5, False, 1),
    ("tag_inf_14", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 128, 64], [1, 1, 64, 128], [32, 8, 16], 0.5, False, 1),
    ("tag_inf_14", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 2, 8, 8], [1, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_16", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 5, 16, 16], [1, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_17", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 7, 32, 32], [1, 7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_18", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 5, 64, 128], [1, 5, 128, 256], [64, 8, 64], 0.5, False, 1),
    ("tag_inf_19", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 12, 128, 64], [1, 12, 64, 128], [32, 8, 16], 0.5, False, 1),
    ("tag_inf_20", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_21", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [5, 5, 16, 16], [5, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_22", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [13, 7, 32, 32], [13, 7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_24", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 12, 128, 64], [1, 12, 64, 128], [32, 8, 16], 0.5, False, 1),

    # 2D, lhs has to be square to take inverse, False
    ("tag_inf_25", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_26", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, False, 1),
    ("tag_inf_27", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [32, 32], [32, 32], [16, 8, 8], 0.8, False, 1),
    ("tag_inf_28", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [64, 64], [64, 64], [64, 8, 64], 0.9, False, 1),
    ("tag_inf_29", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [128, 128], [128, 128], [32, 8, 16], 0.7, False, 1),

    # 3D, lhs has to be square to take, False
    ("tag_inf_30", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [2, 8, 8], [2, 8, 8], [8, 8, 8], 0.1, False, 1),
    ("tag_inf_31", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [5, 16, 16], [5, 16, 16], [8, 8, 8], 0.3, False, 1),
    ("tag_inf_32", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [7, 32, 32], [7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_33", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [11, 64, 64], [11, 64, 64], [64, 8, 64], 0.6, False, 1),
    ("tag_inf_34", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [12, 128, 128], [12, 128, 128], [32, 8, 16], 0.1, False, 1),

    # 4D, lhs has to be square to take, False
    ("tag_inf_36", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 8, 8], [1, 1, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_36", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 16, 16], [1, 1, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_37", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 32, 32], [1, 1, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_38", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 64, 64], [1, 1, 64, 256], [64, 8, 64], 0.5, False, 1),
    ("tag_inf_39", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 128, 128], [1, 1, 128, 128], [32, 8, 16], 0.5, False, 1),
    ("tag_inf_40", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 2, 8, 8], [1, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_41", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 5, 16, 16], [1, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_42", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 7, 32, 32], [1, 7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_43", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 11, 64, 64], [1, 11, 64, 256], [64, 8, 64], 0.5, False, 1),
    ("tag_inf_44", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 12, 128, 128], [1, 12, 128, 128], [32, 8, 16], 0.5, False, 1),
    ("tag_inf_45", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_inf_46", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [5, 5, 16, 16], [5, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_inf_47", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [13, 7, 32, 32], [13, 7, 32, 32], [16, 8, 8], 0.5, False, 1),
    ("tag_inf_49", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 12, 128, 128], [1, 12, 128, 1024], [32, 8, 16], 0.5, False, 1),

    # For transpose_rhs True case, last 2 dimensions of block_size must be 8
    ("tag_inf_50", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_inf_51", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_inf_52", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [32, 32], [32, 32], [16, 8, 8], 0.8, True, 1),
    ("tag_inf_53", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [64, 64], [256, 64], [64, 8, 8], 0.9, True, 1),
    ("tag_inf_54", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [128, 128], [128, 128], [32, 8, 8], 0.2, True, 1),
    ("tag_inf_55", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 8, 8], [2, 8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_inf_56", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [3, 16, 16], [3, 16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_inf_57", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [7, 128, 128], [7, 128, 128], [32, 8, 8], 0.2, True, 1),
    ("tag_inf_58", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [3, 5, 8, 8], [3, 5, 8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_inf_59", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 6, 16, 16], [1, 6, 16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_inf_60", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [4, 4, 32, 32], [4, 4, 32, 32], [16, 8, 8], 0.8, True, 1),

    # 3D, inner group size > 1
    ("tag_inf_61", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [12, 128, 128], [12, 128, 128], [32, 8, 16], 0.8, False, 3),
    ("tag_inf_62", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [12, 128, 128], [12, 128, 128], [32, 8, 16], 0.1, False, 4),

    # 4D, inner group size > 1
    ("tag_inf_23", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [4, 8, 64, 128], [4, 8, 128, 256], [64, 8, 64], 0.5, False, 4),
    ("tag_inf_48", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [3, 11, 64, 64], [3, 11, 64, 256], [64, 8, 64], 0.5, False, 3),
]


@pytest.mark.parametrize("tag, matmul_type, lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size", test_data_infer)
def test_bsmatmul_infer(tag, matmul_type, lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size):
    print("Running test_bsmatmul_infer() with tag: {}, matmul_type:{}, lhs_dims:{}, rhs_dims:{}, block_size:{}, sparsity_level:{}, transpose_rhs:{}, inner_group_size {}"
          .format(tag, "DENSE_LHS_SPARSE_RHS_DENSE_OUT" if matmul_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'] else "DENSE_LHS_DENSE_RHS_SPARSE_OUT",
                  lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size))

    memory_cycle_ratio = 1.0
    ipuOutput, goldOutput = sparse_mm_infer(matmul_type,
                                            lhs_dims,
                                            rhs_dims,
                                            block_size,
                                            sparsity_level,
                                            transpose_rhs,
                                            memory_cycle_ratio,
                                            inner_group_size)
    rtol = 1e-3
    atol = 1e-3
    np.testing.assert_allclose(ipuOutput, goldOutput, rtol=rtol, atol=atol)

#
# TRAINING TEST
#

# test_data_train tuple --> (matMulType, lhs_dims, rhs_dims, block_size, sparsity, transpose_rhs, inner_group_size)
test_data_train = [
    # 2D
    ("tag_tr_0", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_1", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, False, 1),
    ("tag_tr_2", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [32, 32], [32, 32], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_3", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [64, 64], [64, 256], [8, 8, 8], 0.9, False, 1),
    ("tag_tr_4", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [128, 128], [128, 128], [8, 8, 8], 0.2, False, 1),

    # 3D,
    ("tag_tr_5", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 16, 16], [2, 16, 16], [8, 8, 8], 0.1, False, 1),
    ("tag_tr_6", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [5, 16, 16], [5, 16, 16], [8, 8, 8], 0.3, False, 1),
    ("tag_tr_7", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [7, 32, 32], [7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_8", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [11, 64, 64], [11, 64, 64], [8, 8, 8], 0.6, False, 1),
    ("tag_tr_9", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [12, 128, 128], [12, 128, 128], [8, 8, 8], 0.8, False, 1),

    # 4D,
    ("tag_tr_10", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 8, 8], [1, 1, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_11", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 16, 16], [1, 1, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_12", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 32, 32], [1, 1, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_13", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 64, 128], [1, 1, 128, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_14", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 1, 128, 64], [1, 1, 64, 128], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_15", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 2, 8, 8], [1, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_16", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 5, 16, 16], [1, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_17", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 7, 32, 32], [1, 7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_18", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 11, 64, 128], [1, 11, 128, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_19", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 12, 128, 64], [1, 12, 64, 128], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_20", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_21", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [5, 5, 16, 16], [5, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_22", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [13, 7, 32, 32], [13, 7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_23", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [3, 11, 64, 128], [3, 11, 128, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_24", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 12, 128, 64], [1, 12, 64, 128], [8, 8, 8], 0.5, False, 1),

    # 2D, lhs has to be square to take inverse
    ("tag_tr_25", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_26", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, False, 1),
    ("tag_tr_27", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [32, 32], [32, 32], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_28", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [64, 64], [64, 256], [8, 8, 8], 0.9, False, 1),
    ("tag_tr_29", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [128, 128], [128, 128], [8, 8, 8], 0.2, False, 1),

    # 3D, lhs has to be square to take
    ("tag_tr_30", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [2, 8, 8], [2, 8, 8], [8, 8, 8], 0.1, False, 1),
    ("tag_tr_31", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [5, 16, 16], [5, 16, 16], [8, 8, 8], 0.3, False, 1),
    ("tag_tr_32", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [7, 32, 32], [7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_33", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [11, 64, 64], [11, 64, 64], [8, 8, 8], 0.6, False, 1),
    ("tag_tr_34", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [12, 128, 128], [12, 128, 128], [8, 8, 8], 0.3, False, 1),

    # 4D, lhs has to be square to take
    ("tag_tr_35", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 8, 8], [1, 1, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_36", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 16, 16], [1, 1, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_37", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 32, 32], [1, 1, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_38", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 64, 64], [1, 1, 64, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_39", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 1, 128, 128], [1, 1, 128, 128], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_40", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 2, 8, 8], [1, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_41", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 5, 16, 16], [1, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_42", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 7, 32, 32], [1, 7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_43", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 11, 64, 64], [1, 11, 64, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_44", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 12, 128, 128], [1, 12, 128, 128], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_45", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_46", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [5, 5, 16, 16], [5, 5, 16, 16], [8, 8, 8], 0.8, False, 1),
    ("tag_tr_47", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [13, 7, 32, 32], [13, 7, 32, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_48", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [3, 11, 64, 64], [3, 11, 64, 256], [8, 8, 8], 0.5, False, 1),
    ("tag_tr_49", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [1, 12, 128, 128], [1, 12, 128, 1024], [8, 8, 8], 0.5, False, 1),

    # For transpose_rhs True case, last 2 dimensions of block_size must be 8
    ("tag_tr_50", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [8, 8], [8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_tr_51", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [16, 16], [16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_tr_52", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [32, 32], [32, 32], [8, 8, 8], 0.8, True, 1),
    ("tag_tr_53", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [64, 64], [256, 64], [8, 8, 8], 0.9, True, 1),
    ("tag_tr_54", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [128, 128], [128, 128], [8, 8, 8], 0.2, True, 1),
    ("tag_tr_55", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [2, 8, 8], [2, 8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_tr_56", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [3, 16, 16], [3, 16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_tr_57", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [7, 128, 128], [7, 128, 128], [8, 8, 8], 0.2, True, 1),
    ("tag_tr_58", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [3, 5, 8, 8], [3, 5, 8, 8], [8, 8, 8], 0.5, True, 1),
    ("tag_tr_59", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [1, 6, 16, 16], [1, 6, 16, 16], [8, 8, 8], 0.1, True, 1),
    ("tag_tr_60", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [4, 4, 32, 32], [4, 4, 32, 32], [8, 8, 8], 0.8, True, 1),

    # 3D, inner group size > 1
    ("tag_tr_61", g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'], [12, 128, 128], [12, 128, 128], [8, 8, 8], 0.8, False, 3),
    ("tag_tr_62", g_sparseMatMulTypeLookup['DENSE_LHS_DENSE_RHS_SPARSE_OUT'], [12, 128, 128], [12, 128, 128], [8, 8, 8], 0.1, False, 4),
]


@pytest.mark.parametrize("tag, matmul_type, lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size", test_data_train)
def test_bsmatmul_train(tag, matmul_type, lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size):
    print("Running test_bsmatmul_train() with tag: {}, matmul_type:{}, lhs_dims:{}, rhs_dims:{}, block_size:{}, sparsity_level:{}, transpose_rhs:{}, inner_group_size {}"
          .format(tag, "DENSE_LHS_SPARSE_RHS_DENSE_OUT" if matmul_type == g_sparseMatMulTypeLookup['DENSE_LHS_SPARSE_RHS_DENSE_OUT'] else "DENSE_LHS_DENSE_RHS_SPARSE_OUT",
                  lhs_dims, rhs_dims, block_size, sparsity_level, transpose_rhs, inner_group_size))
    memory_cycle_ratio = 1.0

    sparse_mm_train(matmul_type,
                    lhs_dims,
                    rhs_dims,
                    block_size,
                    sparsity_level,
                    transpose_rhs,
                    memory_cycle_ratio,
                    inner_group_size)

# test_data_softmax tuple --> (dims, block_size, sparsity, inner_group_size)
test_data_softmax = [
    # 2D
    ("tag_sm_0", [8, 8], [8, 8], 0.0, 1),
    # 3D
    ("tag_sm_1", [16, 16], [8, 8], 0.4, 1),
    # 4D
    ("tag_sm_2", [2, 2, 16, 16], [8, 8], 0.3, 1),

    # 5D, inner group size = 1
    ("tag_sm_3", [2, 3, 2, 16, 16], [8, 8], 0.1, 1),

    # 5D, inner group size > 1
    ("tag_sm_4", [2, 3, 2, 16, 16], [8, 8], 0.1, 0),
    ("tag_sm_5", [2, 3, 2, 16, 16], [8, 8], 0.1, 6),
]


@pytest.mark.parametrize("tag, dims, block_size, sparsity_level, inner_group_size", test_data_softmax)
def test_bs_softmax(tag, dims, block_size, sparsity_level, inner_group_size):
    print("Running test_bs_softmax() with tag: {}, dims:{}, block_size:{}, sparsity_level:{}, inner_group_size {}"
          .format(tag, dims, block_size, sparsity_level, inner_group_size))
    ipu_output, gold_output = sparse_softmax(dims,
                                             block_size,
                                             sparsity_level,
                                             inner_group_size)
    np.testing.assert_allclose(ipu_output, gold_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    ipu_output, gold_output = sparse_softmax([2, 2, 16, 16],
                                             [8, 8],
                                             0.3,
                                             1)
    np.testing.assert_allclose(ipu_output, gold_output, rtol=1e-2, atol=1e-2)
