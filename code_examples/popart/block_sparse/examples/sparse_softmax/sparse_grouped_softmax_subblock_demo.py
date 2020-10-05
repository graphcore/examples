# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import scipy as sp
from scipy import sparse
import os
import ctypes
import popart
so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

"""
This example presents the extension of the test_block_sparse_softmax.py example to multiple inputs
In NLP the attention matrix will have dims [batch_size, heads, n_sequence, n_sequence]
Where each batch item and head may have a different sparsity pattern.

In this example there are two attention heads each with a different mask (and number of active blocks)
"""

# INPUT DATA
n_windows = 2
n_sequence = 256
window_size = n_sequence//n_windows
blocksize = [16, 16]


def type1_mask(window_size, n_windows, blocksize_x):
    # Add the sparsity for the first attention head (autoregressive windows)
    auto_mask = sp.sparse.tril(np.ones([window_size, window_size]), k = 0)
    summary_mask = sp.sparse.lil_matrix((window_size, window_size))
    summary_mask[:, window_size-blocksize_x:] = 1
    global_mask = sp.sparse.kron(sp.sparse.tril(np.ones([n_windows, n_windows]), k = -1), summary_mask)
    global_mask = (global_mask + sp.sparse.kron(sp.sparse.eye(n_windows), auto_mask)).sign()
    return global_mask


def type2_mask(n_sequence, blocksize_x):
    # Local mask attends to local block plus one backward (a bit like reformer)
    # Autoregressive block on diagonal
    A = np.expand_dims(sp.sparse.tril(np.ones((blocksize_x, blocksize_x)), 0).toarray(), 0)
    A = A.repeat(n_sequence//blocksize_x, axis = 0)
    mask = sp.sparse.block_diag(A)

    # Add full blocks on the -1 diagonal
    C = sp.sparse.dia_matrix((np.ones((1, n_sequence)), [-1]), shape=[n_sequence//blocksize_x]*2)
    mask += sp.sparse.kron(C, np.ones((blocksize_x, blocksize_x)))
    return mask


def mask_to_blocks(global_mask, blocksize):
    # Get the block sparse format
    bsr = sp.sparse.bsr_matrix(global_mask, blocksize = blocksize)
    bsr.eliminate_zeros()  # need to call this to eliminate blocks of all zeros

    # The dense blocks
    blocks = np.reshape(bsr.data, [bsr.data.shape[0], -1])
    blocks = sp.float32(list(blocks))

    # Dense mask for each active block
    mask_data = np.array([[[1]]]*len(bsr.indices))
    active_mask = sp.sparse.bsr_matrix((mask_data, bsr.indices, bsr.indptr)).toarray()
    active_mask = list(active_mask.flatten())
    return blocks, active_mask

# Get the two attention patterns
head1_blocks, head1_sparsity = mask_to_blocks(type1_mask(window_size, n_windows, blocksize[0]), blocksize)
head2_blocks, head2_sparsity = mask_to_blocks(type2_mask(n_sequence, blocksize[0]), blocksize)


def concat(h1, h2):
    out = np.concatenate((h1, h2), 0)
    out = np.tile(out, [2, 1])
    return out

# Build a matrix which is [2, 2, 256, 256] (B, H, S, S)
matrix_dims = [2, 2, n_sequence, n_sequence]
input_blocks = concat(head1_blocks, head2_blocks)
sparsity = np.tile([*head1_sparsity, *head2_sparsity], 2)
# There are 4 groups in total (B*H)
group_sizes = np.tile([len(head1_blocks), len(head2_blocks)], 2)
# note that group_sizes are equal to [80, 31, 80, 31]

# #### MODEL CREATION ####
builder = popart.Builder()
logits = np.array(list(input_blocks), dtype = sp.float32)
logits = builder.addInitializedInputTensor(logits, "logits")

probs = builder.customOp(opName = "BsSoftmax",
                         opVersion = 1,
                         domain = "ai.graphcore",
                         inputs = [logits],
                         attributes = {
                          "matrixDims": matrix_dims,
                          "blockSize": blocksize,
                          "sparsity": sparsity.tolist(),
                          "groupSizes": group_sizes.tolist(),
                          "subBlockMaskPerGroup": "[ZeroUpperTriangle, ZeroUpperTriangle, ZeroUpperTriangle, ZeroUpperTriangle]"
                         })[0]
dlogits = popart.reservedGradientPrefix() + logits  # the gradient tensor's name
upstream_grad = popart.reservedGradientPrefix() + probs  # the gradient tensor's name

# Make some blocks to regress agains just so there are gradients
expected_tokens = np.zeros_like(input_blocks) + np.eye(16).flatten()
expected_tokens = -sp.float32(np.array(list(expected_tokens)))  # negative sign for negative logprob
expected_tokens = builder.aiOnnx.constant(expected_tokens, 'expected_tokens')

pbias = builder.aiOnnx.constant(np.zeros([1, input_blocks.shape[-1]], dtype=np.float32)+1e-6, 'pbias')
biased_probs = builder.aiOnnx.add([probs, pbias])
logprobs = builder.aiOnnx.log([biased_probs])

out = builder.aiOnnx.mul([logprobs, expected_tokens])
loss = builder.aiGraphcore.l1loss([out], 1.0)

# Describe how to run the model
anchor_desc = {probs: popart.AnchorReturnType("ALL"), dlogits: popart.AnchorReturnType("ALL"), upstream_grad: popart.AnchorReturnType("ALL")}
dataFlow = popart.DataFlow(1, anchor_desc)

session = popart.TrainingSession(fnModel = builder.getModelProto(),
                                 loss = loss,
                                 deviceInfo = popart.DeviceManager().createIpuModelDevice({}),
                                 optimizer = popart.ConstSGD(0.01),
                                 dataFlow = dataFlow)

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# TRAINING
session.weightsFromHost()
stepio = popart.PyStepIO({}, anchors)
session.run(stepio)
print("Mean max grad of each row: ", np.mean(np.max(anchors[dlogits].reshape([-1, *blocksize]), axis = -1)))
