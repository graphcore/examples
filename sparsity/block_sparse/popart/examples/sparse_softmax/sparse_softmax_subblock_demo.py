# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import scipy as sp
from scipy import sparse
import os
import ctypes
import popart

import sys
sys.path.append('/path/to/application/app/folder')

so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

"""
This example performs block sparse softmax with causal masking on a seuqnece of
256 tokens, which is broken up into two windows of length 128 in the attention pattern.
Furthermore, the entire matrix is broken up into blocks of 16 by 16 for blocksparse processing.
When fed into popart, the blocks should be flattened into a linear vector [80, 16, 16] -> [80, 256].

For simplicity the input logits set equal to global attention mask (1s for elements which are attended to and 0s elsewhere)
A given row will have a uniform distribution over all attended values.

To produce a loss take the negative log likelihood of the expected tokens (diagonal matrix) under this uniform distribution.
Both the forward and backward pass are verified analytically.

For example:
The fourth row of logits looks like this [1, 1, 1, 1, 0 ... 0]
The corresponding softmax output is [0.25, 0.25, 0.25, 0.25, 0 ... 0]
The upstream grad is [0, 0, 0, -4, 0 ... 0] (NLL grad)
And the expected softmax grad is [0.25, 0.25, 0.25, -0.75, 0 ... 0]
"""


# INPUT DATA
n_windows = 2
n_sequence = 256
window_size = n_sequence//n_windows
blocksize = [16, 16]

# Simulate some uniform logits (this will be explained in a demo later)
auto_mask = sp.sparse.tril(np.ones([window_size, window_size]), k = 0)
summary_mask = sp.sparse.lil_matrix((window_size, window_size))
summary_mask[:, window_size-blocksize[0]:] = 1
global_mask = sp.sparse.kron(sp.sparse.tril(np.ones([n_windows, n_windows]), k=-1), summary_mask)
global_mask = (global_mask + sp.sparse.kron(sp.sparse.eye(n_windows), auto_mask)).sign()

# Get the block sparse format
bsr = sp.sparse.bsr_matrix(global_mask, blocksize=blocksize)
bsr.eliminate_zeros()  # need to call this to eliminate blocks of all zeros

# The dense blocks
blocks = sp.float32(bsr.data)

# Dense mask for each active block
mask_data = np.array([[[1]]]*len(bsr.indices))
active_mask = sp.sparse.bsr_matrix((mask_data, bsr.indices, bsr.indptr)).toarray()
# np.savetxt("active_mask.txt", active_mask, delimiter='', fmt="%i")
active_mask = sp.int64(active_mask.flatten())
print("Done creating input data")

# #### MODEL CREATION ####
builder = popart.Builder()
# Reshape the blocks to the desired format
blocks = sp.reshape(blocks, [blocks.shape[0], -1])
blocks = np.array(list(blocks), dtype=sp.float32)
logits = builder.addInitializedInputTensor(blocks)

probs = builder.customOp(opName = "BsSoftmax",
                         opVersion = 1,
                         domain = "ai.graphcore",
                         inputs = [logits],
                         attributes = {
                          "matrixDims": [1, n_sequence, n_sequence],  # up to 4D tensors are supported
                          "blockSize": blocksize,
                          "sparsity": list(active_mask),
                          "groupSizes": [sum(active_mask)],
                          "subBlockMaskPerGroup": "ZeroUpperTriangle",  # None, ZeroUpperTriangle, ZeroLowerTriangle
                         })[0]
dlogits = popart.reservedGradientPrefix() + logits  # the gradient tensor's name
upstream_grad = popart.reservedGradientPrefix() + probs  # the gradient tensor's name

# Compute negative log-likelihood with respect to diagonal tokens (identity distribution)
expected_tokens = np.zeros([active_mask.shape[0], blocksize[0], blocksize[0]])
expected_tokens[::blocksize[0]+1, :, :] = np.eye(blocksize[0])
expected_tokens = expected_tokens.reshape([expected_tokens.shape[0], -1])
expected_tokens = expected_tokens[np.where(active_mask)]
expected_tokens = -sp.float32(np.array(list(expected_tokens)))  # negative sign for negative logprob
expected_tokens = builder.aiOnnx.constant(expected_tokens, 'expected_tokens')

pbias = builder.aiOnnx.constant(np.zeros([1, blocks.shape[-1]], dtype=np.float32)+1e-6, 'pbias')
biased_probs = builder.aiOnnx.add([probs, pbias])
logprobs = builder.aiOnnx.log([biased_probs])

out = builder.aiOnnx.mul([logprobs, expected_tokens])
loss = builder.aiGraphcore.l1loss([out], 1.0)

# Describe how to run the model
anchor_desc = {probs: popart.AnchorReturnType("ALL"), dlogits: popart.AnchorReturnType("ALL"), upstream_grad: popart.AnchorReturnType("ALL")}
dataFlow = popart.DataFlow(1, anchor_desc)

session = popart.TrainingSession(fnModel = builder.getModelProto(),
                                 loss = loss,
                                 deviceInfo = popart.DeviceManager().acquireAvailableDevice(1),
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

# Reconstruct scipy
# forward activations
result = anchors[probs]
result = result.reshape([result.shape[0], *blocksize])
result = sp.sparse.bsr_matrix((result, bsr.indices, bsr.indptr))

# gradients
dresult = anchors[dlogits]
dresult = dresult.reshape([dresult.shape[0], *blocksize])
dresult = sp.sparse.bsr_matrix((dresult, bsr.indices, bsr.indptr))

dprobs = anchors[upstream_grad]
dprobs = dprobs.reshape([dprobs.shape[0], *blocksize])
dprobs = sp.sparse.bsr_matrix((dprobs, bsr.indices, bsr.indptr))

for i in range(n_sequence):
    row = result.getrow(i)
    # Check that the row sums to approximately 1
    tol = 1e-5
    assert abs(np.sum(row) - 1) < tol, f"Error in a row {i}. Probabilities do not sum to 1 with tolerance {tol}"

    # Verify each element is uniform prob if it's mask is 1 and 0 if mask is 0
    mask = global_mask.getrow(i)
    uniform_prob = 1/np.sum(mask)

    # To check for inclusion the DOK sparse format is best
    mask = mask.todok()
    row = row.todok()

    for position in row.keys():
        value = row[position]
        # The mask is accurate at the subblock level
        if position in mask:
            tol = 1e-6
            assert abs(value - uniform_prob) < tol, f"Error in a row {i}. Probability {pos[1]} is {value} but should be {uniform_prob} with tolerance {tol}"
        # Other positions should be verifyably near 0
        else:
            tol = 1e-20
            assert abs(value) < tol, f"Error in a row {i}. Probability {pos[1]} is {value} but should be 0 with tolerance {tol}"
print("FWD PASSED.")


result = result.tocsc()  # efficient row slicing
dresult = dresult.tocsr()  # efficient row slicing
dprobs = dprobs.tocsr()  # efficient row slicing
# these are matrices, not arrays. Multiplication * is actually matmul

# Analytical softmax grad (uses upstream_grad from popart i.e. dprobs)
expected_grads = result.multiply(dprobs.multiply(sp.sparse.eye(n_sequence).tocsr()) - np.sum(result.multiply(dprobs), axis=1))

# For each row check that the analytic results match those from popart
for i in range(n_sequence):
    erow = expected_grads.getrow(i).toarray().squeeze()
    drow = dresult.getrow(i).toarray().squeeze()
    tol = 1e-6
    try:
        assert np.allclose(erow, drow, tol), f"Error in a row {i}. Gradients of softmax are not within {tol} tolerance."
    except:
        print(f"\n Difference for row {i}: \n", np.around(erow-drow, 2), "\n Argmax of difference:\n", np.argmax(np.abs(erow-drow)))
        raise

print("BWD PASSED.")
