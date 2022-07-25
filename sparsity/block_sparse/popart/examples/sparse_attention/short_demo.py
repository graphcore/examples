# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pdb
import os
import numpy as np
from scipy import stats
import ctypes
import popart
from sparse_attention_utils import Heads, Convert


so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

np.random.seed(0)

# Sequence Parameters
n_batch = 1
n_heads = 4
n_hidden = 256
sequence_length = 1024
blocksize = [16, 16, 16]  # 3d b/c specifying all dimension of a matmul (m, n) x (n, k) -> (m, k)

# Attention parameters
window_size = 128
n_summary_blocks = 1
n_windows = sequence_length//window_size
n_block_gram = 4

# ATTENTION PATTERN DEFINITION
# use the predefined sparse attention heads
assert sequence_length % window_size == 0, "Sequence length must be divisible by window length"
head1 = Heads.causal_windows_with_summaries(window_size, n_windows, n_summary_blocks, blocksize[-2:])
head2 = Heads.causal_block_gram(sequence_length, n_block_gram, blocksize[-2:])

# Use each head twice (multi-head attention with 4 heads)
blocks, sparsity, bsr_rhs_lengths_per_2d_plane = \
        Heads.concatenate_heads([head1, head2], [2, 2])

# BUILD THE MODEL IN POPART
builder = popart.Builder()

# INPUTS
# in this short demo the queries, keys and values are considered direct inputs
queries = builder.addInitializedInputTensor(np.float16(np.random.randn(n_batch, n_heads, sequence_length, n_hidden//n_heads)), 'queries')
keys = builder.addInitializedInputTensor(np.float16(np.random.randn(n_batch, n_heads, n_hidden//n_heads, sequence_length)), 'keys')
values_t = builder.addInitializedInputTensor(np.float16(np.random.randn(n_batch, n_heads, n_hidden//n_heads, sequence_length)), 'values_t')

# Multiply the queries and keys in a sparse way
logits = builder.customOp(opName = "BSMatMul",
                          opVersion = 1,
                          domain = "ai.graphcore",
                          inputs = [queries, keys],
                          attributes = {
                           "bsr_rhs_lengths_per_2d_plane": bsr_rhs_lengths_per_2d_plane,
                           "matrix_dims": [sequence_length, n_hidden//n_heads, sequence_length],
                           "block_size": blocksize,
                           "sparsity_mask": sparsity,
                           "bsmatmul_type": 1,  # Dense @ Dense -> Sparse out
                           "transpose_rhs": False,  # see the project below for an instance of when True
                           "memory_cycle_ratio": 0.6,  # IF YOU ARE NOT GRAPHCORE -- DO NOT TOUCH
                           "in_type": "float16",
                           "out_type": "float16",
                           "pp_type": "float16"
                          })[0]

# The gradients of the logits will have the following name
# we'll show how to inspect them at the end of the demo
dlogits = popart.reservedGradientPrefix() + logits

# Self attention scales the logits by head size to control variance.
variance_scale = np.array([np.sqrt(1.0/(n_hidden/n_heads))])
variance_scale = builder.aiOnnx.constant(np.float16(variance_scale), "variance_scale")
scaled_logits = builder.aiOnnx.mul([logits, variance_scale], "logits_scaling")

# Next apply the special BsSoftmax to convert the logits into probabilities
# this will treat zero-blocks as having 0 probability. There's also on option
# to apply causal masking (per group), which we use
probs = builder.customOp(opName = "BsSoftmax",
                         opVersion = 1,
                         domain = "ai.graphcore",
                         inputs = [scaled_logits],
                         attributes = {
                          "matrixDims": [n_batch, n_heads, sequence_length, sequence_length],
                          "blockSize": blocksize,
                          "sparsity": sparsity,
                          "groupSizes": bsr_rhs_lengths_per_2d_plane,
                          "subBlockMaskPerGroup": "ZeroUpperTriangle " * len(bsr_rhs_lengths_per_2d_plane)
                         })[0]

# The gradients wrt probabilities will have the name (can be visualized in the same way
# as dlogits
dprobs = popart.reservedGradientPrefix() + probs

# Multiply the sparse probabilities with the dense values to produce a dense output
# This requires the operation (values.T @ probs.T).T
# To transpose the probs we set the transpose_rhs flag to True
# This means the matmul shapes are [n_hidden/n_heads, (seq] x [seq), seq], in short
matmul_dims = [n_hidden//n_heads, sequence_length, sequence_length]
attendedValues_t = builder.customOp(opName = "BSMatMul",
                                    opVersion = 1,
                                    domain = "ai.graphcore",
                                    inputs = [values_t, probs],
                                    attributes = {
                                     "bsr_rhs_lengths_per_2d_plane": bsr_rhs_lengths_per_2d_plane,
                                     "matrix_dims": matmul_dims,
                                     "block_size": blocksize,
                                     "sparsity_mask": sparsity,
                                     "bsmatmul_type": 0,  # Dense @ Sparse -> Dense Out
                                     "transpose_rhs": True,  # this will transpose the probabilities matrix for (values.T @ probs.T)
                                     "memory_cycle_ratio": 0.5,  # IF YOU ARE NOT GRAPHCORE -- DO NOT TOUCH
                                     "in_type": "float16",
                                     "out_type": "float16",
                                     "pp_type": "float16"
                                    })[0]
attendedValues = builder.aiOnnx.transpose([attendedValues_t], perm = [0, 1, 3, 2])
# We've now arrived at the attended values which concludes the forward pass

# LOSS
# something arbitrary just so the graph compiles and generates grads
loss = builder.aiGraphcore.l1loss([attendedValues], 1.0)

# SESSION DETAILS
anchor_desc = {loss: popart.AnchorReturnType("ALL"),
               logits: popart.AnchorReturnType("ALL"),
               dlogits: popart.AnchorReturnType("ALL"),
               probs: popart.AnchorReturnType("ALL"),
               dprobs: popart.AnchorReturnType("ALL")}
dataFlow = popart.DataFlow(1, anchor_desc)
session = popart.TrainingSession(fnModel = builder.getModelProto(),
                                 loss = loss,
                                 deviceInfo = popart.DeviceManager().acquireAvailableDevice(1),
                                 optimizer = popart.ConstSGD(0.01),
                                 dataFlow = dataFlow)

# COMPILE
session.prepareDevice()

# RUN A STEP (FWD, BWD, UPDATE)
session.weightsFromHost()
anchors = session.initAnchorArrays()
session.run(popart.PyStepIO({}, anchors))

# INSPECT
# we can use the utilities to convert the sparse probabilities back
# into a dense numpy array for inspection
probBlocks = anchors[probs]
numpyShape = [n_batch, n_heads, sequence_length, sequence_length]
probArray = Convert.to_np_array(numpyShape, probBlocks, sparsity,
                                bsr_rhs_lengths_per_2d_plane, blocksize[-2:])
print(f"Probability mean {np.mean(probArray)}, std {np.std(probArray)}")

# The same process can be used to inspect the grads
dlogitsBlocks = anchors[dlogits]
numpyShape = [n_batch, n_heads, sequence_length, sequence_length]
dlogitsArray = Convert.to_np_array(numpyShape, dlogitsBlocks, sparsity,
                                   bsr_rhs_lengths_per_2d_plane, blocksize[-2:])
print(f"Logits grad mean {np.mean(dlogitsArray)}, std {np.std(dlogitsArray)}")
