# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pdb
import os
import numpy as np
from scipy import stats
import ctypes
import popart
from sparse_attention_utils import Heads, Convert

so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

# Sequence Parameters
n_batch = 1
n_heads = 4
n_hidden = 256
sequence_length = 1024
blocksize = [16, 16, 16]

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
# The main input to attention is a tensor activation with dims [B, S, H]
placeholder_info = popart.TensorInfo("FLOAT16", [n_batch, sequence_length, n_hidden])
input_x = builder.addInputTensor(placeholder_info, 'input_x')
print('Input placeholder has shape: ', builder.getTensorShape(input_x))

# Weights for the query
scale = np.sqrt(1/(n_hidden))
query_weights = stats.truncnorm(-1, 1, loc=0.0, scale=scale).rvs(size=(n_hidden, n_hidden))
query_weights = builder.addInitializedInputTensor(np.float16(query_weights), 'query_weights')
print('Query weights shape: ', builder.getTensorShape(query_weights))

# Weights for the key
scale = np.sqrt(1/(n_hidden))
key_weights = stats.truncnorm(-1, 1, loc=0.0, scale=scale).rvs(size=(n_hidden, n_hidden))
key_weights = builder.addInitializedInputTensor(np.float16(key_weights), 'key_weights')
print('Key weights shape: ', builder.getTensorShape(key_weights))

# Weights for the value
scale = np.sqrt(1/(n_hidden))
value_weights = stats.truncnorm(-1, 1, loc=0.0, scale=scale).rvs(size=(n_hidden, n_hidden))
value_weights = builder.addInitializedInputTensor(np.float16(value_weights), 'value_weights')
print('Value weights have shape: ', builder.getTensorShape(value_weights))

# Attention projection weights
scale = np.sqrt(1/n_hidden)
projection_weights = stats.truncnorm(-1, 1, loc=0.0, scale=scale).rvs(size=(n_hidden, n_hidden))
projection_weights = builder.addInitializedInputTensor(np.float16(projection_weights), 'projection_weights')
print('Projection_weights have shape: ', builder.getTensorShape(projection_weights))

# Norm parameters
gamma = builder.addInitializedInputTensor(np.ones([n_hidden], dtype=np.float16), 'attention/gamma')
print('attention/gamma shape: ', builder.getTensorShape(gamma))

beta = builder.aiOnnx.constant(np.float16(np.zeros([n_hidden])), "FLOAT16")
print('attention/beta shape: ', builder.getTensorShape(beta))


# Some helper functions for dealing with attention heads
def extract_heads(tensor):
    # Desired shape after reshape
    assert n_hidden % n_heads == 0
    comb_shape = [n_batch, sequence_length, n_heads, n_hidden//n_heads]
    tensor = builder.reshape_const(builder.aiOnnx, [tensor], comb_shape)
    return tensor


def transpose_heads(tensor, is_keys=False):
    perm = [0, 2, 1, 3] if not is_keys else [0, 2, 3, 1]
    return builder.aiOnnx.transpose([tensor], perm=perm)

# ATTENTION BLOCK
# 1. QK
queries = builder.aiOnnx.matmul([input_x, query_weights], "Q")
queries = extract_heads(queries)
queries = transpose_heads(queries)

keys = builder.aiOnnx.matmul([input_x, key_weights], "K")
keys = extract_heads(keys)
keys = transpose_heads(keys, is_keys=True)

# 2. MATMUL
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
                           "transpose_rhs": False,
                           "memory_cycle_ratio": 0.6,
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

# Construct the values based on the attention softmax probabilities
values = builder.aiOnnx.matmul([input_x, value_weights], "V")
values = extract_heads(values)
values = transpose_heads(values)

# Multiply the sparse probabilities with the dense values to produce a dense output
# This requires the operation (values.T @ probs.T).T
# To transpose the probs we set the transpose_rhs flag to True
# This means the matmul shapes are [n_hidden/n_heads, (seq] x [seq), seq], in short
matmul_dims = [n_hidden//n_heads, sequence_length, sequence_length]
values_t = builder.aiOnnx.transpose([values], perm=[0, 1, 3, 2])
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
                                     "memory_cycle_ratio": 0.5,
                                     "in_type": "float16",
                                     "out_type": "float16",
                                     "pp_type": "float16"
                                    })[0]
attendedValues = builder.aiOnnx.transpose([attendedValues_t], perm = [0, 1, 3, 2])
# Everything following this line is the same as in regular dense attention.

# 3. PROJECTION OF ATTENDED VALUES
# attendedValues are dense and can be projected as usual
# Transpose [B, H, S, n_h] -> [B, S, H, n_h]
attendedValues = builder.aiOnnx.transpose([attendedValues], perm = [0, 2, 1, 3])

# Reshape [B, S, H, n_h] -> [B, S, hidden]
attendedValues = builder.reshape_const(builder.aiOnnx, [attendedValues], [n_batch, sequence_length, n_hidden])
x = builder.aiOnnx.matmul([attendedValues, projection_weights], "attendedValues@projection_weights")

# 4. DROPOUT, SKIP CONNECT, NORM
dropout_prob = 0.1
x = builder.aiOnnx.dropout([x], 1, dropout_prob)[0]
x = builder.aiOnnx.add([input_x, x])

# Group normalization on 2D tensor
x = builder.reshape_const(builder.aiOnnx, [x], [n_batch*sequence_length, n_hidden])
x = builder.aiGraphcore.groupnormalization([x, gamma, beta], 1, 1e-6)[0]
x = builder.reshape_const(builder.aiOnnx, [x], [n_batch, sequence_length, n_hidden])

# LOSS
# something arbitrary just so the graph compiles and generates grads
loss = builder.aiGraphcore.l1loss([x], 1.0)

# Describe how to run the model
anchor_desc = {loss: popart.AnchorReturnType("ALL"),
               probs: popart.AnchorReturnType("ALL"),
               dprobs: popart.AnchorReturnType("ALL")}
dataFlow = popart.DataFlow(1, anchor_desc)
session = popart.TrainingSession(fnModel = builder.getModelProto(),
                                 loss = loss,
                                 deviceInfo = popart.DeviceManager().createIpuModelDevice({}),
                                 optimizer = popart.ConstSGD(0.01),
                                 dataFlow = dataFlow)

# COMPILE
session.prepareDevice()

# RUN A STEP (FWD, BWD, UPDATE)
session.weightsFromHost()
anchors = session.initAnchorArrays()
feed_dict = {
    input_x: np.float16(np.random.randn(n_batch, sequence_length, n_hidden))
}
stepio = popart.PyStepIO(feed_dict, anchors)
session.run(stepio)

# INSPECT
# Dense tensors can be inspected directly. Here we show how to inspect the
# block-sparse gradient of the probabilities by first converting it to a
# numpy array
gradBlocks = anchors[dprobs]
numpyShape = [n_batch, n_heads, sequence_length, sequence_length]
gradArray = Convert.to_np_array(numpyShape, gradBlocks, sparsity,
                                bsr_rhs_lengths_per_2d_plane, blocksize[-2:])
print("Mean of max prob grad on each row: ", np.mean(np.max(gradArray, axis=-1)))
print("Done")
