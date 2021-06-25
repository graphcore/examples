# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import popart
import numpy as np


def add_inputs(model):
    # make inputs directly accessible to the model
    model.inputs = {}
    model.labels = {}
    config = model.config

    # Sequence inputs
    # Input mask contains a seuqence index for each token in the pack
    sequence_shape = [config.micro_batch_size * config.sequence_length]
    sequence_info = popart.TensorInfo("UINT32", sequence_shape)
    model.inputs["input_ids"] = model.builder.addInputTensor(sequence_info, "input_ids")
    model.inputs["input_mask"] = model.builder.addInputTensor(sequence_info, "input_mask")
    model.inputs["position_ids"] = model.builder.addInputTensor(sequence_info, "position_ids")
    model.inputs["segment_ids"] = model.builder.addInputTensor(sequence_info, "segment_ids")

    # MLM token ids and their respective positions.
    # The masked_lm_weights contain the index of the sequence in the pack to which a token belongs
    mlm_shape = [config.micro_batch_size, config.max_lm_predictions]
    mlm_info = popart.TensorInfo("UINT32", mlm_shape)
    model.inputs["masked_lm_positions"] = model.builder.addInputTensor(mlm_info, "masked_lm_positions")
    model.labels["masked_lm_ids"] = model.builder.addInputTensor(mlm_info, "masked_lm_ids")
    model.inputs["masked_lm_weights"] = model.builder.addInputTensor(mlm_info, "masked_lm_weights")

    # NSP (there are multiple [CLS] tokens per pack)
    nsp_shape = [config.micro_batch_size, config.max_sequences_per_pack]
    nsp_info = popart.TensorInfo("UINT32", nsp_shape)
    model.inputs["nsp_positions"] = model.builder.addInputTensor(nsp_info, "nsp_positions")
    model.labels["nsp_labels"] = model.builder.addInputTensor(nsp_info, "nsp_labels")
    model.labels["nsp_weights"] = model.builder.addInputTensor(nsp_info, "nsp_weights")

    # The shapes for the constructed inputs and labels
    # (in order of appearance in the dataset). Required for compatibility with legacy code
    input_tensor_shapes = [sequence_shape]*4 + [mlm_shape]*3 + [nsp_shape]*3
    input_tensor_names = ["input_ids", "input_mask", "position_ids", "segment_ids"]
    input_tensor_names += ["masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]
    input_tensor_names += ["nsp_positions", "nsp_labels", "nsp_weights"]
    return [(name, shape) for name, shape in zip(input_tensor_names, input_tensor_shapes)]


def logits_graph(model):
    # Glue code for compatibility with non-packing version and naming convention
    indices = model.inputs["input_ids"]
    positions = model.inputs["position_ids"]
    segments = model.inputs["segment_ids"]
    masks = [model.inputs["input_mask"]]
    list_of_logits = model.build_graph(indices, positions, segments, masks)
    return list_of_logits


def attention_mask(model):
    config = model.config
    input_mask = model.inputs["input_mask"]
    # calculate the block-diagonal mask for attention in packed sequence bert
    mask = model.builder.reshape_const(model.builder.aiOnnx, [input_mask], [config.micro_batch_size, config.sequence_length, 1])
    maskT = model.builder.aiOnnx.transpose([mask], perm=[0, 2, 1])

    # Mask between different sub-sequences inside a packed sequences
    final_mask = model.builder.aiOnnx.equal([mask, maskT])  # [B, S, 1]==[B, 1, S] -> [B, S, S]
    final_mask = model.builder.aiOnnx.cast([final_mask], config.popart_dtype)

    # Convert the 0, 1 mask into the -1000, 0 mask needed for masking softmax logits
    final_mask = model.builder.aiOnnx.sub([final_mask, model.constant_tensor(1.0, config.dtype)])
    final_mask = model.builder.aiOnnx.mul([final_mask, model.constant_tensor(1000.0, config.dtype)])

    # Add head dimension to mask
    final_mask = model.builder.reshape_const(model.builder.aiOnnx, [final_mask],
                                             [config.micro_batch_size, 1, config.sequence_length, config.sequence_length])
    return final_mask


def mlm_projection_gather_indexes(model, x):
    config = model.config
    onehot_size = model.builder.aiOnnx.constant(np.array([config.sequence_length], dtype=np.int32), "onehot_size")
    onehot_values = model.builder.aiOnnx.constant(np.array([0, 1], dtype=config.dtype), "onehot_values")
    onehot = model.builder.aiOnnx.onehot([model.inputs["masked_lm_positions"], onehot_size, onehot_values])
    x = model.builder.aiOnnx.matmul([onehot, x])
    x = model.builder.checkpointOutput([x])[0]
    return x


def pooler_gather_indexes(model, pooler_input):
    config = model.config
    onehot_size = model.builder.aiOnnx.constant(np.array([config.sequence_length], dtype=np.int32), "onehot_size")
    onehot_values = model.builder.aiOnnx.constant(np.array([0, 1], dtype=config.dtype), "onehot_values")
    onehot = model.builder.aiOnnx.onehot([model.inputs["nsp_positions"], onehot_size, onehot_values])
    onehot = model.builder.checkpointOutput([onehot])[0]
    pooler_input = model.builder.aiOnnx.matmul([onehot, pooler_input])
    pooler_input = model.builder.checkpointOutput([pooler_input])[0]
    return pooler_input


def pretraining_loss_and_accuracy(model, logits):
    # MLM and NSP loss and accuracy calculation
    # some tensors are shared between the two loss calculations

    # Which outputs should be streamed back to host
    outputs_to_anchor = {}
    config = model.config

    # MLM
    with model.mlm_scope:
        mlm_logits = logits[0]
        mlm_predictions = model.builder.aiOnnx.argmax([mlm_logits], axis=-1,
                                                      keepdims=0, debugContext=f"MLM/ArgMax")
        mlm_labels = model.labels["masked_lm_ids"]
        mlm_labels = model.builder.aiOnnx.cast([mlm_labels], "INT32")
        mlm_seq_ind = model.inputs["masked_lm_weights"]
        mlm_seq_ind = model.builder.reshape_const(model.builder.aiOnnx, [mlm_seq_ind], [config.micro_batch_size, 1, -1])

        # Sequences per pack
        nsp_weights = model.builder.aiOnnx.cast([model.labels["nsp_weights"]], "INT32")
        sequences_in_sample = model.builder.aiOnnx.reducesum([nsp_weights], keepdims=False)
        sequences_in_sample = model.builder.aiOnnx.cast([sequences_in_sample], "FLOAT")
        sequences_in_sample = model.builder.checkpointOutput([sequences_in_sample])[0]

        # First compute the per-token nll and do not perform any reduction
        mlm_probs = model.builder.aiOnnx.softmax([mlm_logits], axis=-1)
        nll_per_token = model.builder.aiGraphcore.nllloss([mlm_probs, mlm_labels], ignoreIndex=0,
                                                          reduction=popart.ReductionType.NoReduction, debugContext=f"MLM/loss")
        nll_per_token = model.builder.checkpointOutput([nll_per_token])[0]
        nll_per_token = model.builder.aiOnnx.cast([nll_per_token], "FLOAT")
        nll_per_token = model.builder.reshape_const(model.builder.aiOnnx, [nll_per_token], [config.micro_batch_size, 1, -1])

        # Calculate the per token accuracy (hit or miss) and do not perform any reduction
        mlm_accuracy_per_token = model.builder.aiOnnx.equal([mlm_predictions, mlm_labels])
        mlm_accuracy_per_token = model.detach(mlm_accuracy_per_token)
        mlm_accuracy_per_token = model.builder.aiOnnx.cast([mlm_accuracy_per_token], "FLOAT")
        mlm_accuracy_per_token = model.builder.reshape_const(model.builder.aiOnnx, [mlm_accuracy_per_token], [config.micro_batch_size, 1, -1])

        # Now expand the per-token loss an a per-sequence basis [B, T] -> [B, max_sequences_per_pack, T]
        # The sequence selection tensor select the different sequences in a pack by masking
        sequence_selection = np.array(range(1, config.max_sequences_per_pack + 1)).reshape([1, -1, 1])
        sequence_selection = model.constant_tensor(sequence_selection, dtype=np.uint32)
        sequence_selection = model.builder.aiOnnx.equal([mlm_seq_ind, sequence_selection])
        sequence_selection = model.detach(sequence_selection)
        sequence_selection_i = model.builder.aiOnnx.cast([sequence_selection], "INT32")
        sequence_selection_f = model.builder.aiOnnx.cast([sequence_selection], "FLOAT")
        sequence_selection_f = model.builder.checkpointOutput([sequence_selection_f])[0]

        # Calculate the per-sequence normalization constants (if 0, increase to 1 to avoid NaNs)
        attempted = model.builder.aiOnnx.reducesum([sequence_selection_i], axes=[-1], keepdims=True)
        zero = model.constant_tensor([0], dtype=np.int32)
        attempted_mask = model.detach(model.builder.aiOnnx.equal([attempted, zero]))  # prevent nans
        attempted_mask_f = model.builder.aiOnnx.cast([attempted_mask], "FLOAT")
        attempted = model.builder.aiOnnx.cast([attempted], "FLOAT")
        attempted = model.builder.aiOnnx.add([attempted, attempted_mask_f])
        attempted = model.detach(attempted)
        attempted = model.builder.checkpointOutput([attempted])[0]

        # Calculate per sequence loss
        nll_per_sequence = model.builder.aiOnnx.mul([nll_per_token, sequence_selection_f])
        nll_per_sequence = model.builder.aiOnnx.div([nll_per_sequence, attempted])

        # Divide totoal loss by number of sequences to get average loss
        mlm_loss = model.builder.aiOnnx.reducesum([nll_per_sequence], keepdims=False)
        mlm_loss = model.builder.aiOnnx.div([mlm_loss, sequences_in_sample])
        outputs_to_anchor[mlm_loss] = popart.AnchorReturnType("SUM")

        # Now compute the MLM accuracy
        mlm_accuracy_per_sequence = model.builder.aiOnnx.mul([mlm_accuracy_per_token, sequence_selection_f])
        mlm_accuracy_per_sequence = model.builder.aiOnnx.div([mlm_accuracy_per_sequence, attempted])
        mlm_accuracy = model.builder.aiOnnx.reducesum([mlm_accuracy_per_sequence], keepdims=False)
        mlm_accuracy = model.builder.aiOnnx.div([mlm_accuracy, sequences_in_sample])
        # For accuracy we need to return all values since each batch has a different number of sequences
        outputs_to_anchor[mlm_accuracy] = popart.AnchorReturnType("ALL")


    # NSP accuracy and loss computed per-pack
    with model.nsp_scope:
        nsp_logits = logits[1]
        nsp_predictions = model.builder.aiOnnx.argmax([nsp_logits], axis=-1,
                                                      keepdims=0, debugContext=f"NSP/ArgMax")
        nsp_labels = model.builder.aiOnnx.cast([model.labels["nsp_labels"]], "INT32")

        # Loss and accuracy per token
        nsp_probs = model.builder.aiOnnx.softmax([nsp_logits], axis=-1)
        nsp_nll_per_token = model.builder.aiGraphcore.nllloss([nsp_probs, model.labels["nsp_labels"]], ignoreIndex=None,
                                                              reduction=popart.ReductionType.NoReduction, debugContext=f"NSP/loss")
        nsp_nll_per_token = model.builder.checkpointOutput([nsp_nll_per_token])[0]
        nsp_nll_per_token = model.builder.aiOnnx.cast([nsp_nll_per_token], "FLOAT")
        nsp_accuracy_per_token = model.builder.aiOnnx.equal([nsp_labels, nsp_predictions])
        nsp_accuracy_per_token = model.builder.aiOnnx.cast([nsp_accuracy_per_token], "FLOAT")

        # Attempted tokens
        nsp_weights_f = model.builder.aiOnnx.cast([nsp_weights], "FLOAT")  # 1 or 0 mask
        attempted = model.builder.aiOnnx.reducesum([nsp_weights_f], axes=[-1], keepdims=True)  # always > 0

        # NSP loss
        nsp_loss = model.builder.aiOnnx.mul([nsp_nll_per_token, nsp_weights_f])
        nsp_loss = model.builder.aiOnnx.reducesum([nsp_loss], axes=[-1], keepdims=False)
        nsp_loss = model.builder.aiOnnx.div([nsp_loss, attempted])
        nsp_loss = model.builder.aiOnnx.reducemean([nsp_loss], keepdims=False)
        outputs_to_anchor[nsp_loss] = popart.AnchorReturnType("SUM")

        # NSP accuracy
        nsp_accuracy = model.builder.aiOnnx.mul([nsp_accuracy_per_token, nsp_weights_f])
        nsp_accuracy = model.builder.aiOnnx.div([nsp_accuracy, attempted])
        nsp_accuracy = model.builder.aiOnnx.reducesum([nsp_accuracy], axes=[-1], keepdims=False)
        nsp_accuracy = model.builder.aiOnnx.reducemean([nsp_accuracy], keepdims=False)
        outputs_to_anchor[nsp_accuracy] = popart.AnchorReturnType("SUM")

    # MLM + NSP is final loss
    with model.final_loss_scope:
        final_loss = model.builder.aiOnnx.add([mlm_loss, nsp_loss], "FinalLoss")

    for out in outputs_to_anchor.keys():
        model.builder.addOutputTensor(out)
    return [mlm_loss, nsp_loss], [mlm_accuracy, nsp_accuracy], final_loss, outputs_to_anchor
