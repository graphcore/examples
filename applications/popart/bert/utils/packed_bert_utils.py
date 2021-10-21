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
    # Input mask contains a sequence index for each token in the pack
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
    model.labels["masked_lm_ids"] = model.builder.addInputTensor(mlm_info, "masked_lm_ids")
    model.inputs["masked_lm_weights"] = model.builder.addInputTensor(mlm_info, "masked_lm_weights")

    # NSP (there are multiple [CLS] tokens per pack)
    nsp_shape = [config.micro_batch_size, config.max_sequences_per_pack]
    nsp_info = popart.TensorInfo("UINT32", nsp_shape)
    model.labels["nsp_labels"] = model.builder.addInputTensor(nsp_info, "nsp_labels")
    model.labels["nsp_weights"] = model.builder.addInputTensor(nsp_info, "nsp_weights")

    # The shapes for the constructed inputs and labels
    # (in order of appearance in the dataset). Required for compatibility with legacy code
    input_tensor_shapes = [sequence_shape]*4 + [mlm_shape]*2 + [nsp_shape]*2
    input_tensor_names = ["input_ids", "input_mask", "segment_ids", "position_ids"]
    input_tensor_names += ["masked_lm_ids", "masked_lm_weights"]
    input_tensor_names += ["nsp_labels", "nsp_weights"]
    return [(name, shape) for name, shape in zip(input_tensor_names, input_tensor_shapes)]


def logits_graph(model):
    # Glue code for compatibility with non-packing version and naming convention
    indices = model.inputs["input_ids"]
    positions = model.inputs["position_ids"]
    segments = model.inputs["segment_ids"]
    masks = [model.inputs["input_mask"]]
    list_of_logits = model.build_graph(indices, positions, segments, masks)
    return list_of_logits


def attention_mask(model, x):
    """
    model.input["input_mask"] is used to create the a mask which
    prevents cross-contamination between sequences in a pack
    """
    config = model.config
    input_mask = model.inputs["input_mask"]
    final_mask = model.builder.customOp(opName="AttentionMask",
                                        opVersion=1,
                                        domain="ai.graphcore",
                                        inputs=[input_mask, x],
                                        attributes={"dataType": model.config.popart_dtype})[0]
    final_mask = model.detach(final_mask)
    return final_mask


def slice_nsp_tokens(model, pooler_input):
    """
    The nsp tokens have been rearranged to the back of the sequence and can
    simply be sliced off.
    """

    config = model.config
    starts = config.sequence_length - config.max_sequences_per_pack
    ends = config.sequence_length
    pooler_input = model.builder.aiOnnxOpset9.slice([pooler_input], axes=[1], starts=[starts], ends=[ends])
    pooler_input = model.builder.reshape_const(model.builder.aiOnnx, [pooler_input],
                                               [config.micro_batch_size, config.max_sequences_per_pack,
                                               config.hidden_size])
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
        mlm_seq_ind = model.builder.reshape_const(model.builder.aiOnnx, [mlm_seq_ind], [config.micro_batch_size, -1])

        # MLM loss
        # computed on a pertoken basis (original BERT implementation)
        mlm_probs = model.builder.aiOnnx.softmax([mlm_logits], axis=-1)
        mlm_loss = model.builder.aiGraphcore.nllloss([mlm_probs, mlm_labels], ignoreIndex=0,
                                                     reduction=popart.ReductionType.Sum, debugContext=f"MLM/loss")
        mlm_loss = model.builder.aiOnnx.cast([mlm_loss], "FLOAT")
        outputs_to_anchor[mlm_loss] = popart.AnchorReturnType("SUM")

        # MLM accuracy
        mlm_accuracy_per_token = model.builder.aiOnnx.equal([mlm_predictions, mlm_labels])
        mlm_accuracy_per_token = model.detach(mlm_accuracy_per_token)
        mlm_accuracy_per_token = model.builder.aiOnnx.cast([mlm_accuracy_per_token], "FLOAT")
        mlm_token_weights = model.builder.aiOnnx.greater([mlm_seq_ind, model.constant_tensor([0], dtype=np.uint32)])
        mlm_token_weights = model.builder.aiOnnx.cast([mlm_token_weights], "FLOAT")
        mlm_accuracy_per_token = model.builder.aiOnnx.mul([mlm_accuracy_per_token, mlm_token_weights])
        mlm_accuracy = model.builder.aiOnnx.reducesum([mlm_accuracy_per_token], keepdims=False)
        outputs_to_anchor[mlm_accuracy] = popart.AnchorReturnType("SUM")


    # NSP accuracy and loss computed per-pack
    with model.nsp_scope:
        nsp_logits = logits[1]
        nsp_predictions = model.builder.aiOnnx.argmax([nsp_logits], axis=-1,
                                                      keepdims=0, debugContext=f"NSP/ArgMax")
        nsp_labels = model.builder.aiOnnx.cast([model.labels["nsp_labels"]], "INT32")
        nsp_weights = model.builder.aiOnnx.cast([model.labels["nsp_weights"]], "INT32")
        nsp_weights_f = model.builder.aiOnnx.cast([nsp_weights], "FLOAT")  # 1 or 0 mask

        # NSP loss
        nsp_probs = model.builder.aiOnnx.softmax([nsp_logits], axis=-1)
        nsp_nll_per_token = model.builder.aiGraphcore.nllloss([nsp_probs, model.labels["nsp_labels"]], ignoreIndex=None,
                                                              reduction=popart.ReductionType.NoReduction, debugContext=f"NSP/loss")
        nsp_nll_per_token = model.builder.aiOnnx.cast([nsp_nll_per_token], "FLOAT")
        nsp_loss = model.builder.aiOnnx.mul([nsp_nll_per_token, nsp_weights_f])
        nsp_loss = model.builder.aiOnnx.reducesum([nsp_loss], keepdims=False)
        outputs_to_anchor[nsp_loss] = popart.AnchorReturnType("SUM")

        # NSP accuracy
        nsp_accuracy_per_token = model.builder.aiOnnx.equal([nsp_labels, nsp_predictions])
        nsp_accuracy_per_token = model.builder.aiOnnx.cast([nsp_accuracy_per_token], "FLOAT")
        nsp_accuracy = model.builder.aiOnnx.mul([nsp_accuracy_per_token, nsp_weights_f])
        nsp_accuracy = model.builder.aiOnnx.reducesum([nsp_accuracy], keepdims=False)
        outputs_to_anchor[nsp_accuracy] = popart.AnchorReturnType("SUM")

    # MLM + NSP is final loss
    with model.final_loss_scope:
        final_loss = model.builder.aiOnnx.add([mlm_loss, nsp_loss], "FinalLoss")

    for out in outputs_to_anchor.keys():
        model.builder.addOutputTensor(out)
    return [mlm_loss, nsp_loss], [mlm_accuracy, nsp_accuracy], final_loss, outputs_to_anchor
