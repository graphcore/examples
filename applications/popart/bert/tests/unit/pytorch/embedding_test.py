# Copyright 2019 Graphcore Ltd.
import numpy as np
import torch
import popart
import pytest
import onnx

from bert_model import BertConfig, Bert, ExecutionMode, get_model
from tests.torch_bert import BertConfig as TorchBertConfig, BertEmbeddings
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model, TestFailureError


'''
Tests the embedding with no projection. This is the case for SQUAD.
'''
test_modes = [ExecutionMode.DEFAULT, pytest.param(ExecutionMode.PHASED, marks=pytest.mark.requires_remote_buffers)]

TORCH_TO_ONNX = {
    ExecutionMode.DEFAULT: {
        "word_embeddings.weight": "Embedding_Dict",
        "position_embeddings.weight": "Positional_Dict",
        "token_type_embeddings.weight": "Segment_Dict",
        "LayerNorm.weight": "Gamma",
        "LayerNorm.bias": "Beta"
    },
    ExecutionMode.PHASED: {
        "word_embeddings.weight": "Embeddings/Token/weight",
        "position_embeddings.weight": "Embeddings/Position/weight",
        "token_type_embeddings.weight": "Embeddings/Segment/weight",
        "LayerNorm.weight": "Embeddings/Norm/Gamma",
        "LayerNorm.bias": "Embeddings/Norm/Beta"
    },
}


@pytest.mark.parametrize("mode", test_modes)
def test_embedding_fwd(custom_ops, mode):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        inference=True,
                        embedding_serialization_vocab_steps=1)
    popart_model = get_model(config, mode, 'embedding')

    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])
    indices = popart_model.builder.addInputTensor(sequence_info)
    positions = popart_model.builder.addInputTensor(sequence_info)
    segments = popart_model.builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32),
        segments:
        np.random.randint(0, 2,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": 1,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(indices, positions, segments)
    else:
        user_options = {"enableStochasticRounding": True}
        output = popart_model.embedding(indices, positions, segments)

    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 user_options=user_options,
                                 execution_mode=mode)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.batch_size, config.sequence_length).astype(np.int32) for t in [indices, positions, segments]]

    #  ------------------- PyTorch -------------------------
    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps))
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, TORCH_TO_ONNX[mode], {})

    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs, margin=5e-7)


@pytest.mark.parametrize("mode", test_modes)
def test_embedding_bwd(custom_ops, mode):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        update_embedding_dict=False,
                        embedding_serialization_vocab_steps=1)

    popart_model = get_model(config, mode, 'embedding')
    # Prevent virtualGraph attributes being added to the ops.

    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])
    indices = popart_model.builder.addInputTensor(sequence_info)
    positions = popart_model.builder.addInputTensor(sequence_info)
    segments = popart_model.builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32),
        segments:
        np.random.randint(0, 2,
                          (config.batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    optimizer = popart.ConstSGD(0.01)
    l1_lambda = 0.1

    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": 1,
            "executionPhases": popart_model.total_execution_phases,
        }
        output = popart_model(indices, positions, segments)
        with popart_model.scope_provider(popart_model.builder, popart_model.norm.scope):
            l1 = popart_model.builder.aiGraphcore.l1loss(
                [output],
                l1_lambda,
                debugPrefix="l1LossVal",
                reduction=popart.ReductionType.Sum)
    else:
        user_options = {"enableStochasticRounding": True}
        output = popart_model.embedding(indices, positions, segments)
        l1 = popart_model.builder.aiGraphcore.l1loss(
            [output],
            l1_lambda,
            debugPrefix="l1LossVal",
            reduction=popart.ReductionType.Sum)

    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 ipus=1,
                                 loss=l1,
                                 optimizer=optimizer,
                                 user_options=user_options,
                                 execution_mode=mode)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.batch_size, config.sequence_length).astype(np.int32) for t in [indices, positions, segments]]

    #  ------------------- PyTorch -------------------------

    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, TORCH_TO_ONNX[mode], {})

    optim = torch.optim.SGD(torch_model.parameters(),
                            0.01,
                            weight_decay=0.0,
                            momentum=0.0)

    torch_output = torch_model(*[torch.from_numpy(t).long() for t in inputs])
    torch_loss = l1_lambda * torch.norm(torch_output, 1)
    torch_loss.backward()
    optim.step()

    torch_outputs = [torch_output.detach().numpy()]

    check_tensors(torch_outputs, outputs, margin=1e-06)

    check_model(torch_model, post_proto, TORCH_TO_ONNX[mode], {}, margin=1e-06)
