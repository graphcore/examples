# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import numpy as np
from pathlib import Path
import torch
from torch import nn
import popart
import onnx

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertEmbeddings
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model, TestFailureError


'''
Tests the embedding with no projection. This is the case for SQUAD.
'''


def test_embedding_fwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })
    config = BertConfig(vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=['gather'],
                        inference=True)
    popart_model = Bert(config, builder=builder)
    # Prevent virtualGraph attributes being added to the ops.
    popart_model.embedding_scope = popart_model.device_scope(None, None)
    popart_model.embedding_split_scope = popart_model.embedding_scope

    sequence_info = popart.TensorInfo(
        "INT32", [config.batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32),
        segments:
        np.random.randint(0, 2,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32)
    }

    # Use the custom embedding for layout
    output = popart_model.embedding(indices, positions, segments)

    proto = builder.getModelProto()

    outputs, post_proto = run_py(proto, data, output,
                                 user_options={"enableStochasticRounding": True})

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.batch_size, config.sequence_length) for t in [indices, positions, segments]]

    torch_to_onnx = {
        "word_embeddings.weight": "Embedding_Dict",
        "position_embeddings.weight": "Positional_Dict",
        "token_type_embeddings.weight": "Segment_Dict",
        "LayerNorm.weight": "Gamma",
        "LayerNorm.bias": "Beta"
    }

    transposed_weights = {
        "word_embeddings.weight": np.transpose,
        "position_embeddings.weight": np.transpose,
    }

    #  ------------------- PyTorch -------------------------
    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps))
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, torch_to_onnx,
                          transposed_weights)

    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs)


def test_embedding_bwd(custom_ops):
    l1_lambda = 0.1

    #  ------------------- PopART --------------------
    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })
    config = BertConfig(vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=['gather'])
    popart_model = Bert(config, builder=builder)
    # Prevent virtualGraph attributes being added to the ops.
    popart_model.embedding_scope = popart_model.device_scope(None, None)
    popart_model.embedding_split_scope = popart_model.embedding_scope

    sequence_info = popart.TensorInfo(
        "INT32", [config.batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32),
        segments:
        np.random.randint(0, 2,
                          (config.batch_size * config.sequence_length)).astype(
                              np.int32)
    }

    output = popart_model.embedding(indices, positions, segments)

    proto = builder.getModelProto()

    l1 = popart.L1Loss(output, "l1LossVal", l1_lambda)
    optimizer = popart.ConstSGD(0.01)

    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 loss=l1,
                                 optimizer=optimizer,
                                 user_options={"enableStochasticRounding": True})

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.batch_size, config.sequence_length) for t in [indices, positions, segments]]

    torch_to_onnx = {
        "word_embeddings.weight": "Embedding_Dict",
        "position_embeddings.weight": "Positional_Dict",
        "token_type_embeddings.weight": "Segment_Dict",
        "LayerNorm.weight": "Gamma",
        "LayerNorm.bias": "Beta"
    }

    transposed_weights = {
        "word_embeddings.weight": np.transpose,
        "position_embeddings.weight": np.transpose,
    }

    #  ------------------- PyTorch -------------------------

    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model,
                          proto,
                          torch_to_onnx,
                          transform=transposed_weights)

    optim = torch.optim.SGD(torch_model.parameters(),
                            0.01,
                            weight_decay=0.0,
                            momentum=0.0)

    torch_output = torch_model(*[torch.from_numpy(t).long() for t in inputs])
    torch_loss = l1_lambda * torch.norm(torch_output, 1)
    torch_loss.backward()
    optim.step()

    torch_outputs = [torch_output.detach().numpy()]

    check_tensors(torch_outputs, outputs)

    check_model(torch_model,
                post_proto,
                torch_to_onnx,
                transform=transposed_weights)
