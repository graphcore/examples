# Copyright 2019 Graphcore Ltd.
import math
import random
import numpy as np
import torch

import popart
import onnx

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertForMaskedLM
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model


def get_mapping(config, init=None):
    if init is None:
        init = {}
    embedding_proj = {
        "bert.embeddings.word_embeddings.weight": "Embedding/Embedding_Dict",
        "bert.embeddings.position_embeddings.weight": "Embedding/Positional_Dict",
        "bert.embeddings.token_type_embeddings.weight": "Embedding/Segment_Dict",
        "bert.embeddings.LayerNorm.weight": "Embedding/Gamma",
        "bert.embeddings.LayerNorm.bias": "Embedding/Beta",
    }
    init.update(**embedding_proj)
    for i in range(config.num_layers):
        layer = {
            f"bert.encoder.layer.{i}.attention.self.query.weight": f"Layer{i}/Attention/QKV",
            f"bert.encoder.layer.{i}.attention.self.key.weight": f"Layer{i}/Attention/QKV",
            f"bert.encoder.layer.{i}.attention.self.value.weight": f"Layer{i}/Attention/QKV",
            f"bert.encoder.layer.{i}.attention.output.dense.weight": f"Layer{i}/Attention/Out",
            f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight": f"Layer{i}/Attention/Gamma",
            f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias": f"Layer{i}/Attention/Beta",
            f"bert.encoder.layer.{i}.intermediate.dense.weight": f"Layer{i}/FF/1/W",
            f"bert.encoder.layer.{i}.intermediate.dense.bias": f"Layer{i}/FF/1/B",
            f"bert.encoder.layer.{i}.output.dense.weight": f"Layer{i}/FF/2/W",
            f"bert.encoder.layer.{i}.output.dense.bias": f"Layer{i}/FF/2/B",
            f"bert.encoder.layer.{i}.output.LayerNorm.weight": f"Layer{i}/FF/Gamma",
            f"bert.encoder.layer.{i}.output.LayerNorm.bias": f"Layer{i}/FF/Beta",
        }
        init.update(**layer)
    return init


def get_transform(config, init=None):
    if init is None:
        init = {}

    def q_transform(arr):
        return arr[:, 0:config.hidden_size].T

    def k_transform(arr):
        return arr[:, config.hidden_size:config.hidden_size * 2].T

    def v_transform(arr):
        return arr[:, config.hidden_size * 2:config.hidden_size * 3].T

    embedding_proj = {
        "bert.embeddings.word_embeddings.weight": np.transpose,
        "bert.embeddings.position_embeddings.weight": np.transpose,
    } if "gather" in config.custom_ops else {}
    init.update(**embedding_proj)
    for i in range(config.num_layers):
        layer = {
            f"bert.encoder.layer.{i}.attention.self.query.weight": q_transform,
            f"bert.encoder.layer.{i}.attention.self.key.weight": k_transform,
            f"bert.encoder.layer.{i}.attention.self.value.weight": v_transform,
            f"bert.encoder.layer.{i}.attention.output.dense.weight": np.transpose,
            f"bert.encoder.layer.{i}.intermediate.dense.weight": np.transpose,
            f"bert.encoder.layer.{i}.output.dense.weight": np.transpose,
        }
        init.update(**layer)
    return init


def fwd_graph(popart_model, torch_model, mapping=None, transform=None):
    #  ------------------- PopART --------------------
    config = popart_model.config
    builder = popart_model.builder

    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)
    data = {
        indices: np.random.randint(
            0, config.vocab_length, (config.batch_size * config.sequence_length)).astype(np.uint32),
        positions: np.random.randint(
            0, config.sequence_length, (config.batch_size * config.sequence_length)).astype(np.uint32),
        segments: np.random.randint(
            0, 2, (config.batch_size * config.sequence_length)).astype(np.uint32)
    }

    output = popart_model.build_graph(indices, positions, segments)
    proto = builder.getModelProto()

    outputs, post_proto = run_py(proto, data, output,
                                 ipus=math.ceil(config.num_layers / config.layers_per_ipu) + popart_model.layer_offset)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = {
        "input_ids": data[indices].reshape(config.batch_size, config.sequence_length).astype(np.int32),
        "position_ids": data[positions].reshape(config.batch_size, config.sequence_length).astype(np.int32),
        "token_type_ids": data[segments].reshape(config.batch_size, config.sequence_length).astype(np.int32)
    }

    torch_to_onnx = get_mapping(config, init=mapping)

    transform_weights = get_transform(config, init=transform)

    #  ------------------- PyTorch -------------------------
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto,
                          torch_to_onnx, transform_weights)

    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs)


def bwd_graph(popart_model,
              torch_model,
              popart_loss_fn,
              torch_loss_fn,
              mapping=None,
              transform=None):
    np.random.seed(1984)
    random.seed(1984)
    torch.manual_seed(1984)

    #  ------------------- PopART --------------------
    config = popart_model.config
    builder = popart_model.builder

    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)
    data = {
        indices: np.random.randint(
            0, config.vocab_length, (config.batch_size * config.sequence_length)).astype(np.uint32),
        positions: np.random.randint(
            0, config.sequence_length, (config.batch_size * config.sequence_length)).astype(np.uint32),
        segments: np.random.randint(
            0, 2, (config.batch_size * config.sequence_length)).astype(np.uint32)
    }

    output = popart_model.build_graph(indices, positions, segments)
    proto = builder.getModelProto()

    losses = popart_loss_fn(output)

    optimizer = popart.ConstSGD(0.01)

    outputs, post_proto = run_py(
        proto, data, output, loss=losses, optimizer=optimizer,
        ipus=math.ceil(config.num_layers / config.layers_per_ipu) + popart_model.layer_offset)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = {
        "input_ids": data[indices].reshape(config.batch_size, config.sequence_length).astype(np.int32),
        "position_ids": data[positions].reshape(config.batch_size, config.sequence_length).astype(np.int32),
        "token_type_ids": data[segments].reshape(config.batch_size, config.sequence_length).astype(np.int32)
    }

    torch_to_onnx = get_mapping(config, init=mapping)

    transform_weights = get_transform(config, init=transform)

    #  ------------------- PyTorch -------------------------
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto,
                          torch_to_onnx, transform_weights)

    optim = torch.optim.SGD(torch_model.parameters(), 0.01,
                            weight_decay=0.0, momentum=0.0)

    torch_outputs = torch_model(
        **{k: torch.from_numpy(t).long() for k, t in inputs.items()})
    torch_loss = torch_loss_fn(torch_outputs)
    torch_loss.backward()
    optim.step()

    check_tensors([output.detach().numpy() for output in torch_outputs], outputs)

    check_model(torch_model, post_proto,
                torch_to_onnx, transform_weights,
                margin=2e-5)
