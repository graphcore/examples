# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import numpy as np
from pathlib import Path
import torch

import popart
import onnx

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertAttention
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model


'''
Tests the attention custom_op.
'''


def get_torch_mask(config, masks):
    base = np.arange(config.sequence_length)
    # Mask tokens mask
    mmask = np.less(base, masks[0])
    _mask = np.greater_equal(base, config.mask_tokens)
    mmask = np.logical_or(mmask, _mask)
    # Sequence mask
    smask = np.less(base, masks[1])
    final_mask = np.logical_and(mmask, smask).astype(config.dtype)
    final_mask -= 1.0
    final_mask *= 1000.0
    return np.broadcast_to(final_mask, [config.sequence_length] * 2)


def test_attention_fwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder()
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=['attention'],
                        inference=True)

    popart_model = Bert(config, builder=builder)

    input_info = popart.TensorInfo(config.popart_dtype, [config.batch_size * config.sequence_length, config.hidden_size])
    input_tensor = builder.addInputTensor(input_info)
    mask_info = popart.TensorInfo("INT32", [config.batch_size])
    mmask_tensor = builder.addInputTensor(mask_info)
    smask_tensor = builder.addInputTensor(mask_info)
    data = {
        input_tensor: np.random.normal(0, 0.02, input_info.shape()).astype(config.dtype),
        mmask_tensor: np.random.randint(0, config.mask_tokens + 1, (config.batch_size,)).astype(np.int32),
        smask_tensor: np.random.randint(config.mask_tokens, config.sequence_length + 1, (config.batch_size,)).astype(np.int32)
    }

    output = popart_model.attention(input_tensor, [mmask_tensor, smask_tensor])
    proto = builder.getModelProto()

    outputs, post_proto = run_py(proto, data, output)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[input_tensor].reshape(config.batch_size, config.sequence_length, config.hidden_size).astype(np.float32),
              get_torch_mask(config, [data[mmask_tensor], data[smask_tensor]])]

    torch_to_onnx = {
        "self.query.weight": "QKV",
        "self.key.weight": "QKV",
        "self.value.weight": "QKV",
        "output.dense.weight": "Out",
        "output.LayerNorm.weight": "Gamma",
        "output.LayerNorm.bias": "Beta"
    }

    split_qkv = {
        "self.query.weight": lambda arr: arr[:, 0:config.hidden_size].T,
        "self.key.weight": lambda arr: arr[:, config.hidden_size:config.hidden_size * 2].T,
        "self.value.weight": lambda arr: arr[:, config.hidden_size * 2:config.hidden_size * 3].T,
        "output.dense.weight": np.transpose
    }

    #  ------------------- PyTorch -------------------------
    torch_model = BertAttention(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        config.num_layers,
                        config.attention_heads,
                        layer_norm_eps=config.layer_norm_eps))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto,
                          torch_to_onnx, transform=split_qkv)

    # Model to test against
    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs)


def test_attention_bwd(custom_ops):
    l1_lambda = 0.1

    #  ------------------- PopART --------------------
    builder = popart.Builder()
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=['attention'])
    popart_model = Bert(config, builder=builder)

    input_info = popart.TensorInfo(config.popart_dtype, [config.batch_size * config.sequence_length, config.hidden_size])
    input_tensor = builder.addInputTensor(input_info)
    mask_info = popart.TensorInfo("INT32", [config.batch_size])
    mmask_tensor = builder.addInputTensor(mask_info)
    smask_tensor = builder.addInputTensor(mask_info)
    data = {
        input_tensor: np.random.normal(0, 0.02, input_info.shape()).astype(config.dtype),
        mmask_tensor: np.random.randint(0, config.mask_tokens + 1, (config.batch_size,)).astype(np.int32),
        smask_tensor: np.random.randint(config.mask_tokens, config.sequence_length + 1, (config.batch_size,)).astype(np.int32)
    }

    output = popart_model.attention(input_tensor, [mmask_tensor, smask_tensor])
    proto = builder.getModelProto()

    l1 = popart.L1Loss(output, "l1LossVal", l1_lambda)
    optimizer = popart.ConstSGD(0.01)

    outputs, post_proto = run_py(
        proto, data, (output, l1.output(0)), loss=l1, optimizer=optimizer)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[input_tensor].reshape(config.batch_size, config.sequence_length, config.hidden_size),
              get_torch_mask(config, [data[mmask_tensor], data[smask_tensor]])]

    torch_to_onnx = {
        "self.query.weight": "QKV",
        "self.key.weight": "QKV",
        "self.value.weight": "QKV",
        "output.dense.weight": "Out",
        "output.LayerNorm.weight": "Gamma",
        "output.LayerNorm.bias": "Beta"
    }

    split_qkv = {
        "self.query.weight": lambda arr: arr[:, 0:config.hidden_size].T,
        "self.key.weight": lambda arr: arr[:, config.hidden_size:config.hidden_size * 2].T,
        "self.value.weight": lambda arr: arr[:, config.hidden_size * 2:config.hidden_size * 3].T,
        "output.dense.weight": np.transpose
    }

    #  ------------------- PyTorch -------------------------
    torch_model = BertAttention(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        config.num_layers,
                        config.attention_heads,
                        layer_norm_eps=config.layer_norm_eps))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto,
                          torch_to_onnx, transform=split_qkv)

    optim = torch.optim.SGD(torch_model.parameters(), 0.01,
                            weight_decay=0.0, momentum=0.0)

    torch_output = torch_model(*[torch.from_numpy(t).float() for t in inputs])[0]
    torch_loss = l1_lambda * torch.norm(torch_output, 1)
    torch_loss.backward()
    optim.step()

    check_tensors([torch_output.detach().numpy()], outputs)

    check_model(torch_model, post_proto, torch_to_onnx,
                transform=split_qkv)
