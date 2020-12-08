# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

import numpy as np
import pytest
from scipy.stats.stats import moment
import torch

import popart
import onnx

from bert_model import BertConfig, ExecutionMode, get_model
from tests.torch_bert import BertConfig as TorchBertConfig, BertAttention
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model
'''
Tests the attention op.
'''
test_modes = [ExecutionMode.DEFAULT, pytest.param(ExecutionMode.PHASED, marks=pytest.mark.requires_remote_buffers)]

TORCH_TO_ONNX = {
    ExecutionMode.DEFAULT: {
        "self.query.weight": "QKV",
        "self.key.weight": "QKV",
        "self.value.weight": "QKV",
        "output.dense.weight": "Out",
        "output.LayerNorm.weight": "Gamma",
        "output.LayerNorm.bias": "Beta"
    },
    ExecutionMode.PHASED: {
        "self.query.weight": "Attention/QKV",
        "self.key.weight": "Attention/QKV",
        "self.value.weight": "Attention/QKV",
        "output.dense.weight": "Attention/Out",
        "output.LayerNorm.weight": "Attention/Norm/Gamma",
        "output.LayerNorm.bias": "Attention/Norm/Beta"
    }
}


def setup_function(func):
    np.random.seed(1984)


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
    final_masks = []
    for i in range(0, config.batch_size):
        final_masks.append(np.broadcast_to(final_mask[i], (1, config.sequence_length, config.sequence_length)))
    return np.stack(final_masks, axis=0)


@pytest.mark.parametrize("mode", test_modes)
@pytest.mark.parametrize("batch_size, batch_serialisation_factor, number_attention_splits", [(1, 1, 1), (2, 2, 1), (2, 2, 2), (2, 2, 4)])
def test_attention_fwd(mode, batch_size, batch_serialisation_factor, number_attention_splits):
    #  ------------------- PopART --------------------
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        batch_size=batch_size,
                        hidden_size=768,
                        attention_heads=4,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        inference=True,
                        num_attention_splits=number_attention_splits)
    popart_model = get_model(config, mode, 'attention')

    input_info = popart.TensorInfo(config.popart_dtype, [config.batch_size * config.sequence_length, config.hidden_size])
    input_tensor = popart_model.builder.addInputTensor(input_info)
    mask_info = popart.TensorInfo("UINT32", [config.batch_size, config.sequence_length])
    mmask_tensor = popart_model.builder.addInputTensor(mask_info)
    smask_tensor = popart_model.builder.addInputTensor(mask_info)
    data = {
        input_tensor: np.random.normal(0, 0.02, input_info.shape()).astype(config.dtype),
        mmask_tensor: np.random.randint(0, config.mask_tokens + 1, (config.batch_size, config.sequence_length,)).astype(np.uint32),
        smask_tensor: np.random.randint(config.mask_tokens, config.sequence_length + 1, (config.batch_size, config.sequence_length, )).astype(np.uint32)
    }

    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": batch_serialisation_factor,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(input_tensor, [mmask_tensor, smask_tensor])
    else:
        user_options = {"enableStochasticRounding": True}
        output = popart_model.attention(input_tensor,
                                        [mmask_tensor, smask_tensor])

    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 user_options=user_options,
                                 execution_mode=mode)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[input_tensor].reshape(config.batch_size, config.sequence_length, config.hidden_size).astype(np.float32),
              get_torch_mask(config, [data[mmask_tensor], data[smask_tensor]])]

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

    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX[mode],
                          transform=split_qkv)

    # Model to test against
    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs)


@pytest.mark.sanity
@pytest.mark.parametrize("momentum", [0.984375])
@pytest.mark.parametrize("mode", test_modes)
@pytest.mark.parametrize("batch_size, batch_serialisation_factor, number_attention_splits", [(2, 1, 1), (2, 2, 4)])
def test_attention_bwd_sanity(mode, momentum, batch_size, batch_serialisation_factor, number_attention_splits):
    attention_bwd(mode, momentum, batch_size, batch_serialisation_factor, number_attention_splits)


@pytest.mark.parametrize("momentum", [0.0, 0.984375])
@pytest.mark.parametrize("mode", test_modes)
@pytest.mark.parametrize("batch_size, batch_serialisation_factor, number_attention_splits", [(1, 1, 1), (2, 2, 1), (2, 2, 2), (4, 2, 4)])
def test_attention_bwd(mode, momentum, batch_size, batch_serialisation_factor, number_attention_splits):
    attention_bwd(mode, momentum, batch_size, batch_serialisation_factor, number_attention_splits)


def attention_bwd(mode, momentum, batch_size, batch_serialisation_factor, number_attention_splits):
    l1_lambda = 0.1
    num_reps = 5

    #  ------------------- PopART --------------------
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        batch_size=batch_size,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        num_attention_splits=number_attention_splits)
    popart_model = get_model(config, mode, 'attention')

    input_info = popart.TensorInfo(config.popart_dtype, [config.batch_size * config.sequence_length, config.hidden_size])
    input_tensor = popart_model.builder.addInputTensor(input_info)
    mask_info = popart.TensorInfo("UINT32", [config.batch_size, config.sequence_length])
    mmask_tensor = popart_model.builder.addInputTensor(mask_info)
    smask_tensor = popart_model.builder.addInputTensor(mask_info)
    data = {
        input_tensor: np.random.normal(0, 0.02, input_info.shape()).astype(config.dtype),
        mmask_tensor: np.random.randint(0, config.mask_tokens + 1, (config.batch_size, config.sequence_length,)).astype(np.uint32),
        smask_tensor: np.random.randint(config.mask_tokens, config.sequence_length + 1, (config.batch_size, config.sequence_length, )).astype(np.uint32)
    }

    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": batch_serialisation_factor,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(input_tensor, [mmask_tensor, smask_tensor])
        with popart_model.scope_provider(popart_model.builder, popart_model.norm.scope):
            l1 = popart_model.builder.aiGraphcore.l1loss(
                [output],
                l1_lambda,
                debugPrefix="l1LossVal",
                reduction=popart.ReductionType.Sum)
    else:
        user_options = {}
        output = popart_model.attention(input_tensor,
                                        [mmask_tensor, smask_tensor])
        l1 = popart_model.builder.aiGraphcore.l1loss(
            [output], l1_lambda, reduction=popart.ReductionType.Sum)

    proto = popart_model.builder.getModelProto()

    if momentum:
        optimizer = popart.SGD({"defaultLearningRate": (0.01, True),
                                "defaultMomentum": (momentum, True)})
    else:
        optimizer = popart.ConstSGD(0.01)

    outputs, post_proto = run_py(proto,
                                 data, (output, l1),
                                 loss=l1,
                                 optimizer=optimizer,
                                 num_reps=num_reps,
                                 user_options=user_options,
                                 execution_mode=mode)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[input_tensor].reshape(config.batch_size, config.sequence_length, config.hidden_size),
              get_torch_mask(config, [data[mmask_tensor], data[smask_tensor]])]

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

    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX[mode],
                          transform=split_qkv)

    optim = torch.optim.SGD(torch_model.parameters(), 0.01,
                            weight_decay=0.0, momentum=momentum)

    if momentum:
        for group in optim.param_groups:
            for p in group['params']:
                optim.state[p]['momentum_buffer'] = p.data * 0
                optim.state[p]['exp_avg'] = p.data * 0
                optim.state[p]['exp_avg_sq'] = p.data * 0
                optim.state[p]['step'] = 0

    for _ in range(num_reps):
        torch_output = torch_model(*[torch.from_numpy(t).float() for t in inputs])[0]
        torch_loss = l1_lambda * torch.norm(torch_output, 1)
        torch_loss.backward()
        optim.step()
        optim.zero_grad()

    check_tensors([torch_output.detach().numpy()], outputs)

    check_model(torch_model,
                post_proto,
                TORCH_TO_ONNX[mode],
                transform=split_qkv)
