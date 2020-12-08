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
import torch
import popart
import pytest
import onnx

from bert_model import BertConfig, Bert, ExecutionMode, get_model
from tests.torch_bert import BertConfig as TorchBertConfig, BertEmbeddings
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model, TestFailureError, requires_remote_buffers


'''
Tests the embedding with no projection. This is the case for SQUAD.
'''

TORCH_TO_ONNX = {
    ExecutionMode.DEFAULT: {
        "word_embeddings.weight": "Embedding/Embedding_Dict",
        "position_embeddings.weight": "Embedding/Positional_Dict",
        "token_type_embeddings.weight": "Embedding/Segment_Dict",
        "LayerNorm.weight": "Embedding/Gamma",
        "LayerNorm.bias": "Embedding/Beta"
    },
    ExecutionMode.PHASED: {
        "word_embeddings.weight": "Embeddings/Token/weight",
        "position_embeddings.weight": "Embeddings/Position/weight",
        "token_type_embeddings.weight": "Embeddings/Segment/weight",
        "LayerNorm.weight": "Embeddings/Norm/Gamma",
        "LayerNorm.bias": "Embeddings/Norm/Beta"
    },
}


def setup_function(func):
    np.random.seed(1984)


def serialized_embedding_mapping(args, mode):
    mapping = {}
    if args.embedding_serialization_vocab_steps > 1:
        if mode == ExecutionMode.PHASED:
            mapping['Embeddings/Token/weight'] = []
            for i in range(args.embedding_serialization_vocab_steps):
                mapping['Embeddings/Token/weight'].append(f'Embeddings/Token/split{i}/weight')
        else:
            mapping['Embedding/Embedding_Dict'] = []
            for i in range(args.embedding_serialization_vocab_steps):
                mapping['Embedding/Embedding_Dict'].append(f'Embedding/Embedding_Dict/split{i}')
    return mapping


def serialized_embedding_transform(args, mode, split_model_prefix='', default_model_prefix=''):
    transform = {}

    # Handle serialisation of layers
    vocab_splits = args.embedding_serialization_vocab_steps
    if vocab_splits > 1:
        def get_concatenate(splits):
            if isinstance(splits, list):
                concat_axis = -1
                for idx in range(len(splits[0].shape)):
                    if splits[0].shape[idx] * vocab_splits == args.vocab_length:
                        if concat_axis != -1:
                            # Error : more than one axis meets the vocab axis criteria. need code change to handle this
                            raise RuntimeError("Weight transform error :more than one axis meets the vocab axis criteria")
                        else:
                            concat_axis = idx
                if(concat_axis >= 0):
                    return np.concatenate(splits, axis=concat_axis)
                else:
                    raise RuntimeError("Weight transform error :could not identify vocab axis")
            else:
                # Error this transform function requires a list of split vocabs to be concatenated
                raise RuntimeError("Weight concatenate transform requires a list of split vocabs to be concatenated")
        if mode == ExecutionMode.PHASED:
            transform[default_model_prefix+"Embeddings/Token/weight"] = get_concatenate
        else:
            transform[default_model_prefix+"Embedding/Embedding_Dict"] = get_concatenate
    return transform


# Expand the map to have list of names to handle embedding serializations in phased mode
def expand_torch_to_onnx_map(torch_to_onnx, args, mode):
    mapping = serialized_embedding_mapping(args, mode)
    transform_map = serialized_embedding_transform(args, mode)
    expanded_name_map = {}
    remapped_transform_map = {}
    for k, v in torch_to_onnx.items():
        if v in mapping.keys():
            expanded_name_map[k] = mapping[v]
        else:
            expanded_name_map[k] = v
        if v in transform_map.keys():
            remapped_transform_map[k] = transform_map[v]
    return expanded_name_map, remapped_transform_map


@pytest.mark.parametrize("mode", [ExecutionMode.DEFAULT, requires_remote_buffers(ExecutionMode.PHASED)])
@pytest.mark.parametrize("batch_size, batch_serialization_factor, embedding_serialization_vocab_steps", [(1, 1, 1), (2, 2, 1), (2, 2, 2), (2, 1, 1), (2, 1, 2)])
def test_embedding_fwd(custom_ops, mode, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        batch_size=batch_size,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        inference=True,
                        embedding_serialization_vocab_steps=embedding_serialization_vocab_steps)
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
            "batchSerializationFactor": batch_serialization_factor,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(indices, positions, segments)
    else:
        user_options = {"enableStochasticRounding": True}
        with popart_model.builder.nameScope("Embedding"):
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

    expanded_name_map, remapped_transform_map = expand_torch_to_onnx_map(TORCH_TO_ONNX[mode], config, mode)
    copy_weights_to_torch(torch_model, proto, expanded_name_map, remapped_transform_map)

    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs, margin=5e-7)


@pytest.mark.parametrize("momentum", [0.0, 0.984375])
@pytest.mark.parametrize("mode, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps",
                         [requires_remote_buffers(ExecutionMode.PHASED, 1, 1, 1),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 2, 1),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 2, 2),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 1, 1),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 1, 2),
                          (ExecutionMode.DEFAULT, 1, 1, 1),
                          (ExecutionMode.DEFAULT, 2, 1, 1),
                          (ExecutionMode.DEFAULT, 2, 1, 2)
                          ])
def test_embedding_bwd(custom_ops, mode, momentum, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps):
    embedding_bwd(custom_ops, mode, momentum, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps)


@pytest.mark.sanity
@pytest.mark.parametrize("momentum", [0.984375])
@pytest.mark.parametrize("mode, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps",
                         [requires_remote_buffers(ExecutionMode.PHASED, 2, 1, 1),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 1, 2),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, 2, 2),
                          (ExecutionMode.DEFAULT, 2, 1, 1),
                          (ExecutionMode.DEFAULT, 2, 1, 2)
                          ])
def test_embedding_bwd_sanity(custom_ops, mode, momentum, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps):
    embedding_bwd(custom_ops, mode, momentum, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps, 9728, 768)


def embedding_bwd(custom_ops, mode, momentum, batch_size, batch_serialization_factor, embedding_serialization_vocab_steps, vocab_length=9728, hidden_size=768):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=vocab_length,
                        batch_size=batch_size,
                        hidden_size=hidden_size,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        update_embedding_dict=True,
                        embedding_serialization_vocab_steps=embedding_serialization_vocab_steps)

    popart_model = get_model(config, mode, 'embedding')
    # Prevent virtualGraph attributes being added to the ops

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

    if momentum:
        optimizer = popart.SGD({"defaultLearningRate": (0.01, True),
                                "defaultMomentum": (momentum, True),
                                "defaultDampening": (0.0, True),
                                "defaultVelocityScaling": (1.0, True),
                                "lossScaling": (1.0, True),
                                "defaultWeightDecay": (0.0, True)})
    else:
        optimizer = popart.ConstSGD(0.01)

    l1_lambda = 0.1

    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": batch_serialization_factor,
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
        with popart_model.builder.nameScope("Embedding"):
            output = popart_model.embedding(indices, positions, segments)
        l1 = popart_model.builder.aiGraphcore.l1loss(
            [output],
            l1_lambda,
            debugPrefix="l1LossVal",
            reduction=popart.ReductionType.Sum)

    num_reps = 5
    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 ipus=1,
                                 loss=l1,
                                 num_reps=num_reps,
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
                        layer_norm_eps=config.layer_norm_eps,
                        update_embedding_dict=config.update_embedding_dict))
    # Turn off dropout
    torch_model.eval()

    expanded_name_map, remapped_transform_map = expand_torch_to_onnx_map(TORCH_TO_ONNX[mode], config, mode)
    copy_weights_to_torch(torch_model, proto, expanded_name_map, remapped_transform_map)

    optim = torch.optim.SGD(torch_model.parameters(),
                            0.01,
                            weight_decay=0.0,
                            dampening=0.0,
                            momentum=momentum)

    if momentum > 0.:
        for group in optim.param_groups:
            for p in group['params']:
                optim.state[p]['momentum_buffer'] = p.data * 0
                optim.state[p]['exp_avg'] = p.data * 0
                optim.state[p]['exp_avg_sq'] = p.data * 0
                optim.state[p]['step'] = 0

    for _ in range(num_reps):
        torch_output = torch_model(*[torch.from_numpy(t).long() for t in inputs])
        torch_loss = l1_lambda * torch.norm(torch_output, 1)
        torch_loss.backward()
        optim.step()
        optim.zero_grad()

    torch_outputs = [torch_output.detach().numpy()]

    check_tensors(torch_outputs, outputs, margin=7e-6)

    expanded_name_map, remapped_transform_map = expand_torch_to_onnx_map(TORCH_TO_ONNX[mode], config, mode)
    check_model(torch_model, post_proto, expanded_name_map, remapped_transform_map, margin=7e-06)
