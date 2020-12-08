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

import random
import numpy as np
import torch

import popart
import onnx

from bert_model import ExecutionMode
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model
from tests import torch_lamb


def get_mapping(config, init=None):
    if init is None:
        init = {}
    if config.execution_mode == ExecutionMode.DEFAULT:
        embedding_proj = {
            "bert.embeddings.word_embeddings.weight": "Embedding/Embedding_Dict",
            "bert.embeddings.position_embeddings.weight": "Embedding/Positional_Dict",
            "bert.embeddings.token_type_embeddings.weight": "Embedding/Segment_Dict",
            "bert.embeddings.LayerNorm.weight": "Embedding/Gamma",
            "bert.embeddings.LayerNorm.bias": "Embedding/Beta",
        }
        init.update(**embedding_proj)
        if config.split_qkv:
            for i in range(config.num_layers):
                layer = {
                    f"bert.encoder.layer.{i}.attention.self.query.weight": f"Layer{i}/Attention/Q",
                    f"bert.encoder.layer.{i}.attention.self.key.weight": f"Layer{i}/Attention/K",
                    f"bert.encoder.layer.{i}.attention.self.value.weight": f"Layer{i}/Attention/V",
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
        else:
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
    else:
        embedding_proj = {
            "bert.embeddings.word_embeddings.weight": "BertModel/Encoder/Embeddings/Token/weight",
            "bert.embeddings.position_embeddings.weight": "BertModel/Encoder/Embeddings/Position/weight",
            "bert.embeddings.token_type_embeddings.weight": "BertModel/Encoder/Embeddings/Segment/weight",
            "bert.embeddings.LayerNorm.weight": "BertModel/Encoder/Embeddings/Norm/Gamma",
            "bert.embeddings.LayerNorm.bias": "BertModel/Encoder/Embeddings/Norm/Beta",
        }
        init.update(**embedding_proj)
        if config.split_qkv:
            for i in range(config.num_layers):
                layer = {
                    f"bert.encoder.layer.{i}.attention.self.query.weight": f'BertModel/Encoder/Layer{i}/Attention/Q',
                    f"bert.encoder.layer.{i}.attention.self.key.weight": f'BertModel/Encoder/Layer{i}/Attention/K',
                    f"bert.encoder.layer.{i}.attention.self.value.weight": f'BertModel/Encoder/Layer{i}/Attention/V',
                    f"bert.encoder.layer.{i}.attention.output.dense.weight": f'BertModel/Encoder/Layer{i}/Attention/Out',
                    f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight": f'BertModel/Encoder/Layer{i}/Attention/Norm/Gamma',
                    f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias": f'BertModel/Encoder/Layer{i}/Attention/Norm/Beta',
                    f"bert.encoder.layer.{i}.intermediate.dense.weight": f'BertModel/Encoder/Layer{i}/FF/1/Dense/Weight',
                    f"bert.encoder.layer.{i}.intermediate.dense.bias": f'BertModel/Encoder/Layer{i}/FF/1/Dense/Bias',
                    f"bert.encoder.layer.{i}.output.dense.weight": f'BertModel/Encoder/Layer{i}/FF/2/Dense/Weight',
                    f"bert.encoder.layer.{i}.output.dense.bias": f'BertModel/Encoder/Layer{i}/FF/2/Dense/Bias',
                    f"bert.encoder.layer.{i}.output.LayerNorm.weight": f'BertModel/Encoder/Layer{i}/FF/Norm/Gamma',
                    f"bert.encoder.layer.{i}.output.LayerNorm.bias": f'BertModel/Encoder/Layer{i}/FF/Norm/Beta',
                }
                init.update(**layer)
        else:
            for i in range(config.num_layers):
                layer = {
                    f"bert.encoder.layer.{i}.attention.self.query.weight": f'BertModel/Encoder/Layer{i}/Attention/QKV',
                    f"bert.encoder.layer.{i}.attention.self.key.weight": f'BertModel/Encoder/Layer{i}/Attention/QKV',
                    f"bert.encoder.layer.{i}.attention.self.value.weight": f'BertModel/Encoder/Layer{i}/Attention/QKV',
                    f"bert.encoder.layer.{i}.attention.output.dense.weight": f'BertModel/Encoder/Layer{i}/Attention/Out',
                    f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight": f'BertModel/Encoder/Layer{i}/Attention/Norm/Gamma',
                    f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias": f'BertModel/Encoder/Layer{i}/Attention/Norm/Beta',
                    f"bert.encoder.layer.{i}.intermediate.dense.weight": f'BertModel/Encoder/Layer{i}/FF/1/Dense/Weight',
                    f"bert.encoder.layer.{i}.intermediate.dense.bias": f'BertModel/Encoder/Layer{i}/FF/1/Dense/Bias',
                    f"bert.encoder.layer.{i}.output.dense.weight": f'BertModel/Encoder/Layer{i}/FF/2/Dense/Weight',
                    f"bert.encoder.layer.{i}.output.dense.bias": f'BertModel/Encoder/Layer{i}/FF/2/Dense/Bias',
                    f"bert.encoder.layer.{i}.output.LayerNorm.weight": f'BertModel/Encoder/Layer{i}/FF/Norm/Gamma',
                    f"bert.encoder.layer.{i}.output.LayerNorm.bias": f'BertModel/Encoder/Layer{i}/FF/Norm/Beta',
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

    if config.split_qkv:
        for i in range(config.num_layers):
            layer = {
                f"bert.encoder.layer.{i}.attention.self.query.weight": np.transpose,
                f"bert.encoder.layer.{i}.attention.self.key.weight": np.transpose,
                f"bert.encoder.layer.{i}.attention.self.value.weight": np.transpose,
                f"bert.encoder.layer.{i}.attention.output.dense.weight": np.transpose,
                f"bert.encoder.layer.{i}.intermediate.dense.weight": np.transpose,
                f"bert.encoder.layer.{i}.output.dense.weight": np.transpose,
            }
            init.update(**layer)
    else:
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


def fwd_graph(popart_model, torch_model, mode, mapping=None, transform=None, replication_factor=1, replicated_weight_sharding = False):
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
            0, config.vocab_length, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32),
        positions: np.random.randint(
            0, config.sequence_length, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32),
        segments: np.random.randint(
            0, 2, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32)
    }


    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": 1,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(indices, positions, segments)
        ipus = 2
    else:
        output = popart_model.build_graph(indices, positions, segments)
        ipus = popart_model.total_ipus

    proto = builder.getModelProto()

    outputs, _ = run_py(proto,
                        data,
                        output,
                        user_options=user_options,
                        execution_mode=mode,
                        replication_factor=replication_factor,
                        replicated_weight_sharding=replicated_weight_sharding,
                        ipus=ipus)


    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = {
        "input_ids": data[indices].reshape(replication_factor * config.batch_size,
                                           config.sequence_length).astype(np.int32),
        "position_ids": data[positions].reshape(replication_factor * config.batch_size,
                                                config.sequence_length).astype(np.int32),
        "token_type_ids": data[segments].reshape(replication_factor * config.batch_size,
                                                 config.sequence_length).astype(np.int32)
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
              mode,
              popart_loss_fn,
              torch_loss_fn,
              mapping=None,
              transform=None,
              replication_factor=1,
              replicated_weight_sharding=False,
              opt_type="SGD"):
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
            0, config.vocab_length, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32),
        positions: np.random.randint(
            0, config.sequence_length, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32),
        segments: np.random.randint(
            0, 2, (replication_factor, config.batch_size * config.sequence_length)).astype(np.uint32)
    }
    num_reps = 5
    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": 1,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(indices, positions, segments)
        ipus = 2
    else:
        output = popart_model.build_graph(indices, positions, segments)
        ipus = popart_model.total_ipus

    loss = popart_loss_fn(output)

    proto = builder.getModelProto()

    if opt_type == "SGD":
        optimizer = popart.ConstSGD(1e-3)
    elif opt_type == "LAMB":
        optMap = {
            "defaultLearningRate": (1e-3, True),
            "defaultBeta1": (0.9, True),
            "defaultBeta2": (0.999, True),
            "defaultWeightDecay": (0.0, True),
            "maxWeightNorm": (10.0, True),
            "defaultEps": (1e-8, True),
            "lossScaling": (1.0, True),
        }
        optimizer = popart.Adam(optMap,
                                mode=popart.AdamMode.Lamb)
    elif opt_type == "LAMB_NO_BIAS":
        optMap = {
            "defaultLearningRate": (1, False),
            "defaultBeta1": (0, False),
            "defaultBeta2": (0, False),
            "defaultWeightDecay": (0.0, False),
            "defaultEps": (1e-8, False),
            "lossScaling": (1.0, False),
        }
        optimizer = popart.Adam(optMap,
                                mode=popart.AdamMode.LambNoBias)
    else:
        raise ValueError(f"Unknown opt_type={opt_type}")

    patterns = popart.Patterns()

    if mode == ExecutionMode.PHASED:
        patterns.enablePattern("TiedGatherPattern", False)
        patterns.enablePattern("SparseAccumulatePattern", False)

    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 loss=loss,
                                 optimizer=optimizer,
                                 user_options=user_options,
                                 execution_mode=mode,
                                 patterns=patterns,
                                 replication_factor=replication_factor,
                                 replicated_weight_sharding=replicated_weight_sharding,
                                 ipus=ipus,
                                 num_reps=num_reps)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = {
        "input_ids": data[indices].reshape(replication_factor * config.batch_size, config.sequence_length).astype(np.int32),
        "position_ids": data[positions].reshape(replication_factor * config.batch_size, config.sequence_length).astype(np.int32),
        "token_type_ids": data[segments].reshape(replication_factor * config.batch_size, config.sequence_length).astype(np.int32)
    }

    torch_to_onnx = get_mapping(config, init=mapping)

    transform_weights = get_transform(config, init=transform)

    #  ------------------- PyTorch -------------------------
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto,
                          torch_to_onnx, transform_weights)

    if opt_type == "SGD":
        optim = torch.optim.SGD(torch_model.parameters(), 1e-3,
                                weight_decay=0.0, momentum=0.0)
    elif opt_type == "LAMB":
        optim = torch_lamb.Lamb(torch_model.parameters(),
                                lr=1e-3, weight_decay=0.0, biasCorrection=True)

    for _ in range(num_reps):
        torch_outputs = torch_model(
            **{k: torch.from_numpy(t).long() for k, t in inputs.items()})
        torch_loss = torch_loss_fn(torch_outputs)
        torch_loss.backward()
        optim.step()
        optim.zero_grad()

    check_tensors([output.detach().numpy()
                   for output in torch_outputs], outputs, margin=1.5e-06)

    check_model(torch_model, post_proto,
                torch_to_onnx, transform_weights,
                margin=5e-5)
