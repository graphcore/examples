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
import math
import os
import tempfile
from functools import partial

import numpy as np
import pytest

from tests.utils import check_onnx_model

import popart
import onnx
from bert_model import BertConfig, ExecutionMode, get_model
from phased_execution.weight_mapping import phased_to_default_mapping, phased_from_default_transform, get_phased_initializers_from_default
from utils import load_initializers_from_onnx


def get_model_proto(config, mode, initializers=None):
    model = get_model(config, mode, initializers=initializers)

    sequence_info = popart.TensorInfo("UINT32", [config.micro_batch_size * config.sequence_length])
    indices = model.builder.addInputTensor(sequence_info)
    positions = model.builder.addInputTensor(sequence_info)
    segments = model.builder.addInputTensor(sequence_info)

    if mode == ExecutionMode.PHASED:
        output = model(indices, positions, segments)
    else:
        output = model.build_graph(indices, positions, segments)
    return onnx.load_model_from_string(model.builder.getModelProto())


@pytest.mark.parametrize("task", ("SQUAD", "PRETRAINING"))
@pytest.mark.parametrize("num_vocab_splits", (1, 2))
def test_weight_mapping(num_vocab_splits, task):
    config = BertConfig(task=task,
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        mask_tokens=8,
                        popart_dtype="FLOAT",
                        num_layers=2,
                        no_mask=True,
                        no_dropout=True,
                        no_attn_dropout=True,
                        embedding_serialization_vocab_steps=num_vocab_splits,
                        inference=True)

    # Run pipelined BERT
    pipelined_proto = get_model_proto(config, mode=ExecutionMode.PIPELINE)

    # Extract weights
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "model.onnx")
        onnx.save(pipelined_proto, file_path)
        initializers = load_initializers_from_onnx(file_path)
        initializers.update(**get_phased_initializers_from_default(config, initializers))

    # Create phased_execution version of the model
    config_nosplit = config._replace(embedding_serialization_vocab_steps=1)
    phased_proto = get_model_proto(config, mode=ExecutionMode.PHASED, initializers=initializers)

    # Create a pipelined version of the model without any embedding split for the comparison
    pipelined_proto_nosplit = get_model_proto(config_nosplit, mode=ExecutionMode.PIPELINE, initializers=initializers)

    # Check inital protos match for pipelined vs phased_execution model
    check_onnx_model(pipelined_proto_nosplit,
                     phased_proto,
                     phased_to_default_mapping(config),
                     phased_from_default_transform(config),
                     allow_missing=False)


def test_trainable_params():
    config = BertConfig(task="PRETRAINING",
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        mask_tokens=8,
                        popart_dtype="FLOAT",
                        num_layers=2,
                        no_mask=True,
                        no_dropout=True,
                        no_attn_dropout=True,
                        embedding_serialization_vocab_steps=4,
                        inference=False)

    # Create phased_execution version of the model
    model = get_model(config, ExecutionMode.PHASED)
    data = {
        'indices': np.random.randint(
            0, config.vocab_length, (config.micro_batch_size * config.sequence_length)).astype(np.uint32),
        'positions': np.random.randint(
            0, config.sequence_length, (config.micro_batch_size * config.sequence_length)).astype(np.uint32),
        'segments': np.random.randint(
            0, 2, (config.micro_batch_size * config.sequence_length)).astype(np.uint32)
    }

    sequence_info = popart.TensorInfo("UINT32", [config.micro_batch_size * config.sequence_length])
    indices = model.builder.addInputTensor(sequence_info)
    positions = model.builder.addInputTensor(sequence_info)
    segments = model.builder.addInputTensor(sequence_info)

    data_popart = {}
    data_popart[indices] = data['indices']
    data_popart[segments] = data['segments']
    data_popart[positions] = data['positions']

    model(indices, positions, segments)
    proto = model.builder.getModelProto()

    # Extract weights from onnx model and check if same number of elements as self.tensors[0]
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.onnx")
        onnx.save(proto, model_path)
        onnx_model = onnx.load(model_path)
        assert len(model.tensors[0]) == len(onnx_model.graph.initializer)
