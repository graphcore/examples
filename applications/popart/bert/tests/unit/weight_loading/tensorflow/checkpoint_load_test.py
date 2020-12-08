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

import pytest
import os
import ctypes
import numpy as np
import popart
from tests.torch_bert import (
    BertConfig as TorchBertConfig,
    BertForMaskedLM as TorchModelPreTraining,
    BertForQuestionAnswering as TorchModelSquad,
    load_tf_weights_in_bert,
)
import random
import torch
import math
import onnx
from bert_model import BertConfig, Bert
from bert_tf_loader import load_model_from_tf
from tests.utils import run_fwd_model, run_py, check_tensors, check_model
from tests.unit.pytorch.full_graph_utils import get_mapping, get_transform

np.random.seed(1984)
random.seed(1984)
torch.manual_seed(1984)


def load_bert_config_tf(config_path, override_vocab=None, chkpt_task="PRETRAINING"):
    """
    Load the bert config data from Google Research's checkpoint format
    into the Popart Bert config format.
    """
    import json
    with open(config_path, "r") as fh:
        config_data = json.load(fh)

    config = BertConfig(
        vocab_length=config_data["vocab_size"] if override_vocab is None else override_vocab,
        hidden_size=config_data["hidden_size"],
        sequence_length=config_data["max_position_embeddings"],
        max_positional_length=config_data["max_position_embeddings"],
        ff_size__=config_data["intermediate_size"],
        attention_heads=config_data["num_attention_heads"],
        num_layers=config_data["num_hidden_layers"],
        # TODO: Read the rest of these in from a GC config?
        embedding_serialization_vocab_steps=4,
        batch_size=1,
        popart_dtype="FLOAT",
        no_dropout=True,
        inference=True,
        activation_type="relu",
        task=chkpt_task
    )

    return config


def run_models(config, proto, indices, positions, segments, output, popart_model, torch_model):
    onnx_proto = onnx.load_model_from_string(proto)
    check_model(torch_model, onnx_proto, get_mapping(config), get_transform(config))

    # Run the models
    popart_inputs = {
        indices: np.random.randint(
            0, config.vocab_length,
            (config.batch_size * config.sequence_length)
        ).astype(np.uint32),
        positions: np.random.randint(
            0,
            config.sequence_length,
            (config.batch_size * config.sequence_length),
        ).astype(np.uint32),
        segments: np.random.randint(
            0,
            2,
            (config.batch_size * config.sequence_length),
        ).astype(np.uint32),
    }

    popart_outputs, post_proto = run_py(
        proto,
        popart_inputs,
        output,
        ipus=popart_model.total_ipus,
    )

    torch_inputs = {
        "input_ids": popart_inputs[indices].reshape(
            config.batch_size, config.sequence_length
        ),
        "position_ids": popart_inputs[positions].reshape(
            config.batch_size, config.sequence_length
        ),
        "token_type_ids": popart_inputs[segments].reshape(
            config.batch_size, config.sequence_length
        ),
    }

    torch_model.eval()
    torch_outputs = run_fwd_model(torch_inputs, torch_model)

    check_model(torch_model, post_proto, get_mapping(config), get_transform(config))
    check_tensors(torch_outputs, popart_outputs)
    print("Test succeeded")


@pytest.mark.requires_frozen
@pytest.mark.requires_config
@pytest.mark.requires_chkpt
def test_load_from_frozen(config_path, chkpt_path, chkpt_task, frozen_path, custom_ops):
    # Vocab-size override is not required, but allows the test to run more quickly
    config = load_bert_config_tf(config_path, override_vocab=9728, chkpt_task=chkpt_task)

    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}
    )

    # Load Torch version
    TorchModel = TorchModelPreTraining if chkpt_task == "PRETRAINING" else TorchModelSquad
    torch_model = TorchModel(
        TorchBertConfig(
            config.vocab_length,
            config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.attention_heads,
            intermediate_size=config.ff_size,
            hidden_act="relu",
            max_position_embeddings=config.max_positional_length,
            layer_norm_eps=config.layer_norm_eps,
            mask_tokens=config.mask_tokens,
        )
    )

    torch_model = load_tf_weights_in_bert(torch_model, config, chkpt_path)

    # Load Popart model
    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])

    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)

    popart_model, proto, output = load_model_from_tf(
        frozen_path, False, config, indices, positions, segments, chkpt_task, builder=builder
    )

    run_models(config, proto, indices, positions, segments, output, popart_model, torch_model)


@pytest.mark.requires_config
@pytest.mark.requires_chkpt
def test_load_from_chkpt(config_path, chkpt_path, chkpt_task, custom_ops):
    """
    Compare the model loaded into our popart model against the modified
    PyTorch model:
        - Load tf weights into BERT using torch impl -> run fwd model
        - Load tf weights into BERT using popart impl -> run fwd model
        - Compare output tensors
    """
    # Vocab-size override is not required, but allows the test to run more quickly
    config = load_bert_config_tf(config_path, override_vocab=9728, chkpt_task=chkpt_task)

    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}
    )

    # Load Torch version
    TorchModel = TorchModelPreTraining if chkpt_task == "PRETRAINING" else TorchModelSquad
    torch_model = TorchModel(
        TorchBertConfig(
            config.vocab_length,
            config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.attention_heads,
            intermediate_size=config.ff_size,
            hidden_act="relu",
            max_position_embeddings=config.max_positional_length,
            layer_norm_eps=config.layer_norm_eps,
            mask_tokens=config.mask_tokens,
        )
    )

    torch_model = load_tf_weights_in_bert(torch_model, config, chkpt_path)

    # Load Popart model
    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])

    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    segments = builder.addInputTensor(sequence_info)

    popart_model, proto, output = load_model_from_tf(
        chkpt_path, True, config, indices, positions, segments, chkpt_task, builder=builder
    )

    run_models(config, proto, indices, positions, segments, output, popart_model, torch_model)
