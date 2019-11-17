# Copyright 2019 Graphcore Ltd.
import pytest
import os
import ctypes
import numpy as np
import popart
from tests.torch_bert import (
    BertConfig as TorchBertConfig,
    BertForMaskedLM as TorchModel,
    load_tf_weights_in_bert,
)
import math
from bert_model import BertConfig, Bert
from bert_tf_loader import load_model_from_tf, load_bert_config_tf
from tests.utils import run_fwd_model, run_py, check_tensors


@pytest.mark.requires_frozen
@pytest.mark.requires_config
@pytest.mark.requires_chkpt
def test_load_from_frozen(config_path, chkpt_path, frozen_path, custom_ops):
    config = load_bert_config_tf(config_path)

    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}
    )

    # Load Torch version
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

    torch_model.eval()
    torch_model = load_tf_weights_in_bert(torch_model, config, chkpt_path)

    # Load Popart model
    sequence_info = popart.TensorInfo(
        "INT32", [config.batch_size * config.sequence_length])

    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)

    popart_model, proto, output = load_from_tf(
        frozen_path, False, config, indices, positions, builder=builder
    )

    # Run the models
    popart_inputs = {
        indices: np.random.randint(
            0, config.vocab_length,
            (config.batch_size * config.sequence_length)
        ).astype(np.int32),
        positions: np.random.randint(
            0,
            config.sequence_length,
            (config.batch_size * config.sequence_length),
        ).astype(np.int32),
    }

    torch_inputs = {
        "input_ids": popart_inputs[indices].reshape(
            config.batch_size, config.sequence_length
        ),
        "position_ids": popart_inputs[positions].reshape(
            config.batch_size, config.sequence_length
        ),
    }

    torch_outputs = run_fwd_model(torch_inputs, torch_model)

    popart_outputs, post_proto = run_py(
        proto,
        popart_inputs,
        output,
        ipus=math.ceil(config.num_layers / config.layers_per_ipu) + 1,
    )

    check_tensors(torch_outputs, popart_outputs)
    print("Test succeeded")


@pytest.mark.requires_config
@pytest.mark.requires_chkpt
def test_load_from_chkpt(config_path, chkpt_path, custom_ops):
    """
    Compare the model loaded into our popart model against the modified
    PyTorch model:
        - Load tf weights into BERT using torch impl -> run fwd model
        - Load tf weights into BERT using popart impl -> run fwd model
        - Compare output tensors
    """
    config = load_bert_config_tf(config_path)

    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}
    )

    # Load Torch version
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

    torch_model.eval()
    torch_model = load_tf_weights_in_bert(torch_model, config, chkpt_path)

    # Load Popart model
    sequence_info = popart.TensorInfo(
        "INT32", [config.batch_size * config.sequence_length])

    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)

    popart_model, proto, output = load_from_tf(
        chkpt_path, True, config, indices, positions, builder=builder
    )

    # Run the models
    popart_inputs = {
        indices: np.random.randint(
            0, config.vocab_length,
            (config.batch_size * config.sequence_length)
        ).astype(np.int32),
        positions: np.random.randint(
            0,
            config.sequence_length,
            (config.batch_size * config.sequence_length),
        ).astype(np.int32),
    }

    torch_inputs = {
        "input_ids": popart_inputs[indices].reshape(
            config.batch_size, config.sequence_length
        ),
        "position_ids": popart_inputs[positions].reshape(
            config.batch_size, config.sequence_length
        ),
    }

    torch_outputs = run_fwd_model(torch_inputs, torch_model)

    popart_outputs, post_proto = run_py(
        proto,
        popart_inputs,
        output,
        ipus=math.ceil(config.num_layers / config.layers_per_ipu) + 1,
    )

    check_tensors(torch_outputs, popart_outputs)
    print("Test succeeded")
