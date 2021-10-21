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

import os
import numpy as np
import popart
import json
from logging import getLogger

from bert_model import BertConfig, Bert

logger = getLogger(__name__)


def get_tf_mapping(config, task="PRETRAINING"):

    squad_mapping = {
        "cls/squad/output_weights": "Squad/SquadW",
        "cls/squad/output_bias": "Squad/SquadB"
    }

    nsp_mapping = {
        "bert/pooler/dense/kernel": "NSP/PoolW",
        "bert/pooler/dense/bias": "NSP/PoolB",
        "cls/seq_relationship/output_weights": "NSP/NspW",
        "cls/seq_relationship/output_bias": "NSP/NspB"
    }

    lm_mapping = {
        "cls/predictions/transform/dense/kernel": "CLS/LMPredictionW",
        "cls/predictions/transform/dense/bias": "CLS/LMPredictionB",
        "cls/predictions/transform/LayerNorm/gamma": "CLS/Gamma",
        "cls/predictions/transform/LayerNorm/beta": "CLS/Beta"
    }

    tf_to_onnx = {
        "bert/embeddings/word_embeddings": "Embedding/Embedding_Dict",
        "bert/embeddings/position_embeddings": "Embedding/Positional_Dict",
        "bert/embeddings/token_type_embeddings": "Embedding/Segment_Dict",
        "bert/embeddings/LayerNorm/gamma": "Embedding/Gamma",
        "bert/embeddings/LayerNorm/beta": "Embedding/Beta"
    }
    for i in range(config.num_layers):
        layer = {
            f"bert/encoder/layer_{i}/attention/self/query/kernel": f"Layer{i}/Attention/QKV",
            f"bert/encoder/layer_{i}/attention/self/query/bias": f"Layer{i}/Attention/QKV_Bias",
            f"bert/encoder/layer_{i}/attention/self/key/kernel": f"Layer{i}/Attention/QKV",
            f"bert/encoder/layer_{i}/attention/self/key/bias": f"Layer{i}/Attention/QKV_Bias",
            f"bert/encoder/layer_{i}/attention/self/value/kernel": f"Layer{i}/Attention/QKV",
            f"bert/encoder/layer_{i}/attention/self/value/bias": f"Layer{i}/Attention/QKV_Bias",
            f"bert/encoder/layer_{i}/attention/output/dense/kernel": f"Layer{i}/Attention/Out",
            f"bert/encoder/layer_{i}/attention/output/dense/bias": f"Layer{i}/Attention/Out_Bias",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/gamma": f"Layer{i}/Attention/Gamma",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/beta": f"Layer{i}/Attention/Beta",
            f"bert/encoder/layer_{i}/intermediate/dense/kernel": f"Layer{i}/FF/1/W",
            f"bert/encoder/layer_{i}/intermediate/dense/bias": f"Layer{i}/FF/1/B",
            f"bert/encoder/layer_{i}/output/dense/kernel": f"Layer{i}/FF/2/W",
            f"bert/encoder/layer_{i}/output/dense/bias": f"Layer{i}/FF/2/B",
            f"bert/encoder/layer_{i}/output/LayerNorm/gamma": f"Layer{i}/FF/Gamma",
            f"bert/encoder/layer_{i}/output/LayerNorm/beta": f"Layer{i}/FF/Beta",
        }
        tf_to_onnx.update(**layer)

    if task == "PRETRAINING":
        tf_to_onnx.update(**lm_mapping)
        tf_to_onnx.update(**nsp_mapping)
    elif task == "SQUAD":
        tf_to_onnx.update(**squad_mapping)

    return tf_to_onnx


def get_tf_transform(task="PRETRAINING"):
    # Some of the head weights are stored transposed in the Google Research checkpoint
    # compared to the Popart model.
    tf_to_onnx_tform = {}
    if task == "PRETRAINING":
        tf_to_onnx_tform.update(**{
            "cls/seq_relationship/output_weights": np.transpose
        })
    elif task == "SQUAD":
        tf_to_onnx_tform.update(**{
            "cls/squad/output_weights": np.transpose
        })

    return tf_to_onnx_tform


def generate_initializers(config, map_names, load_data, mapping, transform={}):
    """
    Generate a graph initializer dictionary from the tensor names and
    data loaded from either a checkpoint or frozen graph using one of
    the methods below (`load_tf_ckpt_data` or `load_tf_frozen_data`).

    In the general case, this will simply map the tensor names from the
    TF model to the Popart model.

    The exception is the query-key-value matrix which is formed by
    concatenating the weight tensors Q, K and V.
    """
    initializers = {}
    qkv_tensor_range = {
        "query": (0, config.hidden_size),
        "key": (config.hidden_size, config.hidden_size * 2),
        "value": (config.hidden_size * 2, config.hidden_size * 3),
    }

    for name, array in zip(map_names, load_data):
        logger.debug(f"Initialising tensor from checkpoint {name} -> {mapping[name]}")

        if array.dtype == np.float32 and config.dtype == np.float16:
            array = array.astype(config.dtype)

        # If it's part of QKV, we need to handle separately as those 3
        # tensors need concatenating into one
        is_qkv = mapping[name][-3:] == "QKV"
        is_qkv_bias = mapping[name][-8:-5] == "QKV"
        if is_qkv or is_qkv_bias:
            qkv_part = name.split("/")[-2]

            if mapping[name] not in initializers.keys():
                if is_qkv:
                    qkv_shape = (array.shape[0], array.shape[1] * 3)
                elif is_qkv_bias:
                    qkv_shape = (array.shape[0] * 3)
                initializers[mapping[name]] = np.empty(
                    qkv_shape, dtype=array.dtype
                )

            start_idx = qkv_tensor_range[qkv_part][0]
            end_idx = qkv_tensor_range[qkv_part][1]
            if is_qkv:
                initializers[mapping[name]][:, start_idx:end_idx] = array
            elif is_qkv_bias:
                initializers[mapping[name]][start_idx:end_idx] = array
            logger.debug(f"Initialising QKV component {name}[{start_idx}:{end_idx}] from checkpoint")
            continue

        if name in transform:
            array = transform[name](array)

        if mapping[name] == "Embedding/Embedding_Dict":
            tf_vocab_length = array.shape[0]
            diff = config.vocab_length - tf_vocab_length
            # Pad or Crop the vocab.
            if diff > 0:
                logger.debug(f"Padding the vocabulary. From {tf_vocab_length} to {config.vocab_length}")
                pad = np.zeros((diff, config.hidden_size)).astype(array.dtype)
                array = np.concatenate((array, pad), axis=0)
            else:
                logger.warning(f"Cropping the vocabulary may negatively effect performance. From {tf_vocab_length} to {config.vocab_length}")
                array = np.array(array[:config.vocab_length, :])

        # FIXME: This copy is currently required to prevent popart misinterpreting the memory layout after the transpose.
        # Remove once T13187 is resolved.
        initializers[mapping[name]] = array.copy()
    return initializers


def load_tf_frozen_data(tf_frozen_path, mapping):
    """
    Parses a frozen-graph and outputs a tensors (lists of names and data) found
    in both the mapping and the checkpoint, ready for importing into the Bert
    model.
    """
    try:
        import tensorflow as tf
        from tensorflow.python.framework import tensor_util
    except ImportError:
        logger.error(
            "Loading a TensorFlow model requires TensorFlow to be installed. "
            "Please see https://www.tensorflow.org/install/ for installation "
            "instructions."
        )
        raise

    tf.reset_default_graph()
    with tf.io.gfile.GFile(tf_frozen_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We'll search the graphdef for the nodes containing data we need to import
    map_names = [n.name for n in graph_def.node if n.name in mapping.keys()]
    load_data = [
        tensor_util.MakeNdarray(n.attr["value"].tensor)
        for n in graph_def.node
        if n.name in mapping.keys()
    ]

    return map_names, load_data


def load_tf_ckpt_data(tf_checkpoint_path, mapping):
    """
    Parses a checkpoint file and outputs a tensors (lists of names and data)
    found in both the mapping and the checkpoint, ready for importing into the
    Bert model.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model requires TensorFlow to be installed. "
            "Please see https://www.tensorflow.org/install/ for installation "
            "instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)

    map_names = [name for name, shape in init_vars if name in mapping.keys()]
    for name in (n for n, _ in init_vars if n not in mapping.keys()):
        logger.debug(f"Skipping load of {name} - Not in mapping")

    load_data = [tf.train.load_variable(tf_path, name) for name in map_names]

    return map_names, load_data


def load_initializers_from_tf(
    file_path,
    is_checkpoint,
    config,
    task
):
    """
    Loads weights, etc. from Tensorflow files into a dictionary of Numpy Arrays.

    Can read either checkpoint files, or frozen graphs, according to the
    `is_checkpoint` flag, passed in as the second argument.
    """
    mapping = get_tf_mapping(config, task=task)
    transform = get_tf_transform(task=task)

    if is_checkpoint:
        names, data = load_tf_ckpt_data(file_path, mapping)
    else:
        names, data = load_tf_frozen_data(file_path, mapping)

    return generate_initializers(config, names, data, mapping, transform)


def load_model_from_tf(
    file_path,
    is_checkpoint,
    config,
    indices,
    positions,
    segments,
    task
):
    """
    Loads weights, etc. from Tensorflow files into the Graphcore IPU BERT
    implementation.

    Can read either checkpoint files, or frozen graphs, according to the
    `is_checkpoint` flag, passed in as the second argument.

    Requires input tensors to be provided to initialise the graph build.

    The user can optionally pass in a builder object (e.g. for compatibility
    with an older ONNX version). If not provided, a default builder is created.
    """
    initializers = load_initializers_from_tf(file_path, is_checkpoint, config, task)
    popart_model = Bert(config, initializers=initializers)

    output_tensor = popart_model.build_graph(indices, positions, segments)
    proto = builder.getModelProto()
    return popart_model, proto, output_tensor
