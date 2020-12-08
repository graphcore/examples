# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

from collections import defaultdict
import re
import numpy as np
import onnx


def load_initializers_from_onnx(model_path):
    """Load initial weights from an onnx checkpoint.

    Args:
        model_path (str): Path to onnx file.

    Returns:
        Dict: Mapping of popart weight names to numpy values.
    """
    initializers = {}
    model = onnx.load(model_path)
    for weight in model.graph.initializer:
        if weight.data_type == onnx.TensorProto.FLOAT16:
            int_data = np.asarray(weight.int32_data, np.int32)
            np_weight = int_data.view(dtype=np.float16).reshape(weight.dims)
        else:
            np_weight = onnx.numpy_helper.to_array(weight)
        initializers[weight.name] = np_weight

    initializers = handle_split_qkv(initializers)
    initializers = handle_split_word_embedding(initializers)

    return initializers


def handle_split_qkv(initializers):
    split_qkv = [
        re.compile(f"Layer([0-9]+)/Attention/({t})$") for t in "QKV"]

    layer_splits = defaultdict(dict)

    for weight in initializers.keys():
        for regex in split_qkv:
            m = regex.match(weight)
            if m:
                layer_splits[m.group(1)][m.group(2)] = initializers[weight]

    for i, layer in layer_splits.items():
        if set(layer.keys()) != {"Q", "K", "V"}:
            raise RuntimeError(f"Could not find Q, K and V in the initializers. Only {layers.keys()}")
        initializers[f"Layer{i}/Attention/QKV"] = np.concatenate([layer["Q"], layer["K"], layer["V"]], axis=1)

    return initializers


def handle_split_word_embedding(initializers):
    split_rgx = re.compile("Embedding/Embedding_Dict/split([0-9]+)$")

    layer_splits = {}

    for weight in initializers.keys():
        m = split_rgx.match(weight)
        if m:
            layer_splits[int(m.group(1))] = initializers[weight]

    if layer_splits:
        initializers["Embedding/Embedding_Dict"] = np.concatenate([layer_splits[i] for i in sorted(layer_splits)], axis=0)

    return initializers
