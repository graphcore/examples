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
import popart
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
from onnx.onnx_pb import TensorProto


def is_external_weight(weight):
    return weight.HasField("data_location") and weight.data_location == TensorProto.EXTERNAL


def load_initializers_from_onnx(model_path, load_optimizer=False):
    """Load initial weights from an onnx checkpoint.

    Args:
        model_path (str): Path to onnx file.

    Returns:
        Dict: Mapping of popart weight names to numpy values.
    """
    initializers = {}
    # By default onnx.load will look for initializers in the same dir as onnx model.
    # However builder.saveIntializersExternally takes real path or path relative to run dir
    # and stores it in the onnxproto.
    model = onnx.load(model_path, load_external_data=False)

    has_external_data = any(is_external_weight(weight) for weight in model.graph.initializer)
    if has_external_data:
        load_external_data_for_model(model, '')

    optimizer_prefix = (popart.reservedAccl1Prefix(),
                        popart.reservedAccl2Prefix(),
                        popart.reservedAcclPrefix(),
                        popart.reservedAccumPrefix(),
                        popart.reservedStepPrefix())

    for weight in model.graph.initializer:
        is_optimizer_state = any(x in weight.name for x in optimizer_prefix)
        if not load_optimizer and is_optimizer_state:
            continue

        if is_external_weight(weight) or weight.data_type != onnx.TensorProto.FLOAT16:
            np_weight = numpy_helper.to_array(weight)
        else:
            int_data = np.asarray(weight.int32_data, np.int32)
            np_weight = int_data.view(dtype=np.float16).reshape(weight.dims)

        if is_optimizer_state:
            initializers[weight.name] = np_weight.astype(np.float32)
        else:
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
            raise RuntimeError(f"Could not find Q, K and V in the initializers. Only {layer.keys()}")
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
