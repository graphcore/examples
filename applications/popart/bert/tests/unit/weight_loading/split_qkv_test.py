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

import os
import tempfile
from functools import partial
import numpy as np
import popart
import onnx

from bert_model import BertConfig, ExecutionMode, get_model
from utils import load_initializers_from_onnx
from tests.utils import check_onnx_model


def get_model_proto(config, initializers=None):
    model = get_model(config, ExecutionMode.PIPELINE, initializers=initializers)

    sequence_info = popart.TensorInfo("UINT32", [config.batch_size * config.sequence_length])
    indices = model.builder.addInputTensor(sequence_info)
    positions = model.builder.addInputTensor(sequence_info)
    segments = model.builder.addInputTensor(sequence_info)

    output = model.build_graph(indices, positions, segments)

    return onnx.load_model_from_string(model.builder.getModelProto())


def get_initializers(proto):
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "model.onnx")
        onnx.save(proto, file_path)
        return load_initializers_from_onnx(file_path)


def test_split_qkv_weight_loading():
    config = BertConfig(task="SQUAD",
                        vocab_length=1024,
                        batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        popart_dtype="FLOAT",
                        num_layers=2,
                        no_mask=True,
                        split_qkv=False)

    def get_split(full_t, t):
        return np.split(full_t, 3, axis=1)["QKV".index(t)]

    mapping = {f"Layer{i}/Attention/{t}": f"Layer{i}/Attention/QKV"
               for i in range(config.num_layers) for t in "QKV"}
    transform = {f"Layer{i}/Attention/{t}": partial(get_split, t=t)
                 for i in range(config.num_layers) for t in "QKV"}

    # Get a unsplit checkpoint
    np.random.seed(123)
    proto_1 = get_model_proto(config)
    initializers = get_initializers(proto_1)

    split_config = config._replace(split_qkv=True)

    # Load the unsplit checkpoint into a split model
    np.random.seed(456)
    proto_2 = get_model_proto(split_config, initializers=initializers)

    check_onnx_model(proto_1,
                     proto_2,
                     mapping,
                     transform,
                     allow_missing=False)

    # Extract weights
    initializers = get_initializers(proto_2)

    # Load the split checkpoint into an unsplit model
    np.random.seed(456)
    proto_3 = get_model_proto(config, initializers=initializers)

    check_onnx_model(proto_3,
                     proto_2,
                     mapping,
                     transform,
                     allow_missing=False)
