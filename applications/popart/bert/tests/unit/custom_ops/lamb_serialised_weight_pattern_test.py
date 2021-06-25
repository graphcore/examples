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

import numpy as np
import json
import pytest
import popart
from tests.utils import run_py, check_tensors, check_onnx_model


def model(splits=1):
    np.random.seed(1984)
    input_data = np.random.rand(4, 20).astype(np.float32)
    weight_data = np.random.rand(20, 8).astype(np.float32)

    builder = popart.Builder()

    d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', input_data.shape), 'data0')

    w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
    x = builder.aiOnnx.matmul([d0, w0])
    if splits > 1:
        builder.setSerializeMatMul(
            {x}, 'output_channels', splits, True)
    loss = builder.aiGraphcore.l1loss([x], 0.1, debugContext='loss')

    return builder.getModelProto(), {d0: input_data}, x, loss


def session(splits=1):
    proto, data, x, loss = model(splits)

    user_options = {
        "enableOutlining": False,
        "enableGradientAccumulation": True,
        "accumulationFactor": 2,
        "optimizerStateTensorLocationSettings": popart.TensorLocationSettings(
            popart.TensorStorage.OffChip, 0)
    }

    optimizer = popart.Adam({
        "defaultLearningRate": (0.1, True),
        "defaultBeta1": (0.1, True),
        "defaultBeta2": (0.1, True)
    }, mode=popart.AdamMode.LambNoBias)  # NoBias to increase the error of incorrect gradients

    return run_py(
        proto,
        data=data,
        outputs=x,
        loss=loss,
        optimizer=optimizer,
        patterns=popart.Patterns(),
        user_options=user_options,
        skip_execution=False)


@pytest.mark.sanity
@pytest.mark.requires_remote_buffers
@pytest.mark.parametrize('splits', (2, 4))
def test_lamb_serialised_pattern_correctness(splits, custom_ops):
    outputs_1, proto_1 = session(splits=1)
    outputs_2, proto_2 = session(splits=splits)

    check_tensors(outputs_1, outputs_2)
    check_onnx_model(proto_1, proto_2)
