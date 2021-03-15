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
from tests.utils import run_py


@pytest.mark.sanity
def test_outline_dropout_pattern_one(custom_ops):
    '''
    Tests that the OutlineDropoutPattern successfully outlines all 3 dropouts (fwd, bwd) into a single subgraph
    Expected IR Graph (excluding adds etc)
    fwd...
        x = add(data0, weight0)
        0_seed = seedModify(seed, 0)
        x = call_0(x, 0_seed)
        1_seed = seedModify(seed, 1)
        x = call_0(x, 1_seed)
        2_seed = seedModify(seed, 2)
        x = call_0(x, 2_seed)
    bwd...
        x = call_0(x, 0_seed)
        x = call_0(x, 1_seed)
        x = call_0(x, 2_seed)

        where call_0(x, seed) = dropout(x, seed)
    '''

    input_data = np.random.rand(2, 2).astype(np.float32)

    builder = popart.Builder()

    d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', input_data.shape), 'data0')

    w0 = builder.addInitializedInputTensor(input_data, 'weight0')

    x = builder.aiOnnx.add([d0, w0])

    x = builder.aiOnnx.dropout([x], 1)[0]

    x = builder.aiOnnx.dropout([x], 1)[0]

    x = builder.aiOnnx.dropout([x], 1)[0]
    loss = builder.aiGraphcore.l1loss([x], 0.1, debugPrefix='loss')

    patterns = popart.Patterns()
    patterns.enablePattern("OutlineDropoutPattern", True)
    patterns.enablePattern("PostNRepl", True)

    session = run_py(
        builder.getModelProto(),
        data={d0: input_data},
        outputs=x,
        loss=loss,
        optimizer=popart.ConstSGD(0.1),
        patterns=patterns,
        user_options={
            "outlineThreshold": -1
        },
        skip_execution=True
    )

    ir = json.loads(
        session._serializeIr(popart.IrSerializationFormat.JSON))

    # There should only be a main graph and 1 subgraph containing dropout
    assert len(ir.keys()) == 2

    ops = [o["type"] for o in ir["call_subgraph(0)"]]
    assert "Dropout" in ops

    ops = [o["type"] for o in ir["maingraph"]]
    # Should only be 1 seed modify per dropout
    assert len(list(filter(lambda op: op == "SeedModify", ops))) == 6
    # The bwd and fwd should be outlined together
    assert len(list(filter(lambda op: op == "Call", ops))) == 6


@pytest.mark.sanity
def test_outline_dropout_pattern_many(custom_ops):
    '''
    Tests that the OutlineDropoutPattern successfully outlines all 3 dropouts (fwd, bwd) into a 3 different subgraphs.
    Expected IR Graph (excluding adds etc)
    fwd...
        x = add(data0, weight0)
        0_seed = seedModify(seed, 0)
        x = call_0(x, 0_seed)
        1_seed = seedModify(seed, 1)
        x = call_1(x, 1_seed)
        2_seed = seedModify(seed, 2)
        x = call_2(x, 2_seed)
    bwd...
        x = call_2(x, 0_seed)
        x = call_1(x, 1_seed)
        x = call_0(x, 2_seed)

        where call_n(x, seed) = dropout(x, seed)
    '''

    input_data = np.random.rand(2, 2).astype(np.float32)

    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})

    d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', input_data.shape), 'data0')

    w0 = builder.addInitializedInputTensor(input_data, 'weight0')

    x = builder.aiOnnx.add([d0, w0])

    x = builder.aiOnnx.dropout([x], 1)[0]

    # Different subgraph as it has a different ratio
    x = builder.aiOnnx.dropout([x], 1, ratio=0.8)[0]

    # Different subgraph as it has a different input shape
    x = builder.aiOnnx.slice([x], axes=[1], starts=[0], ends=[1])

    x = builder.aiOnnx.dropout([x], 1)[0]
    loss = builder.aiGraphcore.l1loss([x], 0.1, debugPrefix='loss')

    patterns = popart.Patterns(popart.PatternsLevel.Minimal)
    patterns.enablePattern("OutlineDropoutPattern", True)
    patterns.enablePattern("PostNRepl", True)

    session = run_py(
        builder.getModelProto(),
        data={d0: input_data},
        outputs=x,
        loss=loss,
        optimizer=popart.ConstSGD(0.1),
        patterns=patterns,
        user_options={
            "outlineThreshold": -np.inf
        },
        skip_execution=True
    )

    ir = json.loads(
        session._serializeIr(popart.IrSerializationFormat.JSON))

    # There should only be a main graph and 3 subgraph containing dropout
    assert len(ir.keys()) == 4
    ops = [o["type"] for i in range(3) for o in ir[f"call_subgraph({i})"]]
    assert "Dropout" in ops

    ops = [o["type"] for o in ir["maingraph"]]
    # Should only be 1 seed modify per dropout
    assert len(list(filter(lambda op: op == "SeedModify", ops))) == 6
    # The bwd and fwd should be outlined together
    assert len(list(filter(lambda op: op == "Call", ops))) == 6
