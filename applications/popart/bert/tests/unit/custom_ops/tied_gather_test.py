# Copyright 2020 Graphcore Ltd.
import numpy as np
import json
import pytest
import popart
from tests.utils import run_py, check_tensors, check_onnx_model


def model(splits=1):
    np.random.seed(1984)
    input_data = np.random.randint(0, 20, (4,)).astype(np.uint32)
    weight_data = np.random.rand(4, 20).astype(np.float32)

    builder = popart.Builder()

    d0 = builder.addInputTensor(popart.TensorInfo('UINT32', input_data.shape), 'data0')

    w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
    w0_t = builder.aiOnnx.transpose([w0])

    x = builder.aiOnnx.gather([w0_t, d0])

    x = builder.aiOnnx.matmul([x, w0])
    if splits > 1:
        builder.setSerializeMatMul(
            {x}, 'output_channels', splits, True)

    return builder.getModelProto(), {d0: input_data}, x


def session(train=False, skip_execution=False, include_patterns=True, splits=1, outline=False):
    proto, data, x = model(splits=splits)
    # Required
    patterns = ["MatMulOp", "MatMulLhsGradOp", "MatMulRhsGradOp", "OpToIdentity", "PreUniRepl"]
    if include_patterns:
        patterns += ["TiedGatherPattern", "TiedGatherGradPattern"]
    if train:
        return run_py(
            proto,
            data=data,
            outputs=x,
            loss=popart.L1Loss(x, 'loss', 0.1),
            optimizer=popart.SGD({
                "defaultLearningRate": (0.1, True),
                "defaultMomentum": (0.9, True),
                "defaultDampening": (0, True)}),  # 0 dampening to increase the error of incorrect gradients
            patterns=popart.Patterns(patterns),
            user_options={
                "enableOutlining": outline
            },
            skip_execution=skip_execution)
    else:
        return run_py(
            proto,
            data=data,
            outputs=x,
            patterns=popart.Patterns(patterns),
            user_options={
                "enableOutlining": outline,
                "constantWeights": False
            },
            skip_execution=skip_execution)


@pytest.mark.parametrize(['phase', 'splits'], [(phase, splits) for phase in ["fwd", "bwd"] for splits in (1, 4)])
def test_tied_gather_pattern_ir(phase, splits, custom_ops):
    train = phase == "bwd"

    sess = session(train, skip_execution=True, splits=splits)

    ir = json.loads(
        sess._serializeIr(popart.IrSerializationFormat.JSON))

    ops = ir["maingraph"]

    # The gatherOp should be replaced with TiedGather
    assert len(list(filter(lambda op: op["type"] == "TiedGather", ops))) == splits
    assert len(list(filter(lambda op: op["type"] == "Gather", ops))) == 0

    # The matmuls should have fully_connected_pass disabled
    assert all(map(lambda op: op["attributes"]["fully_connected_pass"] == '-1',
                   filter(lambda op: op["type"] == "MatMul",
                          ir["maingraph"])))

    if train:
        assert len(list(filter(lambda op: op["type"] == "SparseSGD1Accumulate", ops))) == splits


@pytest.mark.parametrize(['phase', 'splits'], [(phase, splits) for phase in ["fwd", "bwd"] for splits in (1, 4)])
def test_tied_gather_pattern_correctness(phase, splits, custom_ops):
    train = phase == "bwd"

    outputs_1, proto_1 = session(train, skip_execution=False, splits=splits)

    outputs_2, proto_2 = session(train, skip_execution=False, include_patterns=False, splits=splits)

    check_tensors(outputs_1, outputs_2)
    if train:
        check_onnx_model(proto_1, proto_2)


@pytest.mark.parametrize(['phase'], [(phase,) for phase in ["fwd", "bwd"]])
def test_tied_gather_pattern_outlining_correctness(phase, custom_ops):
    train = phase == "bwd"

    outputs_1, proto_1 = session(train, skip_execution=False, splits=4, outline=True)

    outputs_2, proto_2 = session(train, skip_execution=False, include_patterns=False, splits=4, outline=True)

    check_tensors(outputs_1, outputs_2)
    if train:
        check_onnx_model(proto_1, proto_2)
