# Copyright 2020 Graphcore Ltd.
import numpy as np
import json
import pytest
import popart
from tests.utils import run_py


def model():
    np.random.seed(1984)
    input_data = np.random.rand(20, 20).astype(np.float32)

    builder = popart.Builder()

    d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', input_data.shape), 'data0')

    w0 = builder.addInitializedInputTensor(input_data, 'weight0')

    x = builder.aiOnnx.matmul([d0, w0])
    builder.recomputeOutputInBackwardPass(x)

    return builder.getModelProto(), {d0: input_data}, x


def session(skip_execution=False, include_patterns=True, momentum=False):
    proto, data, x = model()
    # Required
    patterns = ["MatMulOp", "MatMulLhsGradOp", "MatMulRhsGradOp", "OpToIdentity", "PreUniRepl", "PostNRepl", "InPlace"]
    if include_patterns:
        patterns += ["InplaceWorkaroundPattern"]
    optimizer = popart.ConstSGD(0.1)
    if momentum:
        optimizer = popart.SGD({
            "defaultLearningRate": (0.1, True),
            "defaultMomentum": (0.9, True)})
    return run_py(
        proto,
        data=data,
        outputs=x,
        loss=popart.L1Loss(x, 'loss', 0.1),
        optimizer=optimizer,
        patterns=popart.Patterns(patterns),
        user_options={
            "enableOutlining": False
        },
        skip_execution=skip_execution
    )


@pytest.mark.parametrize(['optim'], [(optim,) for optim in ("sgd0", "sgd1")])
def test_inplace_workaround_pattern_ir(optim, custom_ops):
    '''
    Tests that the InplaceWorkaroundPattern inplaces the reshape op added by the MatMulOp pattern without effecting the final structure/result of the graph.
    '''
    momentum = optim == "sgd1"

    sess_1 = session(True, True, momentum)
    sess_2 = session(True, False, momentum)

    ir_1 = json.loads(
        sess_1._serializeIr(popart.IrSerializationFormat.JSON))

    ir_2 = json.loads(
        sess_2._serializeIr(popart.IrSerializationFormat.JSON))

    def consumes_stream_or_weight(op):
        inputs = [i["name"] for i in op["inputs"]]
        return 'weight0' in inputs or 'data0' in inputs

    for op_1, op_2 in zip(ir_1["maingraph"], ir_2["maingraph"]):
        # Reshape has been inplaced or equal
        if op_2["type"] == "Reshape" and consumes_stream_or_weight(op_2):
            assert op_1["type"] == "ReshapeInplace"
        else:
            assert op_1["type"] == op_2["type"]
