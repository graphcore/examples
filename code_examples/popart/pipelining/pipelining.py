# Copyright 2019 Graphcore Ltd.

'''
Code Example showing how to use pipelining in PopART on a very simple model
consisting of two dense layers. Run one pipeline length and compute loss.
'''

import numpy as np
import popart
import argparse

# ------------ Model Definition ---------------------------------------
#  <------------ ipu0 --------><---------- ipu1 --------------------->
#
#  d0 --|-- Gemm --|-- Relu --|-- Gemm--|-- Relu --><--Softmax --> Out
#  w0 --|                          w1 --|
#  b0 --|                          b1 --|


def create_pipelined_model(
        num_features,
        num_classes,
        batch_size):

    builder = popart.Builder()

    # Init
    def init_weights(input_size, output_size):
        return np.random.normal(
            0, 1, [input_size, output_size]).astype(np.float32)

    def init_biases(size):
        return np.random.normal(
            0, 1, [size]).astype(np.float32)

    # Labels
    labels_shape = [batch_size]
    labels = builder.addInputTensor(popart.TensorInfo("INT32", labels_shape))

    #  Input
    input_shape = [batch_size, num_features]
    x0 = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape))

    #  Dense 1
    W0 = builder.addInitializedInputTensor(
        init_weights(num_features, 512))
    b0 = builder.addInitializedInputTensor(init_biases(512))

    with builder.virtualGraph(0):
        x1 = builder.aiOnnx.gemm([x0, W0, b0], debugPrefix="gemm_x1")
        x2 = builder.aiOnnx.relu([x1], debugPrefix="relu_x2")

    #  Dense 2
    W1 = builder.addInitializedInputTensor(init_weights(512, num_classes))
    b1 = builder.addInitializedInputTensor(init_biases(num_classes))

    with builder.virtualGraph(1):
        x3 = builder.aiOnnx.gemm([x2, W1, b1], debugPrefix="gemm_x3")
        x4 = builder.aiOnnx.relu([x3], debugPrefix="relu_x4")

    # Outputs
    with builder.virtualGraph(1):
        output_probs = builder.aiOnnx.softmax(
            [x4], axis=1, debugPrefix="softmax_output")

    builder.addOutputTensor(output_probs)

    # Loss
    loss = popart.NllLoss(output_probs, labels, "loss")
    loss.virtualGraph(1)

    # Anchors
    art = popart.AnchorReturnType("ALL")
    anchor_map = {"loss": art}
    anchor_map[popart.reservedGradientPrefix() + x0] = art

    # Protobuffer
    model_proto = builder.getModelProto()

    return x0, labels, model_proto, anchor_map, loss


def main(args):

    # Model parameters
    np.random.seed(1971)
    input_rows = 28
    input_columns = 28
    num_classes = 10
    batch_size = 8
    input_shape = [batch_size, input_rows * input_columns]
    labels_shape = [batch_size]

    # Create model
    x0, labels, model_proto, anchor_map, loss = create_pipelined_model(
        num_features=input_columns * input_rows,
        num_classes=num_classes,
        batch_size=batch_size)

    # Save model (optional)
    if args.export:
        with open(args.export, 'wb') as model_path:
            model_path.write(model_proto)

    # Session options
    opts = popart.SessionOptions()
    opts.enablePipelining = False if args.no_pipelining else True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.engineOptions = {"debug.instrument": "true"}
    pipeline_depth = 64
    num_ipus = 2

    # Create session
    session = popart.TrainingSession(
        fnModel=model_proto,
        dataFeed=popart.DataFlow(pipeline_depth, anchor_map),
        losses=[loss],
        optimizer=popart.ConstSGD(0.01),
        userOptions=opts,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(num_ipus))

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    # Extra data feed for pipeline
    if pipeline_depth > 1:
        labels_shape.insert(0, pipeline_depth)
        input_shape.insert(0, pipeline_depth)

    # Synthetic data input
    data_in = np.random.uniform(
        low=-10.0, high=10.0, size=input_shape).astype(np.float32)

    classes = np.prod(input_shape) / (batch_size * pipeline_depth)

    labels_in = np.random.randint(
        low=0, high=classes, size=labels_shape).astype(np.int32)

    # Run session
    inputs = {x0: data_in, labels: labels_in}
    stepio = popart.PyStepIO(inputs, anchors)
    session.weightsFromHost()
    session.optimizerFromHost()
    session.run(stepio)

    # Save report and return session object (optional)
    if args.report:
        from gcprofile import save_popart_report
        save_popart_report(session)
    if args.test:
        return session


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', help='export model', metavar='FILE')
    parser.add_argument('--report', action='store_true',
                        help='save execution report')
    parser.add_argument('--no_pipelining', action='store_true',
                        help='deactivate pipelining')
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    # Run
    main(args)
