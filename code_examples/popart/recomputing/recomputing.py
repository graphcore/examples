# Copyright 2019 Graphcore Ltd.

'''
Code Example showing how to use recomputing in PopART
on a very simple model consisting of four dense layers.
'''

import numpy as np
import popart
import argparse


# Model
#  < -------------------- ipu0 -------------------------------------------->
#  x0 | Gemm| Relu | Gemm | Relu | Gemm | Relu | Gemm | Relu | Softmax | Out
#  w0 |                w1 |         w2  |         w3  |
#  b0 |                b1 |         b2  |         b3  |


def create_model(
        num_features,
        num_classes,
        batch_size,
        force_recompute=False):

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
    x = builder.aiOnnx.gemm([x0, W0, b0], debugPrefix="gemm_1")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)
    x = builder.aiOnnx.relu([x], debugPrefix="relu_1")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)

    #  Dense 2
    W1 = builder.addInitializedInputTensor(init_weights(512, 512))
    b1 = builder.addInitializedInputTensor(init_biases(512))
    x = builder.aiOnnx.gemm([x, W1, b1], debugPrefix="gemm_2")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)
    x = builder.aiOnnx.relu([x], debugPrefix="relu_2")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)

    #  Dense 3
    W2 = builder.addInitializedInputTensor(init_weights(512, 512))
    b2 = builder.addInitializedInputTensor(init_biases(512))
    x = builder.aiOnnx.gemm([x, W2, b2], debugPrefix="gemm_3")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)
    x = builder.aiOnnx.relu([x], debugPrefix="relu_3")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)

    #  Dense 4
    W3 = builder.addInitializedInputTensor(init_weights(512, num_classes))
    b3 = builder.addInitializedInputTensor(init_biases(num_classes))
    x = builder.aiOnnx.gemm([x, W3, b3], debugPrefix="gemm_4")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(x)
    out = builder.aiOnnx.relu([x], debugPrefix="relu_4")
    if force_recompute:
        builder.recomputeOutputInBackwardPass(out)

    # Outputs
    output_probs = builder.aiOnnx.softmax(
        [out], axis=1, debugPrefix="softmax_output")

    builder.addOutputTensor(output_probs)

    # Loss
    loss = popart.NllLoss(output_probs, labels, "loss")

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
    batch_size = 2048
    input_shape = [batch_size, input_rows * input_columns]
    labels_shape = [batch_size]

    # Create model
    x0, labels, model_proto, anchor_map, loss = create_model(
        num_features=input_columns * input_rows,
        num_classes=num_classes,
        batch_size=batch_size,
        force_recompute=True if args.recomputing == 'ON' else False)

    # Save model (optional)
    if args.export:
        with open(args.export, 'wb') as model_path:
            model_path.write(model_proto)

    # Session options
    num_ipus = 1
    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.engineOptions = {"debug.instrument": "true"}

    if args.recomputing == 'AUTO':
        opts.autoRecomputation = popart.RecomputationType.Standard

    # Create session
    session = popart.TrainingSession(
        fnModel=model_proto,
        dataFeed=popart.DataFlow(1, anchor_map),
        losses=[loss],
        optimizer=popart.ConstSGD(0.01),
        userOptions=opts,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(num_ipus))

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    # Synthetic data input
    data_in = np.random.uniform(
        low=0.0, high=1.0, size=input_shape).astype(np.float32)

    labels_in = np.random.randint(
        low=0, high=num_classes, size=labels_shape).astype(np.int32)

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
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument(
        '--recomputing',
        help='deactivate recompute',
        metavar='STATUS',
        default='ON')
    parser.add_argument('--show-logs', help='show execution logs', action='store_true')
    args = parser.parse_args()

    # (Optional) Logs
    if args.show_logs:
        popart.getLogger().setLevel("DEBUG")

    # Run
    main(args)
