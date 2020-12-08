# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

'''
Code Example showing how to use recomputing in PopART
on a simple model consisting of seven dense layers.
See https://arxiv.org/abs/1604.06174
'''

import numpy as np
import popart
import argparse

# Model
#
#   Dense 1 - 512
#   Dense 2 - 512
#
#   Dense 3 - 2048 (checkpointed)
#   Dense 4 - 512
#
#   Dense 5 - 128  (checkpointed)
#   Dense 6 - 128
#   Dense 7 - 10
#   Softmax


def create_model(num_features, num_classes, batch_size, force_recompute=False):

    builder = popart.Builder()

    # Init
    def init_weights(input_size, output_size):
        return np.random.normal(0, 1,
                                [input_size, output_size]).astype(np.float32)

    def init_biases(size):
        return np.random.normal(0, 1, [size]).astype(np.float32)

    def dense(x, input_size, hidden_size, recompute=False, suffix=""):
        W = builder.addInitializedInputTensor(
            init_weights(input_size, hidden_size))
        b = builder.addInitializedInputTensor(init_biases(hidden_size))
        x = builder.aiOnnx.gemm([x, W, b], debugPrefix="gemm_" + suffix)
        if recompute:
            builder.recomputeOutputInBackwardPass(x)
        x = builder.aiOnnx.relu([x], debugPrefix="relu_" + suffix)
        if recompute:
            builder.recomputeOutputInBackwardPass(x)
        return x

    # Labels
    labels_shape = [batch_size]
    labels = builder.addInputTensor(popart.TensorInfo("INT32", labels_shape))

    #  Input
    input_shape = [batch_size, num_features]
    x0 = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape))

    #  Dense layers
    x = dense(x0, num_features, 512, force_recompute, "1")
    x = dense(x, 512, 512, force_recompute, "2")
    x = dense(x, 512, 2048, False, "3")  # Checkpointed
    x = dense(x, 2048, 512, force_recompute, "4")
    x = dense(x, 512, 128, False, "5")  # Checkpointed
    x = dense(x, 128, 128, force_recompute, "6")
    out = dense(x, 128, num_classes, force_recompute, "7")

    # Outputs
    output_probs = builder.aiOnnx.softmax([out],
                                          axis=1,
                                          debugPrefix="softmax_output")

    builder.addOutputTensor(output_probs)

    # Loss
    loss = builder.aiGraphcore.nllloss([output_probs, labels], popart.ReductionType.Sum, debugPrefix="loss")

    # Anchors
    art = popart.AnchorReturnType("ALL")
    anchor_map = {loss: art}
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
    batch_size = 512
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
        dataFlow=popart.DataFlow(1, anchor_map),
        loss=loss,
        optimizer=popart.ConstSGD(0.01),
        userOptions=opts,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(num_ipus))

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    # Synthetic data input
    data_in = np.random.uniform(low=0.0, high=1.0,
                                size=input_shape).astype(np.float32)

    labels_in = np.random.randint(low=0, high=num_classes,
                                  size=labels_shape).astype(np.int32)

    # Run session
    inputs = {x0: data_in, labels: labels_in}
    stepio = popart.PyStepIO(inputs, anchors)
    session.weightsFromHost()
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
    parser.add_argument('--report',
                        action='store_true',
                        help='save execution report')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--recomputing',
                        help='deactivate recompute',
                        metavar='STATUS',
                        default='ON')  # ON/AUTO/OFF see README
    parser.add_argument('--show-logs',
                        help='show execution logs',
                        action='store_true')
    args = parser.parse_args()

    # (Optional) Logs
    if args.show_logs:
        popart.getLogger().setLevel("DEBUG")

    # Run
    main(args)
