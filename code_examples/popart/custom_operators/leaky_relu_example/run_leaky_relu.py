# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import ctypes
import os

import numpy as np
import popart


# Define a function to build and run the leaky relu graph with
# specified input tensor data and alpha value
def build_and_run_graph(input_data, alpha, run_on_ipu):
    builder = popart.Builder()
    input_len = len(input_data)

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [input_len]))

    output_tensor = builder.customOp(opName="LeakyRelu",
                                     opVersion=1,
                                     domain="custom.ops",
                                     inputs=[input_tensor],
                                     attributes={"alpha": alpha})[0]

    builder.addOutputTensor(output_tensor)

    proto = builder.getModelProto()

    anchors = {output_tensor: popart.AnchorReturnType("FINAL")}
    dataFlow = popart.DataFlow(1, anchors)

    if run_on_ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
        print("IPU hardware device acquired")
    else:
        device = popart.DeviceManager().createIpuModelDevice({})
        print("Running on IPU Model")

    print("alpha={}".format(alpha))

    session = popart.InferenceSession(proto, dataFlow, device)

    session.prepareDevice()
    result = session.initAnchorArrays()

    X = (np.array(input_data)).astype(np.float32)
    print("X={}".format(X))

    stepio = popart.PyStepIO({input_tensor: X},
                             result)
    session.run(stepio, 'LeakyReLU')

    return result


def load_custom_ops_lib():
    so_path = os.path.join(os.path.dirname(__file__),
                           "build/custom_ops.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


if __name__ == '__main__':
    load_custom_ops_lib()

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="sets the lrelu alpha attribute", type=float,
                        default=0.02)
    parser.add_argument("--ipu", help="run on available IPU hardware device",
                        action='store_true')
    parser.add_argument('input_data', metavar='X', type=float, nargs='+',
                        help='input tensor data')

    args = parser.parse_args()

    result = build_and_run_graph(args.input_data, args.alpha, args.ipu)

    print("RESULT X")
    print(result)
