# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time

import numpy as np
import popart
import argparse


class PerfIntervalTimer:
    # Define a simple timer object:
    def __init__(self):
        self.time = None

    def not_set(self):
        return self.time is None

    def last(self):
        return self.time

    def reset(self):
        self.time = time.perf_counter()

    def interval(self):
        now = time.perf_counter()
        interval = now - self.time
        return interval


# Define a function to build and run the graph with
# the specified data size:
def build_and_run_graph(data_size):
    # Create a builder object:
    builder = popart.Builder()

    # Specify two input vectors:
    data_spec = popart.TensorInfo("FLOAT", [data_size])
    id_a = builder.addInputTensor(data_spec)
    id_b = builder.addInputTensor(data_spec)

    # Describe the computation:
    o1 = builder.aiOnnx.add([id_a, id_b])
    o2 = builder.aiOnnx.mul([id_a, id_b])

    # Designate the two output vectors and how
    # often the result will be required:
    builder.addOutputTensor(o1)
    builder.addOutputTensor(o2)
    data_flow = popart.DataFlow(
        1, {o1: popart.AnchorReturnType("ALL"), o2: popart.AnchorReturnType("ALL")}
    )

    # Setup an inference graph:
    proto = builder.getModelProto()
    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=data_flow,
        deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
    )

    # Compile graph:
    session.prepareDevice()

    # Create input data buffers:
    data_a = np.random.rand(data_size).astype(np.float32)
    data_b = np.random.rand(data_size).astype(np.float32)
    inputs = {id_a: data_a, id_b: data_b}

    # Create output data buffers:
    anchors = session.initAnchorArrays()

    # Create timer objects and dictionaries:
    timer = PerfIntervalTimer()
    rtts = {}

    # Input callback is called when the data is needed:
    def input_callback(id, is_prefetch: bool):
        if is_prefetch:
            return

        if timer.not_set():
            timer.reset()
        return inputs[id]

    # Called after the input buffer has been consumed:
    def input_complete_callback(id):
        return

    # Output callback is called when a buffer is needed for the result:
    def output_callback(id):
        return anchors[id]

    # Complete callback is called when the output buffer has
    # been filled (result is ready to be consumed by the host):
    def output_complete_callback(id):
        rtt = timer.interval()
        rtts[id] = rtt

    # Create the callback IO system:
    stepio = popart.PyStepIOCallback(
        input_callback,
        input_complete_callback,
        output_callback,
        output_complete_callback,
    )

    # Run the graph and return timings:
    session.run(stepio, 'AddAndMulCallback')
    print(rtts)
    return rtts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Callbacks in PopART",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-size", help="Set the data size", type=int, default=1000)
    args = parser.parse_args()
    build_and_run_graph(args.data_size)
