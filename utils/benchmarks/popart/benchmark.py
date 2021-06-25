# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse
import time
import json
import popart
import json
import numpy as np
from collections import namedtuple

Benchmark = namedtuple(
    'Benchmark', [
        'graph_builder',     # opts -> proto,data,outputs,losses,optimizer
        'add_args',          # parser -> parser
        'iteration_report',  # duration,opts -> string
    ]
)
Benchmark.__new__.__defaults__ = (lambda parser: parser, lambda *_: "")


def run(benchmark, opts):
    proto, data, outputs, losses, optimizer = benchmark.graph_builder(opts)

    if opts.save_graph:
        with open('model.onnx', "wb") as f:
            f.write(proto)
            print("Written to file: model.onnx")

    dataFlow = popart.DataFlow(opts.batches_per_step, outputs)

    # Create a session to compile and execute the graph
    options = popart.SessionOptions()
    if not opts.use_generated_data:
        options.syntheticDataMode = popart.SyntheticDataMode.Zeros
    options.instrumentWithHardwareCycleCounter = opts.report_hw_cycle_count
    options.engineOptions = {
        "debug.instrumentCompute": "true" if opts.report else "false"
    }
    if opts.convolution_options:
        options.convolutionOptions = json.loads(opts.convolution_options)

    if opts.shards > 1:
        if opts.auto_sharding:
            options.virtualGraphMode = popart.VirtualGraphMode.Auto
        else:
            options.virtualGraphMode = popart.VirtualGraphMode.Manual

    options.enablePipelining = opts.pipeline

    if opts.recompute:
        if opts.pipeline:
            options.autoRecomputation = popart.RecomputationType.Pipeline
        else:
            options.autoRecomputation = popart.RecomputationType.Standard

    # Select a device
    deviceManager = popart.DeviceManager()
    if opts.simulation:
        deviceOptions = {"compileIPUCode": True,
                         'numIPUs': opts.shards, "tilesPerIPU": 1216}
        device = deviceManager.createIpuModelDevice(deviceOptions)
    else:
        device = deviceManager.acquireAvailableDevice(opts.shards)
        if device is None:
            raise OSError("Failed to acquire IPU.")

    if opts.mode == 'train':
        session = popart.TrainingSession(fnModel=proto,
                                         loss=losses,
                                         deviceInfo=device,
                                         optimizer=optimizer,
                                         dataFlow=dataFlow,
                                         userOptions=options)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=dataFlow,
                                          userOptions=options)

    print("Compiling...")
    start = time.time()
    session.prepareDevice()
    compilation_duration = time.time() - start
    print("Duration: {:.3f} seconds\n".format(compilation_duration))

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    # Copy weights and optimization parameters onto the device
    session.weightsFromHost()

    # Add a batches_per_step dimension if needed
    if opts.batches_per_step > 1:
        data = {k: np.repeat(v[np.newaxis], opts.batches_per_step, 0)
                for k, v in data.items()}

    stepio = popart.PyStepIO(data, anchors)

    print("Executing...")
    average_batches_per_sec = 0
    # Steps
    for __ in range(opts.steps):
        # Run
        start = time.time()
        session.run(stepio)
        duration = time.time() - start

        if opts.report:
            return save_reports(opts, session)

        average_batches_per_sec += (opts.batches_per_step /
                                    duration)/opts.steps
        report_string = "{:<8.3} sec/itr.".format(duration)
        report_string += "   " + benchmark.iteration_report(opts, duration)
        print(report_string)

    if opts.report_hw_cycle_count:
        print("Hardware cycle count per 'run':", session.getCycleCount())

    return compilation_duration, average_batches_per_sec


def parse_opts(benchmark, arg_string=None):
    parser = argparse.ArgumentParser(
        description='Synthetic Benchmarks in Popart', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Default Arguments
    parser.add_argument('--mode', choices=["infer", "eval", "train"], default='infer',
                        help='Which graph to run: infer/eval/train')
    parser.add_argument('--use-generated-data', action="store_true",
                        help="Add data transfer ops. Models execution with IO but unbounded by the CPU pipeline.")
    parser.add_argument('--report', action="store_true",
                        help="Generate Graph and Execution Reports")
    parser.add_argument('--batches-per-step', type=int, default=1,
                        help="Number of batches to run per step (on the device)")
    parser.add_argument('--steps', type=int, default=1,
                        help="Number of steps to run (on the host)")
    parser.add_argument('--convolution-options', type=str,
                        help='Set convolution options as a JSON string.')
    parser.add_argument('--shards', type=int, default=1,
                        help="Select a number of IPUs to split across")
    parser.add_argument('--auto-sharding', action="store_true",
                        help="Use auto sharding")
    parser.add_argument('--pipeline', action="store_true",
                        help="Pipeline the model over 'shards' IPUs")
    parser.add_argument('--recompute', action="store_true", default=False,
                        help="Enable recomputations of activations in backward pass")
    parser.add_argument('--simulation', action="store_true",
                        help="Run the program on the IPU Model")
    parser.add_argument('--save-graph', action="store_true",
                        help="Save default graph to model.onnx")
    parser.add_argument('--use-zero-values', action="store_true",
                        help="If True weights and input will be initialised to zeros (otherwise random data)")
    parser.add_argument('--report-hw-cycle-count', type=bool, default=False,
                        help='Report the number of cycles a "run" takes.')
    # Benchmark Arguments
    benchmark.add_args(parser)

    opts = parser.parse_args(arg_string)

    if opts.report:
        opts.batches_per_step = 1

    # Should change this to a dictonary
    return opts


def save_reports(opts, session):
    with open("graph.json", "wb") as f:
        f.write(session.getGraphReport())
    with open("execution.json", "wb") as f:
        f.write(session.getExecutionReport())
    print("Written to: graph.json, execution.json")
