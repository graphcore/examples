#!/usr/bin/env python
# Copyright 2019 Graphcore Ltd.
'''
Benchmarks a TensorFlow graph.

Usage: benchmark.py <benchmark_import_name> [-h --options]
'''
import tensorflow as tf
import copy
import time
import os
import re
import json
import argparse
from collections import namedtuple

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils, loops, ipu_infeed_queue, ipu_compiler

Benchmark = namedtuple(
    'Benchmark', [
        'graph_builder',     # opts,inputs -> [fetches]
        'inputs',            # opts,index -> inputs. This maps index -> tensors to be consumed by graph_builder
        'initializer',       # -> [fetches]
        'add_args',          # parser -> parser
        'iteration_report',  # opts,duration -> string
    ]
)
Benchmark.__new__.__defaults__ = (lambda parser: parser, lambda *_: "")


def run(benchmark, opts):
    '''
    Run the benchmark.

    benchmark - An instance of Benchmark
    opts - Namespace from argparse generated from parse_opts
    '''
    with ipu_scope('/device:IPU:0'):
        # Build graph
        with tf.device('cpu'):
            dataset = tf.data.Dataset \
                .range((opts.steps + 2) * opts.batches_per_step) \
                .map(lambda i: benchmark.inputs(opts, i)) \
                .repeat() \
                .prefetch(opts.batches_per_step)

        if opts.batches_per_step > 1:
            with tf.device('cpu'):
                infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="benchmark_dataset_infeed")
                data_init = infeed_queue.initializer

            with tf.Graph().as_default():  # To get the shape and dtype
                dummy_opts = copy.deepcopy(opts)
                dummy_opts.shards = 1
                d = benchmark.inputs(dummy_opts, tf.constant(0))
                out = benchmark.graph_builder(dummy_opts, d)
            input = tf.constant(0, out.dtype, shape=out.shape)

            def body(inout, *args, **kwargs):
                with tf.control_dependencies([inout]):
                    # Run graph
                    with tf.variable_scope("MainGraph"):
                        out = benchmark.graph_builder(opts, kwargs)
                return out

            out = ipu_compiler.compile(lambda: loops.repeat(opts.batches_per_step,
                                                            body,
                                                            [input],
                                                            infeed_queue), [])
        else:
            with tf.device('cpu'):
                data_tensors = dataset.make_one_shot_iterator().get_next()
                data_init = tf.no_op()
            out = ipu_compiler.compile(lambda: benchmark.graph_builder(opts, data_tensors), [])
            opts.batches_per_step = 1

    # Report
    report = gen_ipu_ops.ipu_event_trace()

    # Dump the graph to a logdir
    if opts.save_graph:
        writer = tf.summary.FileWriter(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', time.strftime('%Y%m%d_%H%M%S_%Z')))
        writer.add_graph(tf.get_default_graph())

    utils.configure_ipu_system(get_config(opts))
    with tf.Session() as sess:
        # Setup
        sess.run([benchmark.initializer(), data_init])
        sess.run(report)

        # Warmup
        print("Compiling and Warmup...")
        start = time.time()
        sess.run(out)
        duration = time.time() - start
        print("Duration: {:.3f} seconds\n".format(duration))

        # Cycle Report
        if opts.cycle_report:
            rep = sess.run(report)
            return extract_runtimes_from_report(rep, opts, display=True)  # Only run once if producing cycle report

        print("Executing...")
        average_batches_per_sec = 0
        # steps
        for i in range(opts.steps):
            # Run
            start = time.time()
            sess.run(out)
            duration = time.time() - start

            average_batches_per_sec += (opts.batches_per_step/duration)/opts.steps
            report_string = "{:<7.3} sec/itr.".format(duration)
            report_string += "   " + benchmark.iteration_report(opts, duration)
            print(report_string)

        return average_batches_per_sec

# --------- UTILS --------------


def parse_opts(benchmark, arg_string=None):
    parser = argparse.ArgumentParser(description='Synthetic Benchmarks in TensorFlow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Default Arguments
    parser.add_argument('--use-data', action="store_true",
                        help="Add data transfer ops. Models execution with IO but unbounded by the CPU pipeline.")
    parser.add_argument('--cycle-report', action="store_true",
                        help="Generate Cycle Report when run on hardware")
    parser.add_argument('--batches-per-step', type=int, default=1,
                        help="Number of batches to run per step (on the device)")
    parser.add_argument('--steps', type=int, default=1,
                        help="Number of steps to run (on the host)")
    parser.add_argument('--save-graph', action="store_true",
                        help="Save default graph to 'logs' directory (used by TensorBoard)")
    parser.add_argument('--tile-activity-report', action="store_true",
                        help="Save a tile activity csv")
    parser.add_argument('--convolution-options', type=str,
                        help='Set convolution options as a JSON string.')
    parser.add_argument('--shards', type=int, default=0,
                        help="Select a number of IPUs to split across")
    parser.add_argument('--device-id', type=int, default=-1,
                        help="Select a device")
    parser.add_argument('--use-zero-values', action="store_true",
                        help="If True weights and input will be initialised to zeros (otherwise random data)")
    # Benchmark Arguments
    benchmark.add_args(parser)

    opts = parser.parse_args()

    if not opts.use_data:
        if opts.use_zero_values:
            initType = "0"
        else:
            initType = "random"
        if "TF_POPLAR_FLAGS" in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += " --use_synthetic_data --synthetic_data_initializer=" + initType
        else:
            os.environ["TF_POPLAR_FLAGS"] = "--use_synthetic_data --synthetic_data_initializer=" + initType

    if opts.tile_activity_report:
        opts.cycle_report = True

    if opts.cycle_report:
        opts.batches_per_step = 1

    # Should change this to a dictionary
    return opts


def get_config(opts):
    """Builds ipu_options"""
    profile = opts.cycle_report

    config = utils.create_ipu_config(profiling=profile,
                                     profile_execution=profile,
                                     report_every_nth_execution=1)
    if opts.device_id == -1:
        config = utils.auto_select_ipus(config, [opts.shards or 1])
    else:
        config = utils.select_ipus(config, [opts.device_id])

    if opts.convolution_options:
        config = utils.set_convolution_options(config, json.loads(opts.convolution_options))
    return config


def extract_runtimes_from_report(raw_report, opts, start_time=0, display=True):
    """Returns timing information from IpuTraceEvent

    report -- Array of text encoded IpuTraceEvent

    """
    if len(raw_report) is 0:
        return

    # Timings from tf xla event timestamps
    from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent

    events = list(map(IpuTraceEvent.FromString, raw_report))
    first = start_time == 0
    if first:
        start_time = events[0].timestamp
        events = events[1:]

    # Retrieve IpuEvents, poplar report and cycles
    if display:
        evt_str = "\nIPU Timings\n"
        compile_started = False

        for evt in events:
            extra_str = ""
            if evt.type == IpuTraceEvent.COMPILE_BEGIN:
                compile_started = True
                evt_name = "Compile start"
            elif evt.type == IpuTraceEvent.COMPILE_END:
                if compile_started:
                    compile_started = False
                    evt_name = "Compile"
                else:
                    continue
            elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
                evt_name = "Host->Device"
                extra_str = "\n  Tensors:"
                transfered_tensors = json.loads(evt.data_transfer.data_transfer.decode('utf-8'))
                for t in transfered_tensors["tensors"]:
                    extra_str += "\n    handle: {:>6}, size: {}".format(t["name"], t["size"])
                extra_str += "\n  Total_size: {}".format(transfered_tensors["total_size"])
            elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
                evt_name = "Device->Host"
                extra_str = "\n  Tensors:"
                transfered_tensors = json.loads(evt.data_transfer.data_transfer.decode('utf-8'))
                for t in transfered_tensors["tensors"]:
                    extra_str += "\n    handle: {:>6}, size: {}".format(t["name"], t["size"])
                extra_str += "\n  Total_size: {}".format(transfered_tensors["total_size"])
            elif evt.type == IpuTraceEvent.LOAD_ENGINE:
                evt_name = "Load engine"
            elif evt.type == IpuTraceEvent.EXECUTE:
                evt_name = "Execute"
            else:
                evt_name = "Unknown event"
            evt_str += "{:<15s}: {:<8.3g} s   {}\n".format(evt_name, (evt.timestamp - start_time), extra_str)
            start_time = evt.timestamp

        print(evt_str)

    # Write Report to file
    if first:
        with open("graph.json", "w") as f:
            graph_report = utils.extract_compile_reports(raw_report)[0][1]
            f.write(graph_report)
        with open("execution.json", "w") as f:
            execution_report = utils.extract_execute_reports(raw_report)[0][1]
            f.write(execution_report)
        print("\nWritten to file: graph.json, execution.json")

    if opts.tile_activity_report:
        generate_tile_activity(raw_report)

    return {"graph": graph_report, "execution": execution_report}, start_time


def generate_tile_activity(report):
    activity = utils.extract_execution_state_timing_list_from_events(report)[0][1]

    if not activity:
        raise Exception("Could not parse report for activity")

    with open('activity.csv', 'w') as file:
        file.write(activity)
