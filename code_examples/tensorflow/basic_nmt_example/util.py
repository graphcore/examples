# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import re
import json

from tensorflow.python.ipu import utils


def get_config(report_n=1):
    """Builds ipu_options"""

    config = utils.create_ipu_config(profiling=False, use_poplar_text_report=False, report_every_nth_execution=report_n)
    config = utils.auto_select_ipus(config, [1])

    return config

start_time = 0


def extract_runtimes_from_report(report, display=True):
    """Returns timing information from IpuTraceEvent

    report -- Array of text encoded IpuTraceEvent

    """
    if len(report) is 0:
        return

    # Timings from tf xla event timestamps
    from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent

    # Retrieve IpuEvents, poplar report and cycles
    events = list(map(IpuTraceEvent.FromString, report))
    report = utils.extract_all_strings_from_event_trace(report)
    m = list(map(int, re.findall('Program cycles\s*:\s*([\d\.]+)', report)))

    global start_time
    first = start_time == 0
    if first:
        start_time = events[0].timestamp
        events = events[1:]
    evt_str = "\nIPU Timings\n"
    exec_num = 0

    for evt in events:
        extra_str = ""
        if evt.type == IpuTraceEvent.COMPILE_BEGIN:
            continue
        elif evt.type == IpuTraceEvent.COMPILE_END:
            evt_name = "Compile"
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

            if m and m[exec_num]:
                execution_time = float(m[exec_num]) / (1 * 1000 * 1000 * 1000)  # Implied 1GHz clock speed
                extra_str = "\n  Execution Time: {:.3g}s".format(execution_time)
                extra_str += "\n  Streaming Time: {:.3g}s".format((evt.timestamp - start_time) - execution_time)
                exec_num += 1
        else:
            evt_name = "Unknown event"
        evt_str += "{:<15s}: {:<8.3g} s   {}\n".format(evt_name, (evt.timestamp - start_time), extra_str)
        start_time = evt.timestamp

    # Print Cycle count from poplar report
    evt_str += "\nCycle counts on IPU\n"
    for execution_num, execution_cycles in enumerate(m):
        evt_str += "Execution {} cycles : {}\n".format(execution_num, execution_cycles)
    if display:
        print(evt_str)
    # Write Report to file
    if first:
        with open("report.txt", "w") as f:
            f.write(report)
        print("\nWritten to file: report.txt")
