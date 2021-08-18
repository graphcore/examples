# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import time
from datetime import datetime
import os
from absl import app, flags
import subprocess
import re
import statistics

"""
This program launches subprocesses to handle data loading and resnext101 inference.

It can also be used to perform inference on other ONNX CNNs that take ImageNet sized input images.
To adapt, download a different ONNX model from the Python package `pretrainedmodels` via get_model.py,
or save your own model to models/<model_name>/<model_name>_<batch_size>.onnx

Then, run with the flag --model_name <model_name> --batch_size <batch_size>
"""


def launch_resnext_subprocess(i, f):
    # parse flags into list of strings to pass through to subprocesses
    # give the i_th process the i_th dataset
    data_sub_dir = FLAGS.data_dir + f"{i}"
    micro_batch_size = int(FLAGS.batch_size/FLAGS.num_ipus)
    micro_batch_size = str(micro_batch_size)
    args = FLAGS.flags_into_string().split('\n')
    command = [
        "python3",
        "resnext101.py",
        "--data_sub_dir",
        data_sub_dir,
        "--micro_batch_size",
        micro_batch_size
    ] + args
    print(f"\n\nRunning subprocess {i}: \t ")
    print(" ".join(command))
    kwargs = {"stdout": f, "stderr": f} if FLAGS.hide_output else {}
    return subprocess.Popen(
        command,
        universal_newlines=True,
        **kwargs
    )


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 48, "Overall size of batch (across all devices).")
flags.DEFINE_integer(
    "num_ipus", 8, "Number of IPUs to be used. One IPU runs one compute process and processes a fraction of the batch of samples.")
flags.DEFINE_string("data_dir", "datasets/",
                    "Parent directory containing subdirectory dataset(s). The number of sub directories should equal num_ipus")
flags.DEFINE_integer("num_workers", 4, "Number of threads per dataloader. There is one dataloader per IPU.")
flags.DEFINE_integer("batches_per_step", 1500,
                     "Number of batches to fetch on the host ready for streaming onto the device, reducing host IO")
flags.DEFINE_string("model_name", "resnext101_32x4d",
                    "model name. Used to locate ONNX protobuf in models/")
flags.DEFINE_bool("synthetic", False, "Use synthetic data created on the IPU.")
flags.DEFINE_integer(
    "iterations", 1, "Number of iterations to run if using synthetic data. Each iteration uses one `batches_per_step` x `batch_size` x `H` x `W` x `C` sized input tensor.")
flags.DEFINE_bool(
    "report_hw_cycle_count",
    False,
    "Report the number of cycles a 'run' takes."
)
flags.DEFINE_string(
    "model_path", None,
    (
        "If set, the model will be read from this"
        " specfic path, instead of models/"
    )
)
flags.DEFINE_string(
    "log_path", None,
    (
        "If set, the logs will be saved to this"
        " specfic path, instead of logs/"
    )
)
flags.DEFINE_bool(
    "hide_output", True,
    (
        "If set to true the subprocess that the model"
        " is run with will hide output."
    )
)


def main(argv):
    FLAGS = flags.FLAGS
    log_str = f"""
            Number of subprocesses created: {FLAGS.num_ipus}
            Per subprocess:
            \t Batch size: {FLAGS.batch_size}
            \t Number of batches prepared by the host at a time: {FLAGS.batches_per_step}
        """
    print(log_str)
    log_path = "logs" if not FLAGS.log_path else FLAGS.log_path
    procs = []
    log_files = []

    timestamp = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    os.mkdir(os.path.join(log_path, timestamp))

    for i in range(FLAGS.num_ipus):
        f = open(f"{log_path}/{timestamp}/log_{i}", "w")
        p = launch_resnext_subprocess(i, f)
        # sleep to prevent race conditions on acquiring IPUs
        time.sleep(1)
        # log
        log_files.append(f)
        procs.append(p)

    exit_codes = [p.wait() for p in procs]

    print(f"All processes finished with exit codes: {exit_codes}")
    for f in log_files:
        f.close()

    regex_throughput = re.compile("Compute .* sec .* (.*) images/sec.")
    regex_latency = re.compile("Total (.*).* sec.   Preprocessing")
    regex_cycle_counts = re.compile("Hardware cycle count per 'run': ([\d.]+)")
    throughputs = []
    latencies = []
    cycle_counts = []
    for i in range(FLAGS.num_ipus):
        sub_throughputs = []
        sub_latencies = []
        sub_cycle_counts = []
        with open(f"{log_path}/{timestamp}/log_{i}") as f:
            for line in f:
                match = regex_throughput.search(line)
                match_lat = regex_latency.search(line)
                match_cycles = regex_cycle_counts.search(line)
                if match:
                    res = match.group(1)
                    sub_throughputs.append(float(res))
                if match_lat:
                    res = match_lat.group(1)
                    sub_latencies.append(float(res))
                if match_cycles:
                    res = match_cycles.group(1)
                    sub_cycle_counts.append(float(res))
        throughputs.append(sub_throughputs)
        latencies.append(sub_latencies)
        cycle_counts.append(sub_cycle_counts)
    sums_throughputs = [sum(l) for l in zip(*throughputs)]
    mean_latencies = [statistics.mean(l) for l in zip(*latencies)]
    mean_cycle_counts = [statistics.mean(c) for c in zip(*cycle_counts)]
    stats = zip(mean_latencies, sums_throughputs)
    start = 2 if len(sums_throughputs) >= 4 else 0
    for (duration, through) in list(stats)[start:]:
        report_string = "Total {:<8.3} sec.".format(duration)
        report_string += "   Preprocessing {:<8.3} sec ({:4.3}%).".format(
            duration, 95.)  # just for the output
        report_string += "   Compute {:<8.3} sec ({:4.3}%).".format(
            duration, 95.)
        report_string += "   {:5f} images/sec.".format(int(through))
        print(report_string)
    if FLAGS.report_hw_cycle_count:
        print(
            "Hardware cycle count per 'run':",
            statistics.mean(mean_cycle_counts)
        )


if __name__ == '__main__':
    app.run(main)
