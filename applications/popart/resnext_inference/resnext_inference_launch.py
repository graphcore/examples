# Copyright 2019 Graphcore Ltd.
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
    args = FLAGS.flags_into_string().split('\n')
    print(f"\n\nRunning subprocess {i}: \t ")
    print(" ".join(["python3", "resnext101.py",
                    "--data_sub_dir", data_sub_dir] + args))
    return subprocess.Popen(["python3", "resnext101.py", "--data_sub_dir", data_sub_dir] + args, stdout=f, stderr=f)


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 6, "Batch size (per device)")
flags.DEFINE_integer(
    "num_ipus", 8, "Number of IPUs to be used. One IPU runs one compute process.")
flags.DEFINE_string("data_dir", "datasets/",
                    "Parent directory containing subdirectory dataset(s). Number of subdirs should equal num_ipus")
flags.DEFINE_integer("num_workers", 12, "Number of threads per dataloader")
flags.DEFINE_integer("batches_per_step", 1500,
                     "Number of batches to fetch on the host ready for streaming onto the device, reducing host IO")
flags.DEFINE_boolean(
    "profile", False, "Saves a GCProfile memory report. Use for debugging")
flags.DEFINE_string("model_name", "resnext101_32x4d",
                    "model name. Used to locate ONNX protobuf in models/")
flags.DEFINE_bool("synthetic", False, "Use synthetic data created on the IPU for inference")
flags.DEFINE_integer(
    "iterations", 1, "Number of iterations to run if using synthetic data. Each iteration uses one `batches_per_step` x `batch_size` x `H` x `W` x `C` sized input tensor.")


def main(argv):
    FLAGS = flags.FLAGS
    log_str = f"""
            Number of subprocesses created: {FLAGS.num_ipus}
            Per subprocess:
            \t Batch size: {FLAGS.batch_size}
            \t Number of batches prepared by the host at a time: {FLAGS.batches_per_step}
        """
    print(log_str)
    procs = []
    log_files = []
    timestamp = datetime.now().strftime("%H-%M-%S")
    if not os.path.exists("logs"):
        os.mkdir("logs")
    os.mkdir(f"logs/{timestamp}")

    for i in range(FLAGS.num_ipus):
        f = open(f"logs/{timestamp}/log_{i}", "w")
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
    throughputs = []
    latencies = []
    for i in range(FLAGS.num_ipus):
        sub_throughputs = []
        sub_latencies = []
        with open(f"logs/{timestamp}/log_{i}") as f:
            for line in f:
                match = regex_throughput.search(line)
                match_lat = regex_latency.search(line)
                if match:
                    res = match.group(1)
                    sub_throughputs.append(float(res))
                if match_lat:
                    res = match_lat.group(1)
                    sub_latencies.append(float(res))
        throughputs.append(sub_throughputs)
        latencies.append(sub_latencies)
    sums_throughputs = [sum(l) for l in zip(*throughputs)]
    mean_latencies = [statistics.mean(l) for l in zip(*latencies)]
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


if __name__ == '__main__':
    app.run(main)
