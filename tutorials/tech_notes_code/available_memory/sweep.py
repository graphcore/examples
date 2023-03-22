# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import subprocess
import argparse
import numpy as np


def run_demo(batch_size, available_memory_proportion):
    cmd = [
        "python",
        "pytorch_demo.py",
        "--available-memory-proportion",
        str(available_memory_proportion),
        "--batch-size",
        str(batch_size),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        out = out.decode("utf-8").splitlines()
        results = out[-1].split(",")
        throughput = float(results[-1].split(":")[-1])
        return False, throughput

    except Exception as err:
        if "popart::Session::prepareDevice: Poplar compilation" in err.stdout.decode("utf-8"):
            return True, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size-min", type=int, default=2, help="Batch size minimum (inclusive)")
    parser.add_argument(
        "--batch-size-max",
        type=int,
        default=26,
        help="Batch size maximum (non-inclusive)",
    )
    parser.add_argument("--batch-size-step", type=int, default=2, help="Batch size step size")
    parser.add_argument(
        "--available-memory-min",
        type=float,
        default=0.1,
        help="Available memory proportion minimum (inclusive)",
    )
    parser.add_argument(
        "--available-memory-max",
        type=float,
        default=1.0,
        help="Available memory proportion maximum (non-inclusive)",
    )
    parser.add_argument(
        "--available-memory-step",
        type=float,
        default=0.1,
        help="Available memory proportion step size",
    )
    opts = parser.parse_args()

    batch_sizes = range(opts.batch_size_min, opts.batch_size_max, opts.batch_size_step)
    amp_values = np.arange(opts.available_memory_min, opts.available_memory_max, opts.available_memory_step)

    ooms = []
    throughputs = []
    for bs in batch_sizes:
        for amp in amp_values:
            (is_oom, throughput) = run_demo(bs, amp)
            ooms.append(is_oom)
            throughputs.append(throughput)
            print(f"bs={bs},amp={amp},is_oom={is_oom},throughput={throughput}")

    ooms_array = np.array(ooms)
    throughputs_array = np.array(throughputs)

    sweep_config_str = f"bs-{opts.batch_size_min}-{opts.batch_size_step}-{opts.batch_size_max}"
    sweep_config_str += f"_amp-{opts.available_memory_min}-{opts.available_memory_step}-{opts.available_memory_max}"

    np.save(f"sweep_ooms_{sweep_config_str}.npy", ooms_array)
    np.save(f"sweep_throughputs_{sweep_config_str}.npy", throughputs_array)
