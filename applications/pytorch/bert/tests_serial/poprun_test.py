# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
import sys
from pathlib import Path
# Append bert directory
bert_root_path = str(Path(__file__).parent.parent)
sys.path.append(bert_root_path)
# Append tests directory
tests_root_path = str(Path(__file__).parent)
sys.path.append(tests_root_path)
import subprocess
import pytest
import numpy as np
import time
import transformers
from args import parse_bert_args
from tests.bert_tiny_test import parse_result_for_loss_accuracy, loss_going_down, accuracy_going_up
from ipu_options import get_options
from pretraining_data import get_dataloader, TFRecordPretrainingDataset


bert_root_dir = Path(__file__).parent.parent.resolve()


def run_poprun_cmdline(poprun_args, cmdline_args, script):
    cmdline_args["--wandb"] = "false"
    cmd = ["poprun"]
    cmd.extend([
        str(item) for sublist in poprun_args.items()
        for item in sublist
        if item != ""
    ])
    cmd.append("python3")
    cmd.append(script)
    cmd.extend([
        str(item) for sublist in cmdline_args.items()
        for item in sublist
        if item != ""
    ])
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=bert_root_dir)
    except subprocess.CalledProcessError as e:
            print(f"TEST FAILED")
            print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
            print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
            raise
    return out, out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def dataset():
    """
    Check if the data in two instances is different
    """
    args = "--config demo_tiny_128".split()
    config = transformers.BertConfig(**(vars(parse_bert_args(args))))
    opts = get_options(config)
    loader = TFRecordPretrainingDataset(config.input_files)
    loader = get_dataloader(config, opts)

    # Save part of the data as list
    loader_list = list(loader)[0][0][0].numpy()

    # MPI to broadcast data in root=1 to root=0
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    loader_list_copy = np.copy(loader_list)
    comm.Bcast(loader_list, root=1)

    # Assert if data broadcast to root=0 is different
    if comm.Get_rank() == 0 and not np.all(loader_list_copy == loader_list):
        print('Passed test: instances have different data')

    # Wait until both roots are finished
    time.sleep(2)


@pytest.mark.ipus(8)
def test_poprun_dataset():
    """
    Launch poprun as subprocess and assert output from poprun_utils.py
    """
    gc.collect()
    out, stdout, stderr = run_poprun_cmdline({
        "--mpi-global-args": "--allow-run-as-root",
        "--num-instances": 2,
        "--num-replicas": 2,
        "--ipus-per-replica": 4,
        "--numa-aware": "yes"},
        {},
        os.path.dirname(os.path.abspath(__file__)) + "/poprun_test.py"
    )
    assert 'Passed test: instances have different data' in stdout


@pytest.mark.ipus(4)
def test_poprun_loss_down_accuracy_up():
    """
    Test that for a 3 layer toy model the loss is trending downwards and the
    accuracy is trending upwards.
    """
    gc.collect()
    out, stdout, stderr = run_poprun_cmdline({
        "--mpi-global-args": "--allow-run-as-root",
        "--num-instances": 2,
        "--num-replicas": 2,
        "--ipus-per-replica": 2,
        "--numa-aware": "yes"
        },
        {
        "--config": "demo_tiny_128",
        "--training-steps": 25,
        "--ipus-per-replica": 2,
        "--layers-per-ipu": 0, "": 3,
        "--disable-progress-bar": "True"
        },
        "run_pretraining.py"
    )
    print(f"'\nOutput was:\n{stderr}")
    assert out.returncode == 0
    losses, accs = parse_result_for_loss_accuracy(stderr)
    loss_going_down(losses)
    accuracy_going_up(accs)


if __name__ == "__main__":
    dataset()
