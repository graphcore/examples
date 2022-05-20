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

import pytest
import re
import subprocess
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory


bert_root_dir = Path(__file__).parent.parent.resolve()


def run_bert_cmdline(cmdline_args):
    with TemporaryDirectory() as tempdir:
        cmdline_args["--wandb"] = "false"
        cmd = ["python3", "run_pretraining.py"]
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
        return out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def parse_result_for_loss_accuracy(out):
    """
    Look for "Loss: <val>" and "Acc/NSP: <val>" in the output
    """
    losses = []
    accs = []
    for line in out.split("\n"):
        match_loss = re.match(r".*(Loss: ([\d.]+)).*", line)
        match_acc = re.match(r".*(Acc\/MLM: ([\d.]+)).*", line)
        if match_loss:
            loss = match_loss.groups()[1]
            losses.append(float(loss))
        if match_acc:
            acc = match_acc.groups()[1]
            accs.append(float(acc))
    losses = np.array(losses)
    accs = np.array(accs)

    # Print output if it failed to run (no losses or accuracies found)
    if len(losses) == 0 or len(accs) == 0:
        raise ValueError("Run Failed. Output:\n" + out)
    return losses, accs


def loss_going_down(losses):
    # Test that loss at end is less than loss at start
    assert losses[-1] < losses[0]

    # Test that direction of loss is on average negative
    assert np.diff(losses).mean() < 0.0


def accuracy_going_up(accs):
    # Test that accuracy at end is greater than accuracy at start
    assert accs[-1] > accs[0]

    # Test that direction of accuracy is on average positive
    assert np.diff(accs).mean() > 0.0


def accuracy_reached_threshold(accs, threshold):
    assert accs[-1] > threshold


@pytest.mark.ipus(8)
@pytest.mark.parametrize("replication", [1, 2])
@pytest.mark.parametrize("embedding_serialization_factor", [3, 1])
def test_loss_down_accuracy_up(embedding_serialization_factor, replication):
    """
    Test that for a 3 layer toy model the loss is trending downwards and the
    accuracy is trending upwards.
    """
    _, stderr = run_bert_cmdline({
        "--config": "demo_tiny_128",
        "--training-steps": 350,
        "--replication-factor": replication,
        "--embedding-serialization-factor": embedding_serialization_factor,
        "--disable-progress-bar": True
    })
    losses, accs = parse_result_for_loss_accuracy(stderr)
    loss_going_down(losses)
    accuracy_going_up(accs)
    accuracy_reached_threshold(accs, 0.9)


@pytest.mark.ipus(8)
@pytest.mark.parametrize("embedding_serialization_factor", [3, 1])
def test_compile_and_train_with_als(embedding_serialization_factor):
    """
    For a 3 layer replicated toy model with and without serialised embeddings,
    ensure that the model compiles, loss trends downwards, and that the accuracy
    trends upwards when training with automatic loss scaling.
    """
    _, stderr = run_bert_cmdline({
        "--config": "demo_tiny_128",
        "--training-steps": 350,
        "--replication-factor": 2,
        "--embedding-serialization-factor": embedding_serialization_factor,
        "--loss-scaling": 1,
        "--auto-loss-scaling": True,
        "--disable-progress-bar": True
    })
    losses, accs = parse_result_for_loss_accuracy(stderr)
    loss_going_down(losses)
    accuracy_going_up(accs)
    accuracy_reached_threshold(accs, 0.9)


@pytest.mark.ipus(16)
@pytest.mark.skip_longtest_needs_dataset
def test_base_convergence():
    """
    Run a short 3 layer convergence run with wikitext dataset
    with replication.
    """
    _, stderr = run_bert_cmdline({
        "--config": "pretrain_base_3L_128_single",
        "--training-steps": 150,
        "--replication": 4,
        "--disable-progress-bar": True
    })
    losses, accs = parse_result_for_loss_accuracy(stderr)
    loss_going_down(losses)
    accuracy_going_up(accs)

    # After 150 steps loss should be ~3.4
    final_loss = losses[-1]
    assert 3.2 < final_loss < 3.5
