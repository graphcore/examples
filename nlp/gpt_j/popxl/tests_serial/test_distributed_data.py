# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import subprocess
import pytest
import gc

root_dir = Path(__file__).parent.parent.resolve()


def run_poprun_cmdline(poprun_args, cmdline_args, script):
    cmd = ["poprun"]
    cmd.extend([str(item) for sublist in poprun_args.items() for item in sublist if item != ""])
    cmd.append("python3")
    cmd.append(script)
    cmd.extend([str(item) for sublist in cmdline_args.items() for item in sublist if item != ""])
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root_dir)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out, out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def test_poprun_dataset():
    """
    Launch poprun as subprocess and assert output
    """
    gc.collect()
    out, stdout, stderr = run_poprun_cmdline(
        {
            "--num-instances": 2,
            "--num-replicas": 2,
        },
        {},
        os.path.join(root_dir, "tests_serial/distributed_sampler.py"),
    )
    assert "Passed test: instances have different data" in stdout, stderr
    assert "Passed test: each epoch samples dataset in different order" in stdout, stderr


def test_poprun_dataloader_checkpoints():
    """
    Launch poprun as subprocess and assert output
    """
    gc.collect()
    out, stdout, stderr = run_poprun_cmdline(
        {
            "--num-instances": 2,
            "--num-replicas": 2,
        },
        {},
        os.path.join(root_dir, "tests_serial/dataloader_checkpoints.py"),
    )
    assert "Passed test: distributed dataloader checkpoint" in stdout, stderr
