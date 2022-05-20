# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import subprocess
import sys


def run_train(instance, *args, poprun_prefix=[]):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    cmd = poprun_prefix + ['python3', 'train.py']
    cmd.extend(args)
    working_path = os.getcwd()
    try:
        return instance.run_command(
            cmd=cmd, working_path=working_path, expected_strings=[])
    except subprocess.CalledProcessError as e:
        print(f"command: {e.cmd}", file=sys.stderr)
        if e.stdout is not None:
            print(f"stdout:\n{e.stdout.decode()}\n", file=sys.stderr)
        if e.stderr is not None:
            print(f"stderr:\n{e.stderr.decode()}\n", file=sys.stderr)
        raise
