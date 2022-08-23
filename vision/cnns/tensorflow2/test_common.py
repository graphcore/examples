# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import subprocess
import sys


def run(script_name, instance, args, poprun_prefix=[], env=None):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    cmd = poprun_prefix + ['python3', script_name]
    cmd.extend(args)
    working_path = os.getcwd()
    try:
        return instance.run_command(
            cmd=cmd, working_path=working_path, expected_strings=[], env=env)
    except subprocess.CalledProcessError as e:
        print(f"command: {e.cmd}", file=sys.stderr)
        if e.stdout is not None:
            print(f"stdout:\n{e.stdout.decode()}\n", file=sys.stderr)
        if e.stderr is not None:
            print(f"stderr:\n{e.stderr.decode()}\n", file=sys.stderr)
        raise


def run_train(instance, *args, poprun_prefix=[]):
    return run("train.py", instance, args, poprun_prefix)


def run_export(instance, *args, poprun_prefix=[]):
    sys.path.append(".")
    python_path = ":".join(sys.path)
    env = dict(os.environ)
    env['PYTHONPATH'] = python_path
    return run('scripts/export_for_serving.py', instance, args, poprun_prefix, env=env)


def run_serving(instance, *args, poprun_prefix=[]):
    sys.path.append(".")
    python_path = ":".join(sys.path)
    env = dict(os.environ)
    env['PYTHONPATH'] = python_path
    return run('send_request.py', instance, args, poprun_prefix, env=env)
