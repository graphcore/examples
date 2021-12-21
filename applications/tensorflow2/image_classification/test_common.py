# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import subprocess


def run_train(*args, poprun_prefix=[]):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    cmd = poprun_prefix + ['python3', 'train.py']
    cmd.extend(args)
    working_path = os.getcwd()
    try:
        completed = subprocess.run(args=cmd, cwd=working_path,
                                   shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=True)

    except subprocess.CalledProcessError as e:
        print(
            f"The following command failed: {cmd}\nWorking path: {working_path}\nOutput of failed command:\n{e.output}")
        raise

    return str(completed.stdout, 'utf-8')
