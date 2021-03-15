# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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


from pathlib import Path
import os
import subprocess
import pytest


@pytest.mark.category1
def run_train(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['python3', 'run_pretraining.py']
    args = [str(item) for sublist in kwargs.items() for item
            in sublist if item != '']
    cmd.extend(args)
    completed = subprocess.run(args=cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               )
    return completed


@pytest.mark.category1
def run_popdist_train(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    poprun_cmd = ['poprun', '-vv', '--mpi-global-args=--allow-run-as-root', '--num-instances', str(kwargs['--replicas']), '--ipus-per-replica', '1', '--num-replicas', str(kwargs['--replicas']), 'python3', 'run_pretraining.py']
    args = [str(item) for sublist in kwargs.items() for item
            in sublist if item != '']
    poprun_cmd.extend(args)
    completed = subprocess.run(args=poprun_cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               )
    return completed
