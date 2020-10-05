# Copyright (c) 2020 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
import tensorflow as tf


def run_train(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['python', 'din_train.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def run_validation(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['python', 'din_infer.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def get_log(out):
    log_dir = './din_log.txt'
    acc = 0.0
    auc = 0.0
    for line in out.split('\n'):
        if line.find('time over batch:') != -1:
            if float(line[43:49]) >= acc:
                acc = float(line[43:49])
                return acc
    else:
        auc = parse_log(log_dir)
        return auc


def parse_log(filepath):
    file = open(filepath)
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
            if line.find('INFO test_auc=') != -1:
                auc = float(line[34:40])
    file.close()
    return auc
