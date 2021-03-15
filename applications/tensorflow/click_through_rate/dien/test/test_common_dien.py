# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
    os.chdir('../')
    cmd = ['python', './dien_train.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def run_validation(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('../')
    cmd = ['python', './dien_infer.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def get_log(out):
    log_dir = './dien_log.txt'
    acc = 0.0
    auc = 0.0
    for line in out.split('\n'):
        if line.find('time over batch:') != -1 and line.find('accuracy: ') != -1:
            startIndex = line.find('accuracy: ')+len('accuracy: ')
            if float(line[startIndex:startIndex+6]) >= acc:
                acc = float(line[startIndex:startIndex+6])
                return acc
    auc = parse_log(log_dir)
    return auc


def parse_log(filepath):
    file = open(filepath)
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
            index = line.find('INFO test_auc=')
            if index != -1:
                startIndex = index + len('INFO test_auc=')
                auc = float(line[startIndex:startIndex+6])
    file.close()
    return auc
