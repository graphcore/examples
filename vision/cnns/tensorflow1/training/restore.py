# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

import argparse
import json
import os
import importlib
import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restoring training run from a checkpoint')
    parser.add_argument(
        '--restore-path', help="Log folder to restore from", required=True)
    args = parser.parse_args()
    args = vars(args)
    with open(os.path.join(args["restore_path"], 'arguments.json'), 'r') as fp:
        opts = json.load(fp)

    print(opts)

    opts['restoring'] = True

    if opts['dataset'] == 'imagenet':
        if opts['image_size'] is None:
            opts['image_size'] = 224
        elif 'cifar' in opts['dataset']:
            opts['image_size'] = 32

    if opts['seed_specified']:
        train.set_seeds(int(opts['seed']))

    try:
        model = importlib.import_module("Models." + opts['model'])
    except ImportError:
        raise ValueError('Models/{}.py not found'.format(opts['model']))

    try:
        lr_schedule = importlib.import_module(
            "LR_Schedules." + opts['lr_schedule'])
    except ImportError:
        raise ValueError(
            "LR_Schedules/{}.py not found".format(opts['lr_schedule']))

    train.train_process(model, lr_schedule.LearningRate, opts)
