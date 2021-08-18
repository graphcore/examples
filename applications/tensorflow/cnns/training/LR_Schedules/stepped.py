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

import numpy as np


class LearningRate:
    """A stepped learning rate schedule with optional warmup"""
    def __init__(self, opts, total_iterations):
        self.learning_rate_decay = opts["learning_rate_decay"]
        self.lr_drops = [int(i * total_iterations) for i in opts['learning_rate_schedule']]
        self.lr = (2 ** opts["base_learning_rate_exponent"]) * opts["total_batch_size"]
        self.next_drop = self.lr_drops.pop(0)

        self.warmup_iterations = 0
        if opts['warmup_epochs'] > 0:
            if opts['epochs']:
                self.warmup_iterations = total_iterations * opts["warmup_epochs"] // opts["epochs"]
            else:
                opts['warmup_epochs'] = 0
                print("--warmup-epochs needs --epochs not --iterations specified. Setting warmup-epochs to zero.")

    def feed_dict_lr(self, iteration):
        if iteration > self.next_drop:
            self.lr *= self.learning_rate_decay
            if len(self.lr_drops) > 0:
                self.next_drop = self.lr_drops.pop(0)
            else:
                self.next_drop = np.inf

        if iteration < self.warmup_iterations:
            return (iteration * self.lr) / self.warmup_iterations
        else:
            return self.lr


def add_arguments(parser):
    lr_group = parser.add_argument_group('Stepped Learning Rate')
    lr_group.add_argument('--learning-rate-decay', type=float, default=0.1,
                          help="Learning rate decay factor (default 0.1)")
    lr_group.add_argument('--learning-rate-schedule', type=str,
                          help="Learning rate drop points (proportional). Comma Separated (eg. '0.5,0.75')")
    lr_group.add_argument('--warmup-epochs', type=int, default=5,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    return parser


def set_defaults(opts):
    if isinstance(opts['learning_rate_schedule'], str):
        opts['learning_rate_schedule'] = list(map(float, opts['learning_rate_schedule'].split(',')))

    opts['summary_str'] += "Stepped LR schedule\n"
    if opts["warmup_epochs"] > 0:
        opts['summary_str'] += " Warmup: {} epochs\n".format('{warmup_epochs}')
    else:
        opts['summary_str'] += " No warmup\n"

    opts['summary_str'] += (" Drops at {learning_rate_schedule}\n"
                            " Decay factor {learning_rate_decay}\n")
    return opts
