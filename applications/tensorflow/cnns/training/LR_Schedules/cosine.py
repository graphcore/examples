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
    """A cosine learning rate schedule with optional warmup."""
    def __init__(self, opts, total_iterations):
        self.base_lr = 2 ** opts["base_learning_rate_exponent"]
        self.initial_lr = self.base_lr * opts["total_batch_size"]
        self.total_iterations = total_iterations

        self.warmup_iterations = 0
        if opts['warmup_epochs'] > 0:
            if opts['epochs']:
                self.warmup_iterations = total_iterations * opts["warmup_epochs"] // opts["epochs"]
            else:
                opts['warmup_epochs'] = 0
                print("--warmup-epochs needs --epochs not --iterations specified. Setting warmup-epochs to zero.")

    def feed_dict_lr(self, iteration):
        lr = self.initial_lr * 0.5 * (1 + np.cos((iteration * np.pi) / self.total_iterations))

        if iteration < self.warmup_iterations:
            return (iteration * lr) / self.warmup_iterations
        else:
            return lr


def add_arguments(parser):
    lr_group = parser.add_argument_group('Cosine Learning Rate')
    lr_group.add_argument('--warmup-epochs', type=int, default=5,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Cosine LR schedule\n"
    if opts["warmup_epochs"] > 0:
        opts['summary_str'] += " Warmup: {} epochs\n".format('{warmup_epochs}')
    else:
        opts['summary_str'] += " No warmup\n"
    return opts
