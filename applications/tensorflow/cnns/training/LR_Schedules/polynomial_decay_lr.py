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

import tensorflow as tf


class LearningRate:

    def __init__(self, opts, total_iterations):
        self.initial_lr = (2 ** opts["base_learning_rate_exponent"]) * opts["total_batch_size"]
        self.end_lr = opts["poly_lr_end_ratio"] * self.initial_lr
        self.power = opts["poly_lr_decay_power"]
        self.warmup_iterations = total_iterations * opts["warmup_epochs"] // opts["epochs"]
        self.decay_steps = total_iterations - self.warmup_iterations


    def feed_dict_lr(self, iteration):
        if iteration <= self.warmup_iterations and self.warmup_iterations > 0:
            return (iteration * self.initial_lr) / self.warmup_iterations
        cycle_step = iteration - self.warmup_iterations
        return (self.initial_lr - self.end_lr) * (1.0 - cycle_step / self.decay_steps) ** self.power + self.end_lr


def add_arguments(parser):
    lr_group = parser.add_argument_group('Polynomial Decay Learning Rate.')
    lr_group.add_argument('--warmup-epochs', type=float, default=5,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    lr_group.add_argument('--poly-lr-decay-power', type=float, default=2.0,
                          help="Exponent of polynomial describing the decay. Default 2.0.")
    lr_group.add_argument('--poly-lr-end-ratio', type=float, default=1e-5,
                          help="The end learning rate is the product of the initial learning rate and this ratio.")
    lr_group.add_argument('--abs-end-learning-rate', type=float,
                          help="Final learning rate is absolute terms.")
    return parser


def set_defaults(opts):
    if opts['abs_end_learning_rate'] is not None:
        opts['poly_lr_end_ratio'] = opts['abs_end_learning_rate'] / ((2 ** opts["base_learning_rate_exponent"]) * opts["total_batch_size"])

    opts['summary_str'] += "Polynomial decay applied to learning rate with exponent {poly_lr_decay_power}.\n"
    opts['summary_str'] += " Ratio of end and initial learning rates: {poly_lr_end_ratio}\n"
    opts['summary_str'] += " Warmup: {warmup_epochs} epochs\n"
    return opts
