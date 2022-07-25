# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
from .scheduled import ScheduledOptimizerFactory, Schedule, ScheduleMode


def linear_schedule(start, end, interval, low, high):
    update_steps = np.arange(start, end, interval)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def linear_schedule_from_args(args, iteration):
    if args.lr_warmup_steps > 0:
        schedule = linear_schedule(0,
                                   args.lr_warmup_steps,
                                   args.lr_steps_per_warmup_update,
                                   args.lr_warmup_start,
                                   args.learning_rate)
    else:
        schedule = {}

    schedule.update(linear_schedule(args.lr_warmup_steps,
                                    iteration.total_steps,
                                    args.lr_steps_per_decay_update,
                                    args.learning_rate,
                                    1e-7))
    return Schedule(
        mode=ScheduleMode.STEP,
        schedule=schedule,
        param="Learning Rate",
        default_value=args.learning_rate)


class LinearOptimizerFactory(ScheduledOptimizerFactory):
    def __init__(self, args, iteration, tensors=None):
        super().__init__(args, iteration, tensors)

        lr_schedule = linear_schedule_from_args(args, iteration)

        self._non_const_options.add("defaultLearningRate")
        self._schedules["defaultLearningRate"] = lr_schedule
        self._fast_forward()
