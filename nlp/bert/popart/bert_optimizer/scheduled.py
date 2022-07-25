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

import enum
import logging

from .optimizer import BaseOptimizerFactory

logger = logging.getLogger(__name__)


class ScheduleMode(enum.Enum):
    CONSTANT = 0
    STEP = 1
    EPOCH = 2


class Schedule:
    def __init__(self, mode, schedule, param, default_value):
        self.mode = mode
        self.schedule = schedule
        self.param = param

        self._initial_value = self.schedule[0] if 0 in self.schedule else default_value
        self.current_value = self.initial_value
        self.current_critereon = 0

    def should_update(self, iteration):
        """If using constant mode, we should never update the learning rate.
        If a shedule has been provided, check whether it's the right mode (i.e.
        due to a step or epoch change), and if so, whether it's the right time."""
        if self.mode == ScheduleMode.CONSTANT:
            return False

        # Check if the relevant critereon has changed (needed because we check epochs and steps)
        criterion = self._read_schedule_criterion(iteration)
        if criterion == self.current_critereon:
            return False

        self.current_critereon = criterion
        return criterion in self.schedule.keys()

    def update(self, iteration):
        criterion = self._read_schedule_criterion(iteration)

        # Sanity check that the learning rate is in the schedule, if not return the current LR
        if criterion is not None:
            self.current_value = self.schedule[criterion]
        return self.current_value

    @property
    def initial_value(self):
        return self._initial_value

    def _read_schedule_criterion(self, iteration):
        if self.mode == ScheduleMode.STEP:
            return iteration.count
        elif self.mode == ScheduleMode.EPOCH:
            return iteration.epoch
        return None

    def fast_forward(self, iteration):
        target_criterion = self._read_schedule_criterion(iteration)

        diffs = {(target_criterion - k): k for k in self.schedule.keys() if k <= target_criterion}
        if len(diffs) > 0:
            closest_key = diffs[min(diffs)]

            self.current_value = self.schedule[closest_key]
        return self.current_value

    @staticmethod
    def from_args(param, schedule_arg_epoch, schedule_arg_steps, default_value):
        # Epoch and step arguments are in a mutually exclusive group in argparse
        if schedule_arg_epoch is not None:
            mode = ScheduleMode.EPOCH
            schedule = Schedule.parse(param, schedule_arg_epoch)
        elif schedule_arg_steps is not None:
            mode = ScheduleMode.STEP
            schedule = Schedule.parse(param, schedule_arg_steps)
        else:
            # If no schedule is provided, set the learning rate mode to constant
            # and initialise it at the provided learning rate.
            mode = ScheduleMode.CONSTANT
            schedule = {0: default_value}
        return Schedule(mode, schedule, param, default_value)

    @staticmethod
    def parse(param, raw_schedule):
        try:
            return {int(k): float(raw_schedule[k]) for k in raw_schedule}
        except ValueError as ex:
            logger.warning(f"Invalid Schedule provided for parameter [{param}]. "
                           "It should be a set of int:float pairs.")
            raise ex


class ScheduledOptimizerFactory(BaseOptimizerFactory):
    def __init__(self, args, iteration, tensors=None):
        super().__init__(args, iteration, tensors)

        self._schedules = {}
        self.awaiting_update = []

        self.current_critereon = 0

        self._create_schedules(args)

        # Since the step count is set > 0 if we start from a given epoch,
        # this will catch either step or epoch start states
        self._fast_forward()

    def should_update(self, iteration):
        self.awaiting_update = [p for p, s in self._schedules.items() if s.should_update(iteration)]
        return len(self.awaiting_update) > 0

    def update(self, iteration):
        for param_name in self.awaiting_update:
            self.option_values[param_name] = self._schedules[param_name].update(iteration)

    def add_schedule(self, schedule):
        # This is required since if we specify any option as const, it cannot then change.
        if self._options_created:
            raise RuntimeError(
                "Cannot add new schedules once options have been created.")
        self._non_const_options.add(schedule.param)
        self._schedules[schedule.param] = schedule
        self.option_values[schedule.param] = schedule.initial_value

    def _create_schedules(self, args):
        if args.lr_schedule_by_epoch is not None or args.lr_schedule_by_step is not None:
            self.add_schedule(Schedule.from_args("defaultLearningRate",
                                                 args.lr_schedule_by_epoch,
                                                 args.lr_schedule_by_step,
                                                 args.learning_rate))
        if args.ls_schedule_by_epoch is not None or args.ls_schedule_by_step is not None:
            self.add_schedule(Schedule.from_args("lossScaling",
                                                 args.ls_schedule_by_epoch,
                                                 args.ls_schedule_by_step,
                                                 args.loss_scaling))

        logger.debug("Created schedules...")
        for schedule in self._schedules.values():
            logger.debug(f"Schedule[{schedule.param} | {str(schedule.mode)}]")
            for key, value in schedule.schedule.items():
                logger.debug(f"\t{key:>6}: {value}")

    def _fast_forward(self):
        for param_name in self._schedules.keys():
            self.option_values[param_name] = self._schedules[param_name].fast_forward(self.iteration)
