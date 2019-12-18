# Copyright 2019 Graphcore Ltd.
import enum
import popart
import sys
import logging

logger = logging.getLogger(__name__)


class ScheduleMode(enum.Enum):
    CONSTANT = 0
    STEP = 1
    EPOCH = 2


class BaseOptimizerFactory():
    def __init__(self, args, iteration, tensors={}):

        self.option_values = {
            "defaultLearningRate": args.learning_rate,
            "defaultMomentum": args.momentum,
            "defaultDampening": args.dampening or args.momentum,
            "defaultVelocityScaling": args.velocity_scaling,
            "lossScaling": args.loss_scaling,
        }

        self._options_created = False
        self._non_const_options = set()

        self.iteration = iteration

        self.projection_lr_scaling = args.task == "PRETRAINING"
        self.projection_lr_scale = args.projection_lr_scale

        self.lr_scaling = args.pipeline_lr_scaling
        self.momentum_scaling = args.pipeline_momentum_scaling

        # If pipelining is enabled, we want to scale the parameters for different
        # pipeline stages. If not, don't perform scaling.
        # Note: This calculates the scale factors, not the absolute values
        if args.execution_mode == "PIPELINE" and (self.lr_scaling or self.momentum_scaling):
            self.pipeline_stage_tensors = tensors
            if self.lr_scaling:
                offset = args.pipeline_lr_scaling_offset
                self.pipeline_stage_lr_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
            if self.momentum_scaling:
                offset = args.pipeline_momentum_scaling_offset
                self.pipeline_stage_momentum_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
                if args.pipeline_dampening_scaling_offset is not None:
                    offset = args.pipeline_dampening_scaling_offset
                self.pipeline_stage_dampening_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
        else:
            self.pipeline_stage_tensors = {}
            self.pipeline_stage_lr_scaling = []

    @property
    def optimizer_options(self):
        self._options_created = True
        # By default, options are const. They only become variable when they're scheduled in some way, at
        # which point their key should be appended to _non_const_options
        return {k: (v, k not in self._non_const_options) for k, v in self.option_values.items()}

    @property
    def learning_rate(self):
        return self.option_values["defaultLearningRate"]

    def update_and_create(self, iteration):
        self.update(iteration)
        return self.create()

    def create(self):
        self.iteration.learning_rate = self.optimizer_options["defaultLearningRate"][0]

        optimizer = popart.SGD(self.optimizer_options)

        projection_scale_added = False

        for stage in self.pipeline_stage_tensors:
            specific_parameters = {}
            if self.lr_scaling:
                default_lr, lr_is_const = self.optimizer_options["defaultLearningRate"]
                specific_parameters["learningRate"] = (default_lr * self.pipeline_stage_lr_scaling[stage], lr_is_const)
            if self.momentum_scaling:
                # Momentum values are scaled inverse to the pipeline_stage
                momentum = 1 - ((1 - self.option_values["defaultMomentum"]) * self.pipeline_stage_momentum_scaling[stage])
                specific_parameters["momentum"] = (momentum, True)
                dampening = 1 - ((1 - self.option_values["defaultDampening"]) * self.pipeline_stage_dampening_scaling[stage])
                specific_parameters["dampening"] = (dampening, True)
            for tensor_id in self.pipeline_stage_tensors[stage]:
                # Special case for embedding/projection variable.
                if self.projection_lr_scaling and "Embedding_Dict" in tensor_id:
                    lr = specific_parameters.get("learningRate", self.optimizer_options["defaultLearningRate"])
                    params = specific_parameters.copy()
                    params["learningRate"] = (lr[0] * self.projection_lr_scale, lr[1])
                    optimizer.insertSpecific(tensor_id, params)
                    projection_scale_added = True
                else:
                    optimizer.insertSpecific(tensor_id, specific_parameters)

        if self.projection_lr_scaling and not projection_scale_added:
            lr = self.optimizer_options["defaultLearningRate"]
            optimizer.insertSpecific(
                "Embedding/Embedding_Dict",
                {"learningRate": (lr[0] * self.projection_lr_scale, lr[1])})

        return optimizer

    def should_update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")

    def update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")

    def _pipeline_stage_parameter_scaling(self, offset, tensors):
        if len(tensors) == 1:
            return {tensors.keys()[0]: 1.0}

        stages = tensors.keys()
        scale_factor = (1 - offset)/max(stages)
        return {stage: abs(scale_factor * stage + offset) for stage in stages}



class Schedule(object):
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
            logger.warn(f"Invalid Schedule provided for parameter [{param}]. "
                        "It should be a set of int:float pairs.")
            raise ex


class ScheduledOptimizerFactory(BaseOptimizerFactory):
    def __init__(self, args, iteration, tensors={}):
        super().__init__(args, iteration, tensors)
        self._schedules = {}
        self.awaiting_update = []

        self.current_critereon = 0

        self._create_schedules(args)

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


class LinearStepOptimizerFactory(BaseOptimizerFactory):
    def __init__(self, args, iteration, tensors={}):
        super().__init__(args, iteration, tensors)

        self.lr_option_name = "defaultLearningRate"

        self.target_lr = args.learning_rate

        self.enable_warmup = args.enable_warmup
        self.warmup_steps = args.warmup_steps
        self.warmup_init_lr = args.warmup_init_lr
        self.steps_per_warmup_update = args.steps_per_warmup_update

        self.enable_lr_decay = args.enable_lr_decay
        self.steps_per_decay_update = args.steps_per_decay_update

        self.option_values[self.lr_option_name] = self.warmup_init_lr if self.enable_warmup else self.target_lr
        self._non_const_options.add(self.lr_option_name)


    def should_update(self, iteration):
        warmup = self.in_warmup(iteration.count) and iteration.count % self.steps_per_warmup_update == 0
        decay = not self.in_warmup(iteration.count) and iteration.count % self.steps_per_decay_update == 0
        return warmup or decay

    def update(self, iteration):
        if self.in_warmup(iteration.count):
            self.option_values[self.lr_option_name] = self._warmup_learning_rate(iteration.count)
        elif self.enable_lr_decay:
            # We'll assume linear LR decay for now
            self.option_values[self.lr_option_name] = self._decayed_learning_rate(iteration.count)
        else:
            self.option_values[self.lr_option_name] = self.target_lr

    def in_warmup(self, step):
        return self.enable_warmup and step < self.warmup_steps

    def _warmup_learning_rate(self, step):
        lr_per_step = (self.target_lr - self.warmup_init_lr) / self.warmup_steps
        return lr_per_step * step + self.warmup_init_lr

    def _decayed_learning_rate(self, step, power=1.0):
        step, total_steps = self._adjust_steps_for_warmup(step)
        decay = (1 - (step / total_steps))
        if power != 1.0:
            decay **= power
        return self.target_lr * decay

    def _adjust_steps_for_warmup(self, step):
        # Adjust the step-count to consider warmup-steps if enabled
        if self.enable_warmup:
            step = step - self.warmup_steps
            total_steps = self.iteration.total_steps - self.warmup_steps
        else:
            total_steps = self.iteration.total_steps

        return step, total_steps
