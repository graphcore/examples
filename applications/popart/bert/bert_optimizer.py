# Copyright 2019 Graphcore Ltd.
import enum
import popart
import logging

logger = logging.getLogger(__name__)


class ScheduleMode(enum.Enum):
    CONSTANT = 0
    STEP = 1
    EPOCH = 2


class BaseOptimizerFactory():
    def __init__(self, args, iteration, tensors=None):

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

        self.squad_lr_scaling = args.task == "SQUAD"
        self.squad_lr_scale = args.squad_lr_scale

        self.lr_scaling = args.pipeline_lr_scaling
        self.weight_decay = args.weight_decay
        self.momentum_scaling = args.pipeline_momentum_scaling
        self.execution_mode = args.execution_mode
        self.tensors = tensors if tensors is not None else {}
        self.pipeline_stage_lr_scaling = []

        # If pipelining is enabled, we want to scale the parameters for different
        # pipeline stages. If not, don't perform scaling.
        # Note: This calculates the scale factors, not the absolute values
        if self.execution_mode == "PIPELINE" and (self.lr_scaling or self.momentum_scaling):
            if self.lr_scaling:
                offset = args.pipeline_lr_scaling_offset
                self.pipeline_stage_lr_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
            if self.momentum_scaling:
                offset = args.pipeline_momentum_scaling_offset
                self.pipeline_stage_momentum_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
                if args.pipeline_dampening_scaling_offset is not None:
                    offset = args.pipeline_dampening_scaling_offset
                self.pipeline_stage_dampening_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)

    @property
    def optimizer_options(self):
        self._options_created = True
        # By default, options are const. They only become variable when they're scheduled in some way, at
        # which point their key should be appended to _non_const_options
        return {k: (v, k not in self._non_const_options) for k, v in self.option_values.items()}

    @property
    def learning_rate(self):
        return self.option_values["defaultLearningRate"]

    def include_for_weight_decay(self, tensor_id):
        """ Do not include bias and norms for weight decay."""

        return self.weight_decay > 0 and not tensor_id.endswith(
            'B') and not tensor_id.endswith('Beta') and not tensor_id.endswith(
                'Gamma')

    def update_and_create(self, iteration):
        self.update(iteration)
        return self.create()

    def create(self):
        self.iteration.learning_rate = self.optimizer_options["defaultLearningRate"][0]

        optimizer = popart.SGD(self.optimizer_options)

        projection_scale_added = False
        weight_decay_tensor_list = []

        if self.execution_mode == "PIPELINE":
            for stage in self.tensors:

                specific_parameters = {}

                if self.lr_scaling:
                    default_lr, lr_is_const = self.optimizer_options["defaultLearningRate"]
                    specific_parameters["learningRate"] = (default_lr * self.pipeline_stage_lr_scaling[stage], lr_is_const)

                if self.momentum_scaling:
                    # Momentum values are scaled inverse to the pipeline_stage
                    if self.option_values["defaultMomentum"] != 0:
                        # This arithmetic will create FP rounding errors if momentum == 0.
                        momentum = 1 - ((1 - self.option_values["defaultMomentum"]) * self.pipeline_stage_momentum_scaling[stage])
                    else:
                        momentum = 0
                    specific_parameters["momentum"] = (momentum, True)

                    if self.option_values["defaultDampening"] != 0:
                        dampening = 1 - ((1 - self.option_values["defaultDampening"]) * self.pipeline_stage_dampening_scaling[stage])
                    else:
                        dampening = 0
                    specific_parameters["dampening"] = (dampening, True)

                for tensor_id in self.tensors[stage]:

                    if self.include_for_weight_decay(tensor_id):
                        specific_parameters["weightDecay"] = (self.weight_decay, True)
                        weight_decay_tensor_list.append(tensor_id)

                    # Special case for embedding/projection variable
                    if self.projection_lr_scaling and "Embedding_Dict" in tensor_id:
                        lr = specific_parameters.get("learningRate", self.optimizer_options["defaultLearningRate"])
                        params = specific_parameters.copy()
                        params["learningRate"] = (lr[0] * self.projection_lr_scale, lr[1])
                        optimizer.insertSpecific(tensor_id, params)
                        projection_scale_added = True
                    elif self.squad_lr_scaling and "Squad" in tensor_id:
                        logger.info(f"Setting SQuAD LR scaling for tensor [{tensor_id}]: {self.squad_lr_scale}")
                        lr = specific_parameters.get("learningRate", self.optimizer_options["defaultLearningRate"])
                        params = specific_parameters.copy()
                        params["learningRate"] = (lr[0] * self.squad_lr_scale, lr[1])
                        optimizer.insertSpecific(tensor_id, params)
                    else:
                        optimizer.insertSpecific(tensor_id, specific_parameters)

            if self.projection_lr_scaling and not projection_scale_added:
                lr = self.optimizer_options["defaultLearningRate"]
                optimizer.insertSpecific(
                    "Embedding/Embedding_Dict",
                    {"learningRate": (lr[0] * self.projection_lr_scale, lr[1])})

        else:
            for tensor_id in self.tensors[0]:
                if self.include_for_weight_decay(tensor_id):
                    specific_parameters = {"weightDecay": (self.weight_decay, True)}
                    weight_decay_tensor_list.append(tensor_id)
                    optimizer.insertSpecific(tensor_id, specific_parameters)

        if len(weight_decay_tensor_list) != 0:
            logger.info(f" Weight decay of {self.weight_decay} applied to: {weight_decay_tensor_list}")

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

    def fast_forward(self, iteration):
        target_criterion = self._read_schedule_criterion(iteration)

        diffs = {(target_criterion - k): k for k in self.schedule.keys() if k <= target_criterion}
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
        if iteration.count > 0:
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
