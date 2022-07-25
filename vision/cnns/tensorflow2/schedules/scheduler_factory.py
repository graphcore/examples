# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedQueue
import logging

from schedules.lr_schedules import CosineLRSchedule, ConstLRSchedule
from schedules.decorators import ShiftWarmup, FadingMaskWarmup, FP32StepLearningRateSchedule, StairCase

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utilities import verify_params_present


logger = logging.getLogger('scheduler_builder')

AVAILABLE_SCHEDULERS = {
    'const': ConstLRSchedule,
    'cosine': CosineLRSchedule,
    'stepped': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    'polynomial': tf.keras.optimizers.schedules.PolynomialDecay
}


def get_lr_scheduler(scheduler_name: str,
                     schedule_params: dict,
                     global_batch_size: int,
                     weight_updates_per_epoch: int,
                     warmup_params: dict = None,
                     staircase: bool = False,
                     queue: IPUOutfeedQueue = None,
                     factor: float = 1.):

    if scheduler_name not in AVAILABLE_SCHEDULERS.keys():
        raise NameError(f'Schedule {scheduler_name} is not supported. Supported schedules: '
                        f'{list(AVAILABLE_SCHEDULERS.keys())}')

    schedule_params = schedule_params.copy()

    if scheduler_name == 'cosine':
        expected_params = ['epochs_to_total_decay', 'initial_learning_rate']
        verify_params_present(list(schedule_params.keys()), expected_params, scheduler_name, '--lr-schedule-params')

        # convert epochs to weight updates
        schedule_params['weight_updates_to_total_decay'] = int(
            schedule_params['epochs_to_total_decay'] * weight_updates_per_epoch)
        schedule_params.pop('epochs_to_total_decay')
        # scale initial learning rate
        schedule_params['initial_learning_rate'] *= global_batch_size * factor
        logger.info(f'initial learning rate = {schedule_params["initial_learning_rate"]}')

    elif scheduler_name == 'stepped':
        expected_params = ['boundaries', 'values']
        verify_params_present(list(schedule_params.keys()), expected_params, scheduler_name, '--lr-schedule-params')
        if not len(schedule_params['boundaries']) == len(schedule_params['values']) - 1:
            raise ValueError(
                'When using --lr-schedule=\'stepped\', number of elements in \'boundaries\' has to be one less than \'values\'.')
        # convert epochs to weight updates
        schedule_params['boundaries'] = [int(epoch * weight_updates_per_epoch)
                                         for epoch in schedule_params['boundaries']]
        schedule_params['values'] = [value * factor for value in schedule_params['values']]

    elif scheduler_name == 'const':
        schedule_params['initial_learning_rate'] *= global_batch_size * factor

    elif scheduler_name == 'polynomial':
        expected_params = ['initial_learning_rate', 'epochs_to_total_decay', 'end_learning_rate_ratio', 'power']
        verify_params_present(list(schedule_params.keys()), expected_params, scheduler_name, '--lr-schedule-params')
        schedule_params['initial_learning_rate'] *= float(global_batch_size * factor)
        schedule_params['decay_steps'] = float(schedule_params['epochs_to_total_decay'] * weight_updates_per_epoch)
        schedule_params.pop('epochs_to_total_decay')
        schedule_params['end_learning_rate'] = (float(schedule_params['initial_learning_rate']) *
                                                float(schedule_params['end_learning_rate_ratio']))
        schedule_params.pop('end_learning_rate_ratio')
        schedule_params['power'] = float(schedule_params['power'])

    scheduler = AVAILABLE_SCHEDULERS[scheduler_name](**schedule_params)

    if warmup_params:
        warmup_params = warmup_params.copy()
        expected_params = ['warmup_mode', 'warmup_epochs']
        verify_params_present(list(warmup_params.keys()), expected_params, 'warmup', '--lr-warmup-params')

        # convert epochs to weight updates
        warmup_params['warmup_weight_updates'] = int(warmup_params['warmup_epochs'] * weight_updates_per_epoch)
        warmup_params.pop('warmup_epochs')

        # validate warmup mode
        VALID_WARMUP_MODES = {'shift': ShiftWarmup, 'mask': FadingMaskWarmup}
        if warmup_params["warmup_mode"] not in VALID_WARMUP_MODES.keys():
            raise NameError(f'Warmup decorator of name {warmup_params["warmup_mode"]} is not valid. '
                            f'Available warmup modes include: {list(VALID_WARMUP_MODES.keys())}')

        scheduler = VALID_WARMUP_MODES[warmup_params['warmup_mode']](
            scheduler, warmup_params['warmup_weight_updates'])

    if staircase:
        logger.info(f'staircase is enabled with weight_updates_per_epoch = {weight_updates_per_epoch}')
        scheduler = StairCase(scheduler, weight_updates_per_epoch)

    scheduler = FP32StepLearningRateSchedule(scheduler)

    return scheduler
