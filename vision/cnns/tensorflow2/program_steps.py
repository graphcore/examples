# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import batch_config
import logging
from utilities import get_closest_divisor
from typing import Tuple
import math


def calculate_program_steps(hparams: argparse.Namespace,
                            batch_config: batch_config.BatchConfig,
                            dataset_size: int) -> Tuple[int, int, int, int]:

    if hparams.weight_updates_per_epoch == -1:
        hparams.weight_updates_per_epoch = dataset_size // batch_config.global_batch_size
    micro_batches_per_epoch = hparams.weight_updates_per_epoch * batch_config.num_micro_batches_per_weight_update

    weight_updates_per_log = calculate_weight_updates_per_event(
        hparams.logs_per_epoch, hparams.weight_updates_per_epoch, hparams.num_epochs)
    micro_batches_per_log = weight_updates_per_log * batch_config.num_micro_batches_per_weight_update
    logging.info(f'micro batches per log {micro_batches_per_log}')

    weight_updates_per_ckpt = calculate_weight_updates_per_event(
        hparams.ckpts_per_epoch, hparams.weight_updates_per_epoch, int(hparams.num_epochs - hparams.first_ckpt_epoch))
    micro_batches_per_ckpt = weight_updates_per_ckpt * batch_config.num_micro_batches_per_weight_update
    logging.info(f'micro batches per checkpoint {micro_batches_per_ckpt}')

    # the common frequency that samples both the logging and checkpointing events is given by the
    # greatest common divisor between the two.
    weight_updates_per_execution = math.gcd(weight_updates_per_log, weight_updates_per_ckpt)
    if weight_updates_per_execution == 0:
        # run training run in a single call
        logging.warn('The entire training run will be executed in a single call to the device.')
        weight_updates_per_execution = hparams.weight_updates_per_epoch * hparams.num_epochs
    micro_batches_per_execution = weight_updates_per_execution * batch_config.num_micro_batches_per_weight_update

    # if we do more than one epoch per device call we need to adjust the number of epochs
    # and the number of micro batches processed in an epoch
    if micro_batches_per_execution > micro_batches_per_epoch:
        total_num_micro_batches = micro_batches_per_epoch * hparams.num_epochs
        hparams.num_epochs = int(total_num_micro_batches / micro_batches_per_execution)
        micro_batches_per_epoch = micro_batches_per_execution

    # micro_batches_per_epoch is the number of running micro batches per epoch which can be larger or smaller
    # than the actual number of steps per epoch ( = number of micro batches per epoch covering the whole dataset)
    if micro_batches_per_epoch % micro_batches_per_execution:
        raise ValueError(
            f'micro_batches_per_execution {micro_batches_per_execution} should divide micro_batches_per_epoch = {micro_batches_per_epoch}')

    return micro_batches_per_epoch, micro_batches_per_execution, micro_batches_per_log, micro_batches_per_ckpt


def calculate_weight_updates_per_event(events_per_epoch: float, weight_updates_per_epoch: int, num_epochs: int) -> int:

    # handle no events
    if events_per_epoch == 0:
        return 0

    if events_per_epoch < 1:
        # frequency in terms of the entire program
        weight_update_freq = weight_updates_per_epoch * num_epochs
        event_freq = int(events_per_epoch * num_epochs)
    else:
        weight_update_freq = weight_updates_per_epoch
        event_freq = events_per_epoch

    closest_event_freq = get_closest_divisor(weight_update_freq, event_freq)
    weight_updates_per_event = weight_update_freq // closest_event_freq
    if closest_event_freq != event_freq:
        if events_per_epoch < 1:
            # convert closest freq back to events per epoch
            closest_event_freq /= num_epochs
        logging.warn(f'The dataset size and batch configuration doesn\'t allow for {events_per_epoch} events per '
                     f'epoch. The closest possible frequency is {closest_event_freq} events per epoch.')

    return weight_updates_per_event
