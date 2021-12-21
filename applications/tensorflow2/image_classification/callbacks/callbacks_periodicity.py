# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
from batch_config import BatchConfig
from utilities import get_closest_divisor


def calculate_log_period(weight_updates_per_epoch: int, num_epochs: int, logs_per_epoch: float,
                         batch_config: BatchConfig) -> int:

    # handle disabling logs
    if logs_per_epoch == 0:
        return 0

    if logs_per_epoch < 1:
        # frequency in terms of the entire program
        weight_update_freq = weight_updates_per_epoch * num_epochs
        log_freq = int(logs_per_epoch * num_epochs)
    else:
        weight_update_freq = weight_updates_per_epoch
        log_freq = logs_per_epoch

    closest_log_freq = get_closest_divisor(weight_update_freq, log_freq)
    weight_updates_per_log = weight_update_freq / closest_log_freq
    if closest_log_freq != log_freq:
        if logs_per_epoch < 1:
            # convert closest freq back to logs per epoch
            closest_log_freq /= num_epochs
        logging.warn(f'The dataset size and batch configuration doesn\'t allow for {logs_per_epoch} logs per '
                     f'epoch. The closest possible frequency is {closest_log_freq} logs per epoch.')

    micro_batches_per_log = int(weight_updates_per_log * batch_config.num_micro_batches_per_weight_update)

    return micro_batches_per_log
