# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time
import logging
import wandb
import datetime

logger = logging.getLogger(__name__)


class TimeToTrain:

    def __init__(self):
        self._start_time = None
        self._total_time = None

    def start(self) -> None:
        self._start_time = time.time()

    def stop(self) -> None:
        stop_time = time.time()
        if self._start_time is None:
            raise RuntimeError('Timer was not initialized. Call TimeToTrain.start().')
        self._total_time = (stop_time - self._start_time) / 60.0

    def get_time_to_train(self) -> float:
        if self._total_time is None:
            raise RuntimeError('Timer was not stopped. Call TimeToTrain.stop().')
        return self._total_time


def log_time_to_train(time_to_train: TimeToTrain, log_to_wandb: bool) -> None:
    total_time_to_train = time_to_train.get_time_to_train()
    logger.info(f'time to train = {datetime.timedelta(minutes=total_time_to_train)}')
    if log_to_wandb:
        wandb.log({'time_to_train': total_time_to_train})
