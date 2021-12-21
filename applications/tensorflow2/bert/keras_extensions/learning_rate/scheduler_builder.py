# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedQueue

from keras_extensions.learning_rate.decorators import FP32StepLearningRateSchedule
from keras_extensions.learning_rate.decorators import LearningRateEnqueuer
from keras_extensions.learning_rate.lr_schedules import LearningRateWarmupAndDecay

AVAILABLE_SCHEDULERS = {
        'up_down': LearningRateWarmupAndDecay
    }


def get_lr_scheduler(scheduler_name: str,
                     schedule_params: dict,
                     queue: IPUOutfeedQueue = None):

    if scheduler_name not in AVAILABLE_SCHEDULERS.keys():
        raise NameError(f'Schedule {scheduler_name} is not supported. Supported schedules: '
                        f'{list(AVAILABLE_SCHEDULERS.keys())}')

    if scheduler_name == 'up_down':
        # Process the schedule params for the up and down LR
        pass

    scheduler = AVAILABLE_SCHEDULERS[scheduler_name](**schedule_params)

    if queue is not None:
        scheduler = LearningRateEnqueuer(scheduler, queue)

    scheduler = FP32StepLearningRateSchedule(scheduler)

    return scheduler
