# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Optional
import logging
from dataclasses import dataclass, field
import warnings
from callbacks.callbacks_periodicity import calculate_log_period


@dataclass(frozen=True)
class BatchConfig:

    micro_batch_size: int = 1
    num_replicas: int = 1
    gradient_accumulation_count: Optional[int] = None
    global_batch_size: Optional[int] = None

    num_micro_batches_per_weight_update: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.micro_batch_size > 0
        assert self.num_replicas > 0
        if not (self.gradient_accumulation_count is None or self.global_batch_size is None):
            raise ValueError('Can not specify both gradient accumulation count and total batch size')
        elif self.gradient_accumulation_count is None and self.global_batch_size is None:
            raise ValueError('Either gradient accumulation count or total batch size must to be specified')

        if self.gradient_accumulation_count is not None:
            assert self.gradient_accumulation_count > 0
            object.__setattr__(self,
                               'global_batch_size',
                               (self.micro_batch_size *
                                self.num_replicas *
                                self.gradient_accumulation_count))
        elif self.global_batch_size is not None:
            assert self.global_batch_size > 0
            global_batch_size = self.global_batch_size  # for logging purposes
            gradient_accumulation_count = (self.global_batch_size /
                                           self.micro_batch_size /
                                           self.num_replicas)
            if self.global_batch_size % (self.micro_batch_size * self.num_replicas) == 0:
                object.__setattr__(self, 'gradient_accumulation_count', int(gradient_accumulation_count))
            else:
                object.__setattr__(self, 'gradient_accumulation_count', int(round(gradient_accumulation_count)))
                object.__setattr__(self,
                                   'global_batch_size',
                                   (self.micro_batch_size *
                                    self.num_replicas *
                                    self.gradient_accumulation_count))

                logging.warning(f'total batch size not divisible by micro batch size and number of replicas '
                                f'({global_batch_size}/{self.micro_batch_size}/{self.num_replicas} = {gradient_accumulation_count:.2f}). '
                                f'Gradient accumulation count rounded to {self.gradient_accumulation_count} and new '
                                f'global batch size is {self.global_batch_size}')

        object.__setattr__(self,
                           'num_micro_batches_per_weight_update',
                           self.gradient_accumulation_count * self.num_replicas)

        logging.info(f'micro batch size {self.micro_batch_size}')
        logging.info(f'global batch size {self.global_batch_size}')
        logging.info(f'gradient accumulation {self.gradient_accumulation_count}')
        logging.info(f'num replicas {self.num_replicas}')

    def get_num_micro_batches_per_epoch(self, dataset_size: int) -> int:
        return dataset_size // (self.micro_batch_size * self.num_micro_batches_per_weight_update) * (self.num_micro_batches_per_weight_update)

    def get_num_discarded_samples_per_instance(self, dataset_size: int, num_instances: int) -> int:
        dataset_size_per_instance = dataset_size // num_instances
        instance_batch_size = self.global_batch_size // num_instances  # batch size to feed all replicas in 1 instance

        return int(dataset_size_per_instance % instance_batch_size)

    def get_num_padding_samples_per_instance(self, num_instances: int, num_discarded_samples_per_instance: int) -> int:
        if num_discarded_samples_per_instance == 0:
            return 0

        instance_batch_size = self.global_batch_size // num_instances  # batch size to feed all replicas in 1 instance
        return instance_batch_size - num_discarded_samples_per_instance

    def get_padded_dataset_size(self, dataset_size: int, num_instances: int) -> int:
        num_discarded_samples = self.get_num_discarded_samples_per_instance(dataset_size, num_instances)
        if num_discarded_samples == 0:
            return dataset_size

        num_padding_samples = self.get_num_padding_samples_per_instance(num_instances, num_discarded_samples)

        return ((dataset_size // num_instances) + num_padding_samples) * num_instances


def calculate_micro_batch_periodicity(hparams, batch_config, dataset_size):
    if hparams.weight_updates_per_epoch == -1:
        hparams.weight_updates_per_epoch = dataset_size // batch_config.global_batch_size
    micro_batches_per_epoch = hparams.weight_updates_per_epoch * batch_config.num_micro_batches_per_weight_update

    micro_batches_per_log = calculate_log_period(
        hparams.weight_updates_per_epoch, hparams.num_epochs, hparams.logs_per_epoch, batch_config)
    logging.info(f'micro batches per log {micro_batches_per_log}')

    # steps_per_execution is the number of weight updates in term of micro batches before going back to the host
    if micro_batches_per_log != 0:
        micro_batches_per_execution = micro_batches_per_log
    else:
        # run training run in a single call
        logging.warn('The entire training run will be executed in a single call to the device.')
        micro_batches_per_execution = micro_batches_per_epoch * hparams.num_epochs

    # if we do more than one epoch per device call we need to adjust the number of epochs
    # and the number of micro batches processed in an epoch
    if micro_batches_per_epoch < micro_batches_per_execution:
        total_num_micro_batches = micro_batches_per_epoch * hparams.num_epochs
        hparams.num_epochs = int(total_num_micro_batches / micro_batches_per_execution)
        micro_batches_per_epoch = micro_batches_per_execution

    if (micro_batches_per_execution > micro_batches_per_epoch):
        warnings.warn(
            f'micro_batches_per_execution = {micro_batches_per_execution} > micro_batches_per_epoch = {micro_batches_per_epoch}')
        warnings.warn(
            f'This is not possible as micro_batches_per_epoch is a series of micro_batches_per_execution')
        warnings.warn(f'You might consider changing the number of micro_batches and / or weight_updates_per_execution')
        micro_batches_per_execution = micro_batches_per_epoch

    # micro_batches_per_epoch is the number of running micro batches per epoch which can be larger or smaller
    # than the actual number of steps per epoch ( = number of micro batches per epoch covering the whole dataset)
    if micro_batches_per_epoch % micro_batches_per_execution:
        raise ValueError(
            f'micro_batches_per_execution {micro_batches_per_execution} should divide micro_batches_per_epoch = {micro_batches_per_epoch}')

    return micro_batches_per_epoch, micro_batches_per_execution, micro_batches_per_log
