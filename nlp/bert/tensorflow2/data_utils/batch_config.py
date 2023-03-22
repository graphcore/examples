# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import math
from enum import Enum
from typing import Optional

import popdist.tensorflow


class Task(Enum):
    PRETRAIN_PHASE_ONE = 1
    PRETRAIN_PHASE_TWO = 2
    OTHER = 3


def get_pretraining_task(max_seq_length):
    if max_seq_length == 128:
        return Task.PRETRAIN_PHASE_ONE
    if max_seq_length == 384 or max_seq_length == 512:
        return Task.PRETRAIN_PHASE_TWO
    raise ValueError("Sequence length must be 128 or 384 or 512")


class BatchConfig:

    phase1_samples = 460849152
    phase2_samples = 34239360

    def __init__(
        self,
        micro_batch_size: int = 1,
        num_replicas: int = 1,
        gradient_accumulation_count: int = 1,
        num_pipeline_stages: int = 1,
        dataset_size: int = 1,
        global_batches_per_log: int = 1,
        total_num_train_samples: Optional[int] = None,
        task: Task = Task.PRETRAIN_PHASE_ONE,
    ):

        self.logger = logging.getLogger("batch_config")

        assert micro_batch_size > 0, "micro_batch_size must be greater than 0"
        self.micro_batch_size = micro_batch_size

        assert num_replicas > 0, "num_replicas must be greater than 0"
        self.num_replicas = num_replicas

        assert gradient_accumulation_count > 0, "gradient_accumulation_count must be greater than 0"
        self.gradient_accumulation_count = gradient_accumulation_count

        if self.gradient_accumulation_count % (num_pipeline_stages * 2) != 0 and num_pipeline_stages > 1:
            raise ValueError(
                "Gradient accumulation steps per replica"
                f" ({self.gradient_accumulation_count}) must be"
                " divisible by 2 times the number of pipeline"
                f" stages ({num_pipeline_stages}). Try adjusting"
                " the gradient accumulation steps per replica to"
                " achieve this."
            )

        assert dataset_size > 0, "dataset_size must be greater than 0"
        self.dataset_size = dataset_size

        assert dataset_size >= self.global_batch_size, "dataset_size must be greater than global batch size"

        self.task = task

        if total_num_train_samples is None:
            if self.task == Task.OTHER:
                raise ValueError(
                    "If not pretraining phase 1 or phase 2" " then the total_num_train_samples must" " be specified"
                )
            else:
                self.total_num_train_samples = self.get_num_training_samples_pretraining(self.task)
        else:
            self.total_num_train_samples = total_num_train_samples

        self.num_train_steps = self.total_num_train_samples // self.global_batch_size

        self.num_micro_batches_per_epoch = self.dataset_size // self.micro_batch_size

        self.steps_per_execution = self.gradient_accumulation_count * global_batches_per_log

        if self.steps_per_execution == 0:
            raise ValueError(
                "Steps per execution is zero, try increasing" " the total_num_train_samples or the dataset size"
            )

        all_replicas_steps_per_execution = self.steps_per_execution * self.num_replicas

        if self.num_micro_batches_per_epoch % all_replicas_steps_per_execution != 0:
            new_num_micro_batches_per_epoch = self.round_down_to_multiple(
                self.num_micro_batches_per_epoch, all_replicas_steps_per_execution
            )
            self.logger.warning(
                "Steps per execution across all replicas"
                ", steps_per_execution * num_replicas, "
                f" (set to {all_replicas_steps_per_execution})"
                " must be a factor of number of micro batches per epoch"
                f" {self.num_micro_batches_per_epoch}. Truncating number"
                f" of micro batches per epoch to {new_num_micro_batches_per_epoch}."
            )
            self.num_micro_batches_per_epoch = new_num_micro_batches_per_epoch

        if all_replicas_steps_per_execution > self.num_micro_batches_per_epoch:
            self.logger.warning(
                "Steps per execution across all replicas"
                ", steps_per_execution * num_replicas,"
                f" (set to {all_replicas_steps_per_execution})"
                f" is too large. Decreasing to number of micro batches"
                f" per epoch {self.num_micro_batches_per_epoch}."
            )
            self.steps_per_execution = self.num_micro_batches_per_epoch // self.num_replicas

        if self.num_micro_batches_per_epoch == 0:
            raise ValueError(
                "Number of micro batches per epoch is zero, try increasing"
                " the total_num_train_samples or the dataset size"
            )

        self.total_num_micro_batches = math.floor(self.num_micro_batches_per_epoch * self.epochs)
        self.total_num_micro_batches_per_instance = self.total_num_micro_batches // popdist.getNumInstances()
        self.total_num_micro_batches_per_instance = self.round_down_to_multiple(
            self.total_num_micro_batches_per_instance, all_replicas_steps_per_execution
        )
        self.total_num_micro_batches = self.total_num_micro_batches_per_instance * popdist.getNumInstances()

        if self.total_num_micro_batches == 0:
            raise ValueError(
                "Total number of micro batches is zero, try increasing"
                " the total_num_train_samples or the dataset size"
            )

        # Update the number of training steps to actual number of steps
        # given the updates above.
        self.num_train_steps = self.total_num_micro_batches * self.micro_batch_size // self.global_batch_size

    @staticmethod
    def round_down_to_multiple(in_val, round_to):
        return in_val // round_to * round_to

    @property
    def global_batch_size(self):
        return self.micro_batch_size * self.num_replicas * self.gradient_accumulation_count

    @property
    def num_micro_batches_per_weight_update(self):
        return self.gradient_accumulation_count * self.num_replicas

    @property
    def num_samples_processed_per_execution(self):
        return self.steps_per_execution * self.micro_batch_size * self.num_replicas

    @property
    def epochs(self):
        return self.num_train_steps / (self.dataset_size / self.global_batch_size)

    def __str__(self):
        return (
            f"\tMicro batch size: {self.micro_batch_size}\n\t"
            f"Number of replicas: {self.num_replicas}\n\t"
            f"Gradient accumulation count: {self.gradient_accumulation_count}\n\t"
            f"Global batch size: {self.global_batch_size}\n\t"
            f"Number of samples required to train: {self.total_num_train_samples}\n\t"
            f"Number of training steps: {self.num_train_steps}\n\t"
            f"Micro batches per epoch: {self.num_micro_batches_per_epoch}\n\t"
            f"Total number of epochs: {self.epochs:.2f}\n\t"
            f"Total number of micro batches: {self.total_num_micro_batches}\n\t"
            f"Steps per execution: {self.steps_per_execution}\n\t"
            f"Number of samples processed per execution: {self.num_samples_processed_per_execution}"
        )

    @classmethod
    def get_num_training_samples_pretraining(cls, task):
        if task == Task.PRETRAIN_PHASE_ONE:
            return cls.phase1_samples
        elif task == Task.PRETRAIN_PHASE_TWO:
            return cls.phase2_samples
        else:
            return None
