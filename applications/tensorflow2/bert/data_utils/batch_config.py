# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
from enum import Enum
from typing import Optional


class Task(Enum):
    PRETRAIN_PHASE_ONE = 1
    PRETRAIN_PHASE_TWO = 2
    OTHER = 3


class BatchConfig:

    phase1_samples = 460849152
    phase2_samples = 34239360

    def __init__(self,
                 micro_batch_size: int = 1,
                 num_replicas: int = 1,
                 gradient_accumulation_count: int = 1,
                 dataset_size: int = 1,
                 global_batches_per_log: int = 1,
                 total_num_train_samples: Optional[int] = None,
                 task: Task = Task.PRETRAIN_PHASE_ONE):

        assert micro_batch_size > 0, "micro_batch_size must be greater than 0"
        self.micro_batch_size = micro_batch_size

        assert num_replicas > 0, "num_replicas must be greater than 0"
        self.num_replicas = num_replicas

        assert gradient_accumulation_count > 0, "gradient_accumulation_count must be greater than 0"
        self.gradient_accumulation_count = gradient_accumulation_count

        assert dataset_size > 0, "dataset_size must be greater than 0"
        self.dataset_size = dataset_size

        assert dataset_size >= self.global_batch_size, "dataset_size must be greater than global batch size"

        self.task = task

        if total_num_train_samples is None:
            if self.task == Task.OTHER:
                raise ValueError("If not pretraining phase 1 or phase 2"
                                 "then the total_num_train_samples must"
                                 " be specified")
            else:
                self.total_num_train_samples = self.get_num_training_samples_pretraining(self.task)
        else:
            self.total_num_train_samples = total_num_train_samples

        self.num_train_steps = self.total_num_train_samples // self.global_batch_size

        self.num_micro_batches_per_epoch = self.dataset_size // self.micro_batch_size

        self.steps_per_execution = self.num_micro_batches_per_weight_update * global_batches_per_log

        if self.steps_per_execution == 0:
            raise ValueError("Steps per execution is zero, try increasing"
                             " the total_num_train_samples or the dataset size")

        if self.num_micro_batches_per_epoch % self.steps_per_execution != 0:
            new_num_micro_batches_per_epoch = (self.num_micro_batches_per_epoch // self.steps_per_execution) * self.steps_per_execution
            print(f"Steps per execution (set to {self.steps_per_execution})"
                  " must be a factor of number of micro batches per epoch"
                  f" {self.num_micro_batches_per_epoch}. Truncating number"
                  f" of micro batches per epoch to {new_num_micro_batches_per_epoch}.")
            self.num_micro_batches_per_epoch = new_num_micro_batches_per_epoch

        if self.steps_per_execution > self.num_micro_batches_per_epoch:
            print(f"Steps per execution (set to {self.steps_per_execution})"
                  f" is too large. Decreasing to number of micro batches"
                  f" per epoch {self.num_micro_batches_per_epoch}.")
            self.steps_per_execution = self.num_micro_batches_per_epoch

        if self.num_micro_batches_per_epoch == 0:
            raise ValueError("Number of micro batches per epoch is zero, try increasing"
                             " the total_num_train_samples or the dataset size")

        self.total_num_micro_batches = math.floor(self.num_micro_batches_per_epoch * self.epochs)
        self.total_num_micro_batches = self.total_num_micro_batches // self.steps_per_execution * self.steps_per_execution
        if self.total_num_micro_batches == 0:
            raise ValueError("Total number of micro batches is zero, try increasing"
                             " the total_num_train_samples or the dataset size")

        # Update the number of training steps to actual number of steps
        # given the updates above.
        self.num_train_steps = self.total_num_micro_batches * self.micro_batch_size // self.global_batch_size

    @property
    def global_batch_size(self):
        return self.micro_batch_size * self.num_replicas * self.gradient_accumulation_count

    @property
    def num_micro_batches_per_weight_update(self):
        return self.gradient_accumulation_count * self.num_replicas

    @property
    def epochs(self):
        return self.num_train_steps / (self.dataset_size / self.global_batch_size)

    def __str__(self):
        return (f"\tMicro batch size: {self.micro_batch_size}\n\t"
                f"Number of replicas: {self.num_replicas}\n\t"
                f"Gradient accumulation count: {self.gradient_accumulation_count}\n\t"
                f"Global batch size: {self.global_batch_size}\n\t"
                f"Number of samples required to train: {self.total_num_train_samples}\n\t"
                f"Number of training steps: {self.num_train_steps}\n\t"
                f"Micro batches per epoch: {self.num_micro_batches_per_epoch}\n\t"
                f"Total number of epochs: {self.epochs:.2f}\n\t"
                f"Total number of micro batches: {self.total_num_micro_batches}\n\t"
                f"Steps per execution: {self.steps_per_execution}")

    @classmethod
    def get_num_training_samples_pretraining(cls, task):
        if task == Task.PRETRAIN_PHASE_ONE:
            return cls.phase1_samples
        elif task == Task.PRETRAIN_PHASE_TWO:
            return cls.phase2_samples
        else:
            return None
