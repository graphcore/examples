# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Optional
import logging
from dataclasses import dataclass, field


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

    def get_num_micro_batches_per_epoch(self, dataset_size: int) -> int:
        return dataset_size // (self.micro_batch_size * self.num_micro_batches_per_weight_update) * (self.num_micro_batches_per_weight_update)
