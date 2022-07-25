# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import math
from optimum.graphcore import IPUTrainer
import torch


class PerceiverTrainer(IPUTrainer):

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):

        print('Create scheduler')

        optimizer = self.optimizer if optimizer is None else optimizer
        if self.args.constant_cosine:
            print('Setting constant cosine lr schedule')

            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            total_decay_steps = num_training_steps - warmup_steps

            def lr_schedule_fn(step):
                if step <= warmup_steps:
                    return 1
                else:
                    step = step - warmup_steps
                    return 0.5 * (1 + math.cos((step * math.pi) / total_decay_steps))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
            print(self.lr_scheduler)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)
