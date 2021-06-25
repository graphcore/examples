# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from torch.optim.lr_scheduler import _LRScheduler
import warnings


class WarmUpLRDecorator(_LRScheduler):
    def __init__(self, lr_scheduler, optimizer, warmup_epoch, last_epoch=-1):
        self.lr_scheduler = lr_scheduler
        self.warmup_epoch = warmup_epoch
        super(WarmUpLRDecorator, self).__init__(optimizer, last_epoch=last_epoch)


    def get_lr(self):
        lr = self.lr_scheduler._get_closed_form_lr()
        if self.last_epoch < self.warmup_epoch:
            return [e*(self.last_epoch)/float(self.warmup_epoch) for e in lr]
        else:
            return lr

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch - 1  # It is going to be increased by one in the parent's step function. Need to decrease by one to revert this effect.
        super(WarmUpLRDecorator, self).step()
        self.lr_scheduler.step(epoch)  # This has to be stepped in this order since the constructor calls step()


class PeriodicLRDecorator(_LRScheduler):
    def __init__(self, lr_scheduler, optimizer, period, last_epoch=-1):
        self.lr_scheduler = lr_scheduler
        self.period = period
        self.next_update = 0
        super(PeriodicLRDecorator, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return self.lr_scheduler._get_closed_form_lr()

    def step(self, epoch=None):
        epoch = epoch if epoch is not None else self.last_epoch + 1
        self.last_epoch = epoch
        if epoch >= self.next_update:
            self.next_update += self.period
            self.optimizer._step_count = 1  # Initialize step as Poptorch does not call optimizer.step() explicitly
            self.lr_scheduler.step()

    def _get_closed_form_lr(self):
        return self.lr_scheduler._get_closed_form_lr()
