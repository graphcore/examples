# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from torch.optim.lr_scheduler import _LRScheduler
import warnings


class WarmUpLRDecorator(_LRScheduler):
    def __init__(self, lr_scheduler, optimizer, warmup_epoch, last_epoch=-1):
        self.lr_scheduler = lr_scheduler
        self.warmup_epoch = warmup_epoch
        super(WarmUpLRDecorator, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        lr = self.lr_scheduler._get_closed_form_lr()
        if self.last_epoch < self.warmup_epoch:
            return [e*(self.last_epoch)/float(self.warmup_epoch) for e in lr]
        else:
            return lr

    def step(self, epoch=None):
        if epoch is None:
            super(WarmUpLRDecorator, self).step(epoch)
            self.lr_scheduler.step(epoch)  # this has to be stepped in this order since the constructor calls step()
        else:
            self.lr_scheduler.step(epoch)
            super(WarmUpLRDecorator, self).step(epoch)


class PeriodicLRDecorator(_LRScheduler):
    def __init__(self, lr_scheduler, optimizer, period, last_epoch=-1):
        self.lr_scheduler = lr_scheduler
        self.period = period
        self.next_update = 0
        super(PeriodicLRDecorator, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self.lr_scheduler.get_last_lr()

    def step(self, epoch=None):
        epoch = epoch if epoch is not None else self.last_epoch + 1
        if epoch >= self.next_update:
            self.next_update += self.period
            self.lr_scheduler.step(epoch)
        super(PeriodicLRDecorator, self).step(epoch)

    def _get_closed_form_lr(self):
        return self.lr_scheduler.get_last_lr()
