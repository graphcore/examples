# Copyright 2020 Graphcore Ltd.
from torch.optim.lr_scheduler import MultiStepLR
import warnings


class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, lr, warmup_epoch=0, gamma=0.1, last_epoch=-1):
        self.initial_lr = lr
        self.warmup_epoch = warmup_epoch
        super(WarmupMultiStepLR, self).__init__(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch < self.warmup_epoch:
            new_lr = self.initial_lr*((self.last_epoch+1)/(float)(self.warmup_epoch))
            return [new_lr for group in self.optimizer.param_groups]
        else:
            return super(WarmupMultiStepLR, self).get_lr()
