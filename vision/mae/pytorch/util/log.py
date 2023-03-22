# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys
import datetime
from logging import handlers
import time
import wandb


class WandbLog(object):
    def __init__(self, meters):
        self.meters = meters

    def log(self):
        log_dict = {}
        for m in self.meters:
            log_dict[m.name] = m.val
            log_dict[m.name + "_avg"] = m.avg
        wandb.log(log_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        info = "\t".join(entries)
        return info

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    # Predefined log level includes, from highest to lowest severity:
    # CRITICAL, ERROR, WARNING, INFO, DEBUG

    def __init__(self, level="INFO", when="D", backCount=3, fmt="[%(asctime)s] %(message)s"):
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.getLevelName(level))
        # fileHandler = logging.FileHandler('log.log', mode='w', encoding='UTF-8')
        # fileHandler.setLevel(logging.NOTSET)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format_str)
        self.logger.addHandler(sh)
        # self.logger.addHandler(fileHandler)


logger = Logger(level="INFO").logger
