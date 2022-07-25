# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import deque
import torch
import horovod.torch as hvd
from .distributed import allreduce_values


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    accuracy = torch.mean(torch.eq(ind, labels).float()) * 100.0
    return accuracy


class Metrics:
    def __init__(self, running_mean_length=100, distributed=False):
        self.storage = {}
        self.running_mean_length = running_mean_length
        self.distributed = distributed

    def save_value(self, name, value):
        if name not in self.storage.keys():
            entry = {
                "sum": value,
                "count": 1,
                "running_mean_sum": value,
                "running_mean_items": deque([value], maxlen=self.running_mean_length),
            }
            self.storage[name] = entry
        else:
            entry = self.storage[name]
            entry["sum"] += value
            entry["count"] += 1
            entry["running_mean_sum"] += value
            if len(entry["running_mean_items"]) == self.running_mean_length:
                entry["running_mean_sum"] -= entry["running_mean_items"].popleft()
            entry["running_mean_items"].append(value)

    def reset_values(self):
        # Running mean values are 'reset' automatically based
        # on self.running_mean_length.
        for name in self.storage.keys():
            self.storage[name]["sum"] = 0.0
            self.storage[name]["count"] = 0

    def compute_mean_values(self, names, running_mean_names):
        values = []
        for name in names:
            values.append(self.storage[name]["sum"])
            values.append(self.storage[name]["count"])
        for name in running_mean_names:
            values.append(self.storage[name]["running_mean_sum"])
            values.append(len(self.storage[name]["running_mean_items"]))

        if self.distributed:
            values = [t.item() for t in allreduce_values(values, op=hvd.Sum)]

        mean_values = []
        for i in range(0, len(values), 2):
            mean_values.append(values[i] / values[i + 1])
        return mean_values[:len(names)], mean_values[len(names):]
