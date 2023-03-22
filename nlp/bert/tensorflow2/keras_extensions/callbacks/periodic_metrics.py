# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List, Dict


class PeriodicMetrics:
    def __init__(self, metrics_keys: List[str]):
        self.accumulators = {metric: 0.0 for metric in metrics_keys}
        self.counters = {metric: 0 for metric in metrics_keys}

    def update(self, metrics: Dict[str, float]) -> None:
        for metric in metrics:
            self.accumulators[metric] += metrics[metric]
            self.counters[metric] += 1

    def reset(self) -> None:
        for metric in self.accumulators:
            self.accumulators[metric] = 0.0
            self.counters[metric] = 0

    def get_normalized(self) -> Dict[str, float]:
        normalized_metrics = {metric: self.accumulators[metric] / self.counters[metric] for metric in self.accumulators}
        return normalized_metrics
