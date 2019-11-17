# Copyright 2019 Graphcore Ltd.
import json
import os
import time
import numpy as np
from typing import Union, List, Dict, Optional, NamedTuple
from enum import Flag


class BaselineRegressionResult(NamedTuple):
    final_accuracy: float
    total_memory: float
    tile_memory: float
    avg_throughput: float
    epochs_to_full: int

    def to_dict(self):
        return {key: getattr(self, key) for key in self._fields}


class ResultStatus(Flag):
    SUCCESS = 0b00000000
    FAILED_OOM = 0b00000010
    FAILED_ACCURACY = 0b00000100
    FAILED_MEM_USAGE = 0b00001000
    FAILED_TILE_MEM = 0b00010000
    FAILED_THROUGHPUT = 0b00100000
    FAILED_NO_BASELINE = 0b10000000

    @staticmethod
    def get_active(flags):
        return [status.name for status in ResultStatus if flags & status]


class RegressionResult(object):
    def __init__(self,
                 input_filenames: List[str],
                 baseline: BaselineRegressionResult,
                 accuracies: Optional[np.ndarray] = None,
                 total_memory: Optional[np.ndarray] = None,
                 tile_memory: Optional[np.ndarray] = None,
                 throughput: Optional[np.ndarray] = None,
                 epochs_to_full: Optional[int] = None,
                 status: Optional[ResultStatus] = None):

        self.input_filenames = input_filenames

        self.accuracies = accuracies
        self.total_memory = total_memory
        self.tile_memory = tile_memory
        self.throughput = throughput
        self.epochs_to_full = epochs_to_full

        self.baseline = baseline

        self.status = status

        if status is None:
            stats = self._calculate_stats()
            self.final_accuracy, self.avg_throughput = stats

            self.status = self._status_compared_to_baseline()

    def _calculate_stats(self):
        accuracy = (None
                    if self.accuracies is None else self.accuracies[-1])

        throughput = (None
                      if self.throughput is None else np.mean(self.throughput))
        return accuracy, throughput

    def _status_compared_to_baseline(self):
        if self.baseline is None:
            return ResultStatus.FAILED_NO_BASELINE

        status = ResultStatus.SUCCESS

        # Accuracy should never fall at all
        if self.final_accuracy is not None and self.final_accuracy < self.baseline.final_accuracy:
            status |= ResultStatus.FAILED_ACCURACY

        # Allow 10% leeway on the memory usage
        if self.total_memory is not None and self.total_memory > 1.1 * self.baseline.total_memory:
            status |= ResultStatus.FAILED_MEM_USAGE

        # Allow 10% leeway on the memory usage
        if self.tile_memory is not None and self.tile_memory > 1.1 * self.baseline.tile_memory:
            status |= ResultStatus.FAILED_TILE_MEM

        # Also allow leeway on the throughput
        if self.avg_throughput is not None and self.avg_throughput < 0.9 * self.baseline.avg_throughput:
            status |= ResultStatus.FAILED_THROUGHPUT

        return status

    def to_dict(self):
        return {
            "status": ResultStatus.get_active(self.status),
            "final_accuracy": self.final_accuracy,
            "total_memory": self.total_memory,
            "tile_memory": self.tile_memory,
            "avg_throughput": self.avg_throughput,
            "epochs_to_full": self.epochs_to_full,
            "meta": {
                "test_baseline": self.baseline.to_dict() if self.baseline is not None else None,
                "input_filenames": self.input_filenames
            }
        }

    def write(self, output_directory, dbg_prefix: str = ""):
        filename = self._get_filename(dbg_prefix)
        full_path = os.path.join(output_directory, filename)
        print("Storing test output in file: " + full_path)
        with open(full_path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def _get_filename(self, dbg_prefix: str = ""):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        success_str = "pass" if self.status == ResultStatus.SUCCESS else "fail"
        return f"{dbg_prefix}{timestamp}-{success_str}.json"


class OutOfMemoryTestResult(object):
    def __init__(self, input_files):
        super().__init__(input_files, None, status=ResultStatus.FAILED_OOM)
