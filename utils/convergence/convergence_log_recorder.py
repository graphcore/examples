# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import logging
import json
import os
from collections import defaultdict

logger = logging.getLogger("convergence-harness")


class RecorderIsFullException(Exception):
    pass


class CheckpointRecording(object):
    def __init__(self, start_step, ckpt_path, max_len):
        self.start_step = start_step
        self.checkpoint_path = ckpt_path

        # Max-Len is used to check whether we've recorded sufficient steps (see `can_add_step` below)
        self.max_len = max_len

        self._results = defaultdict(dict)
        self._recorded_steps = set()

    def record_metric(self, step, metric, value):
        if not self.can_add_step(step):
            raise RecorderIsFullException("Cannot record into a full recorder")
        self._results[step][metric] = value
        self._recorded_steps.add(step)

    @property
    def current_len(self):
        return len(self._recorded_steps)

    def can_add_step(self, step):
        is_full = self.current_len >= self.max_len
        return (not is_full) or (is_full and step in self._recorded_steps)

    def to_json(self):
        return {
            "checkpoint": self.checkpoint_path,
            "start_step": self.start_step,
            "results": self._results
        }


class MockCheckpointRecording(object):
    def __init__(self, start_step):
        self.start_step = start_step

        self.last_step = None
        self.metrics = {}

        self._can_add = True

    def record_metric(self, step, metric, value):
        self.last_step = step
        self.metrics[metric] = value

    def can_add_step(self, step):
        return self._can_add


class RecorderInferface(object):
    def start_recording(self, ckpt_path):
        raise NotImplementedError

    def record_step(self, losses, accuracies):
        raise NotImplementedError

    def stop_recording(self):
        raise NotImplementedError

    def is_recording(self):
        raise NotImplementedError

    def update_step(self, step):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class LogRecorder(RecorderInferface):
    def __init__(self, recording_length, manifest_path, ckpt_root):
        self.current_step = 0
        self.checkpoint_logs = {}
        self.recording_length = recording_length
        self._current_recorder = None

        self._manifest_path = manifest_path
        self._ckpt_root = ckpt_root

    def start_recording(self, ckpt_path):
        if self._current_recorder is not None:
            logger.warning(
                "Already recording logs for the previous checkpoint. Stopping here and starting a fresh log.")
            self.stop_recording()

        # The log entry will contain the full abs-path to the ckpt, but we only want to
        # store it relative to the checkpoint-root, which uses the command-line arg
        # <args.storage-path>/ckpts
        ckpt_relpath = os.path.relpath(ckpt_path, self._ckpt_root)

        logger.debug(f"Starting recording of Recorder[{self.current_step}]")
        self._current_recorder = CheckpointRecording(
            self.current_step, ckpt_relpath, self.recording_length)

    def record_step_metric(self, metric, metric_value):
        if self._current_recorder is None:
            logger.warning(
                f"Trying to record step {self.current_step}, but recorder is None. Skipping entry.")
            return

        logger.debug(
            f"Recording metric {metric} for step {self.current_step} into Recorder[{self._current_recorder.start_step}]")
        logger.debug(f"Value: {metric}")

        if not self._current_recorder.can_add_step(self.current_step):
            raise RecorderIsFullException("Cannot record into a full recorder")

        self._current_recorder.record_metric(
            self.current_step, metric, metric_value)

    def stop_recording(self):
        logger.debug(
            f"Stopping recording of Recorder[{self._current_recorder.start_step}]")
        self.checkpoint_logs[self._current_recorder.start_step] = self._current_recorder
        self._current_recorder = None

    def update_step(self, step):
        self.current_step = step

        rec = self._current_recorder
        if rec is not None and not rec.can_add_step(self.current_step):
            logger.debug(
                f"Recorder[{self._current_recorder.start_step}] full.")
            self.stop_recording()
            self.save()

    def is_recording(self):
        return self._current_recorder is not None

    def to_json(self):
        json_recorders = {step: ckpt_log.to_json()
                          for step, ckpt_log in self.checkpoint_logs.items()}
        if (self.is_recording()):
            logger.warning(
                "Saving mid-recording, final log entry may be incomplete.")
            json_recorders[self._current_recorder.start_step] = self._current_recorder.to_json()

        return {
            "ckpt_logs": json_recorders
        }

    def save(self):
        logger.debug(f"Saving all recorders.")
        with open(self._manifest_path, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


class MockLogRecorder(RecorderInferface):
    def __init__(self):
        self.step_metrics = defaultdict(dict)
        self.step = -1

        self._started = False
        self._stopped = False
        self._saved = False

    def record_step_metric(self, metric, value):
        self.step_metrics[metric][self.step] = value

    def start_recording(self, ckpt_path):
        self._started = True

    def stop_recording(self):
        self._stopped = True

    def is_recording(self):
        return True

    def update_step(self, step):
        self.step = step

    def save(self):
        self._saved = True
