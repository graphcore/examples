# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from contextlib import contextmanager
import random


@contextmanager
def does_not_raise():
    yield


test_log = """
2020-03-13 15:55:58 INFO Program Start
2020-03-13 15:55:58 INFO Building Model
2020-03-13 15:56:02.945631: I tensorflow/compiler/plugin/poplar/driver/poplar_platform.cc:50] Poplar version: 1.0.172 (e3f5f01337) Poplar package: 7e2488e8b4 Poplar Tensorflow version: v1.0.157_view_75eba25_up_439c0df6f0-3155-g5296066c3b)
2020-03-13 15:56:04 INFO Dataset length: 1
2020-03-13 15:56:04 INFO Need 2 IPUs. Requesting 2
2020-03-13 15:56:04 INFO Acquired device: Device Type:ipu Id:18 Version:1.0.38 NumIPUs:2 NumTilesPerIPU:1216 DeviceIds: {18}
2020-03-13 15:56:04 INFO Creating Session
2020-03-13 15:56:39 INFO Compiling Training Graph
2020-03-13 16:00:13 INFO Compiled. Duration 214.1097915172577 seconds
2020-03-13 16:00:13 INFO Training Started
2020-03-13 16:00:13 INFO Saving model to: /output/path/ckpts/20-03-13-15-55-58/model_0.onnx
2020-03-13 16:00:23 INFO Iteration:      0 Epoch:   0.00/200 Loss (MLM NSP): 9.750 0.729 Accuracy (MLM NSP): 0.042 0.344 Learning Rate: 0.00080 Duration: 0.1027 s Throughput:  623.1 samples/s
2020-03-13 16:00:23 INFO Iteration:      1 Epoch:   1.00/200 Loss (MLM NSP): 7.996 0.711 Accuracy (MLM NSP): 0.054 0.547 Learning Rate: 0.00080 Duration: 0.0959 s Throughput:  667.5 samples/s
2020-03-13 16:00:23 INFO Iteration:      2 Epoch:   2.00/200 Loss (MLM NSP): 7.336 0.673 Accuracy (MLM NSP): 0.055 0.562 Learning Rate: 0.00080 Duration: 0.0958 s Throughput:  668.2 samples/s
2020-03-13 16:00:24 INFO Iteration:      3 Epoch:   3.00/200 Loss (MLM NSP): 6.578 0.667 Accuracy (MLM NSP): 0.068 0.656 Learning Rate: 0.00080 Duration: 0.0958 s Throughput:  668.3 samples/s
2020-03-13 16:00:24 INFO Iteration:      4 Epoch:   4.00/200 Loss (MLM NSP): 6.312 0.645 Accuracy (MLM NSP): 0.086 0.594 Learning Rate: 0.00080 Duration: 0.0959 s Throughput:  667.5 samples/s
2020-03-13 16:00:13 INFO Saving model to: /output/path/ckpts/20-03-13-15-55-58/model_1.onnx
"""

logging_config_yaml = '''
step_num:
  regex: "Iteration:\\\\s+(\\\\d+)"
epoch_num:
  regex: "Epoch:\\\\s+([\\\\d\\\\.]+)"
loss:
  regex: "Loss\\\\s+\\\\(MLM\\\\s+NSP\\\\):\\\\s+([\\\\d\\\\.]+)\\\\s+([\\\\d\\\\.]+)"
  labels:
    - MLM
    - NSP
accuracy:
  regex: "Accuracy\\\\s+\\\\(MLM\\\\s+NSP\\\\):\\\\s+([\\\\d\\\\.]+)\\\\s+([\\\\d\\\\.]+)"
  labels:
    - MLM
    - NSP
model_save:
  regex: "Saving model to: ([^\\\\s+]+)"
'''

metric_config_yaml = '''
loss:
    margin: 0.05
    comparison: "<="
accuracy:
    margin: 0.05
    comparison: ">="
'''

save_model_log = "2020-03-13 16:00:34 INFO Saving model to: /output/path/ckpts/20-03-13-15-55-58/model_9:10.onnx"
iteration_log_dual_loss = "2020-03-13 16:00:34 INFO Iteration:      6 Epoch:   6.00/200 Loss (MLM NSP): 5.484 0.644 Accuracy (MLM NSP): 0.092 0.672 Learning Rate: 0.00080 Duration: 0.0957 s Throughput:  668.8 samples/s"
iteration_log_single_loss = "2020-03-13 16:00:34 INFO Iteration:      6 Epoch:   6.00/200 Loss: 0.644 Accuracy: 0.672 Learning Rate: 0.00080 Duration: 0.0957 s Throughput:  668.8 samples/s"


regexes = {
    "accuracy_dual": "Accuracy\\s+\\(MLM\\s+NSP\\):\\s+([\\d\\.]+)\\s+([\\d\\.]+)",
    "loss_dual": "Loss\\s+\\(MLM\\s+NSP\\):\\s+([\\d\\.]+)\\s+([\\d\\.]+)",
    "accuracy_single": "Accuracy:\\s+([\\d\\.]+)",
    "loss_single": "Loss:\\s+([\\d\\.]+)",
    "iteration": "Iteration:\\s+([0-9]+)",
    "epoch": "Epoch:\\s+([\\d\\.]+)",
}


def random_result(labels=("MLM", "NSP")):
    return {
        "losses": {l: random.random() for l in labels},
        "accuracies": {l: random.random() for l in labels}
    }


def build_sample_flag_data(checkpoint_path, is_gather, has_common, has_misc, missing_gather, missing_run, missing_mandatory):
    """ This function serves two purposes;
            (1) Generate the dictionary that would be read from YAML under various conditions (including erroroneous ones)
            (2) Generate the expected output under those conditions
    """
    # Dictionary sections, toggled using parameters
    misc = {
        "misc_flag_1": {
            "key": "--misc-flag",
            "value": "misc value"
        },
        "misc_flag_no_value": {
            "key": "--another-misc-flag"
        },
    }

    common = {
        "log_steps": {
            "key": "--steps-per-log",
            "value": 1
        },
        "config": {
            "key": "--config",
            "value": "configs/demo.json"
        }
    }

    gather_mandatory = {
        "checkpoint_output": {
            "set_in_script": True,
            "key": "--checkpoint-dir"
        }
    }

    run_mandatory = {
        "checkpoint_input": {
            "key": "--onnx-checkpoint",
            "set_in_script": True
        },
        "start_step": {
            "key": "--continue-training-from-step",
            "set_in_script": True
        },
        "run_for": {
            "key": "--epochs",
            "set_in_script": True
        }
    }

    def add_kv(expected, d):
        for elem in d.values():
            expected[elem["key"]] = elem["value"] if "value" in elem else None

    # Build the flags dictionary and populate the expected values
    test_flags = {}
    expected_kv = {}

    gather_all = {} if missing_mandatory else {**gather_mandatory}
    run_all = {} if missing_mandatory else {**run_mandatory}

    if is_gather:
        add_kv(expected_kv, gather_mandatory)
    else:
        add_kv(expected_kv, run_mandatory)

    if has_misc == "gather" or has_misc == "both":
        gather_all["misc"] = misc
        if is_gather:
            add_kv(expected_kv, misc)
    if has_misc == "run" or has_misc == "both":
        run_all["misc"] = misc
        if not is_gather:
            add_kv(expected_kv, misc)

    if has_common:
        test_flags["common"] = common
        add_kv(expected_kv, common)

    if not missing_gather:
        test_flags["gather"] = gather_all

    if not missing_run:
        test_flags["run"] = run_all

    return test_flags, expected_kv


class MockPipe():
    def __init__(self, logstr=""):
        self.logstr = logstr

    def readline(self):
        return self.logstr


class MockSubProcess(object):
    def __init__(self, output_stream):
        self._isopen = True
        self._iskilled = False

        self._stdout_line = iteration_log_single_loss if output_stream == "stdout" else ""
        self._stderr_line = iteration_log_single_loss if output_stream == "stderr" else ""

        self._stdout_called = False
        self._stderr_called = False

    @property
    def stderr(self):
        self._stderr_called = True
        m = MockPipe(self._stderr_line)
        self._isopen = False
        return m

    @property
    def stdout(self):
        self._stdout_called = True
        m = MockPipe(self._stdout_line)
        self._isopen = False
        return m

    def poll(self):
        return None if self._isopen else 0

    def kill(self):
        self._iskilled = True
