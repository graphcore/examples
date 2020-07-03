# Copyright 2020 Graphcore Ltd.

import subprocess
import pytest
import yaml
import tempfile
import random
import logging
import itertools
import numpy as np
import tempfile
import os
from collections import Counter

import convergence_harness
from convergence_harness import (
    gather_log_handler,
    run_log_handler,
    parse_log_regex,
    flags_from_config,
    MissingConfigurationOptionException,
    MissingMandatoryFlagException,
    compare_log_result,
    create_run_process,
    do_convergence_test
)
from convergence_log_recorder import (
    RecorderIsFullException,
    CheckpointRecording,
    MockCheckpointRecording,
    LogRecorder,
    MockLogRecorder
)
from test_utils import (
    test_log,
    logging_config_yaml,
    metric_config_yaml,
    does_not_raise,
    save_model_log,
    iteration_log_dual_loss,
    iteration_log_single_loss,
    regexes,
    random_result,
    build_sample_flag_data,
    MockSubProcess
)


@pytest.mark.category1
@pytest.mark.parametrize(
    "item, log_entry, expected",
    [
        ({"regex": regexes["accuracy_dual"]},
         iteration_log_dual_loss, {0: '0.092', 1: '0.672'}),
        ({"regex": regexes["accuracy_dual"], "labels": ["MLM", "NSP"]},
         iteration_log_dual_loss, {"MLM": '0.092', "NSP": '0.672'}),
        ({"regex": regexes["accuracy_dual"], "labels": ["MLM"]},
         iteration_log_dual_loss, {0: '0.092', 1: '0.672'}),
        ({"regex": regexes["accuracy_single"], "labels": ["MLM"]},
         iteration_log_single_loss, {"MLM": '0.672'}),
        ({"regex": regexes["accuracy_single"], "labels": ["Loss"]},
         iteration_log_single_loss, {"Loss": '0.672'}),
        ({"regex": regexes["accuracy_single"], "labels": ["MLM", "NSP"]},
         iteration_log_single_loss, {0: '0.672'}),
        ({"regex": regexes["loss_dual"]},
         iteration_log_dual_loss, {0: '5.484', 1: '0.644'}),
        ({"regex": regexes["loss_single"]},
         iteration_log_single_loss, {0: '0.644'}),
        ({"regex": regexes["iteration"]}, iteration_log_dual_loss, {0: '6'}),
        ({"regex": regexes["epoch"]}, iteration_log_dual_loss, {0: '6.00'})
    ])
def test_regex_parser(item, log_entry, expected):
    output = parse_log_regex(log_entry, item)
    assert output is not None
    assert output == expected


@pytest.mark.category1
@pytest.mark.parametrize(
    "log_lines, e_loss, e_accuracy",
    [
        ([iteration_log_dual_loss], {6: {"MLM": '5.484', "NSP": '0.644'}}, {
         6: {"MLM": '0.092', "NSP": '0.672'}}),
        (test_log.splitlines(), {
            0: {"MLM": "9.750", "NSP": "0.729"},
            1: {"MLM": "7.996", "NSP": "0.711"},
            2: {"MLM": "7.336", "NSP": "0.673"},
            3: {"MLM": "6.578", "NSP": "0.667"},
            4: {"MLM": "6.312", "NSP": "0.645"}
        }, {
            0: {"MLM": "0.042", "NSP": "0.344"},
            1: {"MLM": "0.054", "NSP": "0.547"},
            2: {"MLM": "0.055", "NSP": "0.562"},
            3: {"MLM": "0.068", "NSP": "0.656"},
            4: {"MLM": "0.086", "NSP": "0.594"}
        })
    ])
def test_gather_log_handler(log_lines, e_loss, e_accuracy):
    logging_config = yaml.load(logging_config_yaml, Loader=yaml.FullLoader)
    metric_config = yaml.load(metric_config_yaml, Loader=yaml.FullLoader)

    # Setup a dummy recorder to store results in
    mock_recorder = MockLogRecorder()
    for line in log_lines:
        gather_log_handler(line, logging_config, metric_config, mock_recorder)

    assert mock_recorder.step_metrics['loss'] == e_loss
    assert mock_recorder.step_metrics['accuracy'] == e_accuracy


@pytest.mark.category1
@pytest.mark.parametrize(
    "log_steps, manifest_steps, metric_names, parser_names, result_names, expected_warning, e_comparison_count",
    [
        # Everything as it should be - no warning
        (
            range(0, 10, 2),
            list(range(0, 10, 2)),
            ("loss", "accuracy"),
            ("step_num", "loss", "accuracy"),
            ("loss", "accuracy"),
            None,
            10,

        ),
        # "accuracy" definition is missing from the metric config,  no warning, but
        # doesn't process accuracy (check with spy)
        (
            range(0, 10, 2),
            list(range(0, 10, 2)),
            ("loss",),
            ("step_num", "loss", "accuracy"),
            ("loss", "accuracy"),
            None,
            5
        ),
        # accuracy missing from parser list, warn 'Cannot parse requested...'
        (
            range(0, 10, 2),
            list(range(0, 10, 2)),
            ("loss", "accuracy"),
            ("step_num", "loss"),
            ("loss", "accuracy"),
            "Cannot parse requested metric: accuracy - parser not supplied in the config. Skipping.",
            5
        ),
        # accuracy missing from result list, warn 'Cannot compare requested...'
        (
            range(0, 10, 2),
            list(range(0, 10, 2)),
            ("loss", "accuracy"),
            ("step_num", "loss", "accuracy"),
            ("loss",),
            "Cannot compare requested metric: accuracy - result not present in manifest. Skipping.",
            5
        ),
        # step missing from result, no warning, but not processed
        (
            range(0, 10, 2),
            list(range(0, 10, 4)),
            ("loss", "accuracy"),
            ("step_num", "loss", "accuracy"),
            ("loss", "accuracy"),
            None,
            6
        ),
        # Step returns None: drop out early
        (
            [None],
            list(range(0, 10, 4)),
            ("loss", "accuracy"),
            ("step_num", "loss", "accuracy"),
            ("loss", "accuracy"),
            None,
            0
        ),
    ])
def test_run_log_handler(monkeypatch, caplog, log_steps, manifest_steps, metric_names, parser_names, result_names, expected_warning, e_comparison_count):
    """
    Test the behaviour of the run log handler. This will check that all metrics are
    provided (and if not output a warning), all log parsers are provided (also outputting
    a warning), and ensure that the loop is skipped if either the step is None, or not
    provided in the results.
    """
    comparison_count = 0

    def generate_log_configs():
        return {key: key for key in parser_names}

    def generate_metric_configs():
        return {key: {"margin": random.random(), "comparison": ""} for key in metric_names}

    def generate_manifest_results():
        return {str(step): {key: {} for key in result_names} for step in manifest_steps}

    def mock_comparison_fn(*args, **kwargs):
        nonlocal comparison_count
        comparison_count += 1

    def mock_parser(line, lc):
        # If logging_config is the step parser, return the next step from `log_steps`
        # if it's the metric parser, return a random number to fill `metric_value`
        if lc == "step_num":
            # The actual regex parser always returns list of results
            if line == "None":
                return None
            return [line]

        return {
            "MLM": random.random(),
            "NSP": random.random()
        }

    monkeypatch.setattr(convergence_harness,
                        "compare_log_result", mock_comparison_fn)
    monkeypatch.setattr(convergence_harness,
                        "parse_log_regex", mock_parser)

    logging_config = generate_log_configs()
    metric_config = generate_metric_configs()
    manifest_results = generate_manifest_results()

    for log_step in log_steps:
        # Rather than actually parsing anything, we'll pass the logging step as a
        # string directly and handle accordingly in the mock parser
        with caplog.at_level(logging.INFO, logger='convergence-harness'):
            run_log_handler(str(log_step), logging_config,
                            metric_config, manifest_results)
            if expected_warning is not None:
                assert caplog.records[0].message == expected_warning
    assert comparison_count == e_comparison_count


@pytest.mark.category1
def test_checkpoint_recorder_can_add_step():
    """Ensure that the checkpoint recording stores data and correctly observes the max log length."""
    max_len = 10
    current_step = 0
    cr = CheckpointRecording(10, f"mock/ckpt_10/path", max_len)
    assert cr.can_add_step(current_step) is True
    cr._recorded_steps.add(current_step)
    assert cr.can_add_step(current_step) is True
    assert cr.can_add_step(current_step+1) is True

    for current_step in range(1, max_len):
        cr._recorded_steps.add(current_step)

    assert cr.can_add_step(current_step) is True
    assert cr.can_add_step(current_step+1) is False
    assert cr.can_add_step(current_step+2) is False


@pytest.mark.category1
def test_checkpoint_recorder_record_metric_current_len():
    """The length of the recorded should only change when a new step is added, not
    when a new metric is added."""
    max_len = 10
    cr = CheckpointRecording(10, f"mock/ckpt_10/path", max_len)
    assert cr.current_len == 0
    cr.record_metric(0, "metric_0", random.random())
    assert cr.current_len == 1
    cr.record_metric(0, "metric_1", random.random())
    assert cr.current_len == 1
    cr.record_metric(0, "metric_2", random.random())
    assert cr.current_len == 1
    cr.record_metric(1, "metric_0", random.random())
    assert cr.current_len == 2
    cr.record_metric(1, "metric_1", random.random())
    assert cr.current_len == 2
    cr.record_metric(2, "metric_2", random.random())
    assert cr.current_len == 3


@pytest.mark.category1
@pytest.mark.parametrize(
    "max_len, num_losses",
    [
        (1, 1),
        (5, 1),
        (10, 1),
        (1, 2),
        (5, 2),
        (10, 2),
        (1, 3),
        (5, 3),
        (10, 3)
    ])
def test_checkpoint_recorder_record(max_len, num_losses):
    """Ensure that the checkpoint recording stores data and correctly observes the max log length."""
    start_step = 0
    cr = CheckpointRecording(
        start_step, f"mock/ckpt_{start_step}/path", max_len)

    step_losses = np.random.random(size=(max_len, num_losses))
    step_accuracies = np.random.random(size=(max_len, num_losses))

    for i in range(max_len):
        losses = {j: step_losses[i, j] for j in range(num_losses)}
        accuracies = {j: step_accuracies[i, j] for j in range(num_losses)}
        print(f"Recording step: {start_step + i}")
        print(f"Current length: {cr.current_len}")
        cr.record_metric(start_step + i, "loss", losses)
        cr.record_metric(start_step + i, "accuracy", accuracies)

    assert cr.current_len == max_len

    with pytest.raises(RecorderIsFullException):
        cr.record_metric(start_step + max_len, {}, {})


@pytest.mark.category1
@pytest.mark.parametrize(
    "loss_labels",
    [
        (("MLM", "NSP")),
        (("Single_Loss")),
        (("Three", "Loss", "Example")),
        ((0, 1, 2, 3))
    ])
def test_checkpoint_recorder_to_json(loss_labels):
    """Test that JSON output is of the expected format with correct data"""
    start_step = 0
    max_len = 5
    cr = CheckpointRecording(
        start_step, f"mock/ckpt_{start_step}/path", max_len)

    cr.losses = {s: random_result(loss_labels) for s in range(
        cr.start_step, cr.start_step+cr.max_len)}
    cr.accuracies = {s: random_result(loss_labels) for s in range(
        cr.start_step, cr.start_step+cr.max_len)}

    cj = cr.to_json()

    for k, v in cj["results"].items():
        assert v == cr.results[k]


@pytest.mark.category1
def test_log_recorder_is_recording():
    """Ensure that is_recording is correctly set depending on whether a recorder has been started"""
    rec = LogRecorder(4, "/mock/recording/path", "/mock/recording/path/ckpt")
    assert rec._current_recorder is None
    assert rec.is_recording() is False

    rec.start_recording("/mock/ckpt/path")
    assert rec._current_recorder is not None
    assert rec.is_recording() is True

    rec.stop_recording()
    assert rec._current_recorder is None
    assert rec.is_recording() is False


@pytest.mark.category1
def test_log_recorder_start_recording(caplog):
    """Correct behaviour for starting a recording:
        1) Creating the checkpoint recorder correctly
        2) Stopping stores the recorder and clears ready for the next start
        3) If a run is still recording, stop the previous one and start anew"""

    MAX_LEN = 5

    def check_ckpt_recorder(rec, e_step, e_path, e_max_length):
        assert rec.is_recording()
        assert rec._current_recorder.start_step == e_step
        assert rec._current_recorder.checkpoint_path == e_path
        assert rec._current_recorder.max_len == e_max_length

    storage_path = "/mock/recording/path/ckpt"
    rel_ckpt_paths = [f"mock_ckpt_{c}/path" for c in range(3)]
    full_ckpt_paths = [os.path.join(storage_path, c) for c in rel_ckpt_paths]

    current_step = 5

    rec = LogRecorder(MAX_LEN, "/mock/recording/path",
                      "/mock/recording/path/ckpt")
    rec.update_step(current_step)

    # (1)
    rec.start_recording(full_ckpt_paths[0])
    check_ckpt_recorder(rec, current_step, rel_ckpt_paths[0], MAX_LEN)

    # (2)
    rec.stop_recording()
    assert rec._current_recorder is None
    assert current_step in rec.checkpoint_logs

    current_step = 10
    rec.update_step(current_step)
    rec.start_recording(full_ckpt_paths[1])
    check_ckpt_recorder(rec, current_step, rel_ckpt_paths[1], MAX_LEN)

    last_step = current_step
    current_step = 20
    rec.update_step(current_step)
    # (3)
    with caplog.at_level(logging.INFO, logger='convergence-harness'):
        rec.start_recording(full_ckpt_paths[2])
        expected_warning = "Already recording logs for the previous checkpoint. Stopping here and starting a fresh log."
        assert caplog.records[0].message == expected_warning
    assert last_step in rec.checkpoint_logs
    assert current_step not in rec.checkpoint_logs
    check_ckpt_recorder(rec, current_step, rel_ckpt_paths[2], MAX_LEN)

    rec.stop_recording()
    assert rec._current_recorder is None
    assert current_step in rec.checkpoint_logs


@pytest.mark.category1
def test_log_recorder_updates(caplog):
    """Ensure correct update behaviour:
        1) If we try to record without having initialised a post-checkpoint recording - warn and ignore
        2) If we have a recorder set, call its record method (internals tested in a separate test - this should be mocked)
        3) If the recorder is full, raise a relevant exception.
    """
    MAX_LEN = 5

    def check_rec(rec, e_step, e_loss, e_acc):
        assert rec.is_recording()
        assert rec._current_recorder.last_step == e_step
        assert rec._current_recorder.metrics["loss"] == e_loss
        assert rec._current_recorder.metrics["accuracy"] == e_acc

    # Stub out the stop and save methods -> we're not testing their behaviour here, just
    # need to make sure they're called correctly for (3)
    mock_status = {"recording_stopped": False, "recording_saved": False}

    def mock_stop_recording():
        mock_status["recording_stopped"] = True

    def mock_save_recording():
        mock_status["recording_saved"] = True
    # Start test
    rec = LogRecorder(MAX_LEN, "/mock/recording/path",
                      "/mock/recording/path/ckpt")
    rec.update_step(0)

    rec.stop_recording = mock_stop_recording
    rec.save = mock_save_recording

    # (1)
    with caplog.at_level(logging.INFO, logger='convergence-harness'):
        rec.record_step_metric("some_metric", {})
        expected_warning = "Trying to record step 0, but recorder is None. Skipping entry."
        assert caplog.records[0].message == expected_warning

    mock_loss = random_result()
    mock_accuracy = random_result()

    cr = MockCheckpointRecording(0)
    rec._current_recorder = cr

    # (2)
    rec.update_step(1)
    assert rec.current_step == 1
    rec.record_step_metric("loss", mock_loss)
    rec.record_step_metric("accuracy", mock_accuracy)
    check_rec(rec, 1, mock_loss, mock_accuracy)

    rec.update_step(5)
    assert rec.current_step == 5

    mock_loss = random_result()
    mock_accuracy = random_result()

    rec.record_step_metric("loss", mock_loss)
    rec.record_step_metric("accuracy", mock_accuracy)
    check_rec(rec, 5, mock_loss, mock_accuracy)

    assert mock_status["recording_stopped"] is False
    assert mock_status["recording_saved"] is False

    # (3)
    rec._current_recorder._can_add = False
    with pytest.raises(RecorderIsFullException):
        rec.record_step_metric("loss", mock_loss)
        rec.record_step_metric("accuracy", mock_accuracy)


@pytest.mark.category1
@pytest.mark.parametrize(
    "num_checkpoints, recording_length, loss_labels",
    [
        (1, 1, ("MLM", "NSP")),
        (1, 5, ("MLM", "NSP")),
        (5, 1, ("MLM", "NSP")),
        (5, 5, ("MLM", "NSP")),
        (5, 5, ("Single_Loss")),
        (5, 5, ("Three", "Loss", "Example")),
        (5, 5, (0, 1, 2, 3))
    ])
def test_log_recorder_to_json(num_checkpoints, recording_length, loss_labels):
    """Test that JSON output is of the expected fornmat with correct data"""

    rec = LogRecorder(recording_length, "/mock/recording/path",
                      "/mock/recording/path/ckpt")

    start_steps = [i * 2 * recording_length for i in range(num_checkpoints)]
    ckpt_recordings = [CheckpointRecording(
        s, f"/mock/ckpt_{i}/path", recording_length) for i, s in enumerate(start_steps)]

    for cr in ckpt_recordings:
        cr._results = {s: random_result(loss_labels) for s in range(
            cr.start_step, cr.start_step+cr.max_len)}
        rec.checkpoint_logs[cr.start_step] = cr

    # Sanity check the mocking code above
    assert len(rec.checkpoint_logs) == num_checkpoints

    json_output = rec.to_json()
    assert len(json_output["ckpt_logs"]) == num_checkpoints

    for i, log_key in enumerate(json_output["ckpt_logs"]):
        cj = json_output["ckpt_logs"][log_key]
        cr = rec.checkpoint_logs[log_key]
        assert cj["checkpoint"] == f"/mock/ckpt_{i}/path"
        assert cj["start_step"] == start_steps[i]

        assert len(cj["results"]) == len(cr._results)

        for step, result in cj["results"].items():
            assert result["losses"] == cr._results[step]["losses"]
            assert result["accuracies"] == cr._results[step]["accuracies"]


@pytest.mark.category1
def test_log_recorder_cannot_add_should_stop(monkeypatch):
    """
    If the checkpoint recorder cannot add another step (due to being full),
    the log recorder should stop and save the checkpoint recorder
    """
    rec = LogRecorder(1, "/mock/recording/path", "/mock/recording/path/ckpt")

    stopped_calls = 0
    saved_calls = 0

    def mock_stop_recording():
        nonlocal stopped_calls
        stopped_calls += 1

    def mock_save():
        nonlocal saved_calls
        saved_calls += 1

    monkeypatch.setattr(rec, "stop_recording", mock_stop_recording)
    monkeypatch.setattr(rec, "save", mock_save)

    rec.start_recording("/mock/ckpt/path")
    rec.update_step(0)
    assert rec._current_recorder.can_add_step(0) is True
    assert stopped_calls == 0
    assert saved_calls == 0

    rec.record_step_metric("metric", {"value": 0})
    assert rec._current_recorder.can_add_step(0) is True
    assert rec._current_recorder.can_add_step(1) is False
    assert stopped_calls == 0
    assert saved_calls == 0

    rec.update_step(1)
    assert stopped_calls == 1
    assert saved_calls == 1


@pytest.mark.category1
@pytest.mark.parametrize(
    "checkpoint_path, is_gather, has_common, has_misc, missing_gather, missing_run, missing_mandatory",
    list(itertools.product(
        ("/checkpoint/path/test/1", "/another/checkpoint/path"),
        (True, False),
        (True, False),
        ("gather", "run", "both"),
        (False, True),
        (False, True),
        (False, True)
    ))
)
def test_flags_from_config(checkpoint_path, is_gather, has_common, has_misc, missing_gather, missing_run, missing_mandatory):
    test_flags, expected_result = build_sample_flag_data(
        checkpoint_path, is_gather, has_common, has_misc, missing_gather, missing_run, missing_mandatory)

    should_succeed = False
    if missing_gather or missing_run:
        ctx = pytest.raises(MissingConfigurationOptionException)
    elif missing_mandatory:
        ctx = pytest.raises(MissingMandatoryFlagException)
    else:
        should_succeed = True
        ctx = does_not_raise()

    if is_gather:
        flag_values = {
            "checkpoint_output": checkpoint_path
        }
    else:
        flag_values = {
            "checkpoint_input": checkpoint_path,
            "start_step": 5,
            "run_for": 3
        }

    with ctx:
        flag_list = flags_from_config(
            test_flags, gather_mode=is_gather, flag_values=flag_values)

    if should_succeed:
        for k, v in expected_result.items():
            assert k in flag_list

            # For argument key/value pairs...
            if v is not None:
                # The value is in the list
                assert str(v) in flag_list
                k_idx = flag_list.index(k)

                # The value is next to its named parameter
                assert flag_list[k_idx+1] == str(v)


@pytest.mark.category1
@pytest.mark.parametrize(
    "expected,observed,margin,operator, should_match",
    [
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.0", "NSP": "1.0"}, 0.000001, "==",  True),
        ({"MLM": "1.0", "NSP": "1.0"}, {"NOTMLM": "1.0",
                                        "NOTMLM": "1.0"}, 0.000001, "==",  False),
        ({"MLM": "1.0", "NSP": "1.0"}, {"NOTMLM": "1.0",
                                        "NOTMLM": "1.0", "EXTRA": "0.3"}, 0.000001, "==",  False),
        ({"MLM": "1.0", "NSP": "1.0"}, {"MLM": "1.0"}, 0.000001, "==",  False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.049", "NSP": "1.049"}, 0.05, "==", True),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "0.950001", "NSP": "0.950001"}, 0.05, "==", True),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.05", "NSP": "1.05"}, 0.05, "==", True),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "0.95", "NSP": "0.95"}, 0.05, "==", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.05", "NSP": "1.1"}, 0.05, "==", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.1", "NSP": "1.1"}, 0.05, "==", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.15", "NSP": "1.1"}, 0.1, "==", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "0.9", "NSP": "1.0"}, 0.05, ">=", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "0.91", "NSP": "1.0"}, 0.05, "<=", True),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.01", "NSP": "1.0"}, 0.05, "<=", True),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.01", "NSP": "1.0"}, 0.05, "!=", False),
        ({"MLM": "1.0", "NSP": "1.0"}, {
         "MLM": "1.1", "NSP": "1.2"}, 0.05, "!=", True),
    ]
)
def test_compare_result(expected, observed, margin, operator, should_match):
    """
    Test the logging metric comparison function. Passes a variety of multi-label observations
    and expected values into the function and checks the results are as intended.

    Includes cases for non-matching labels, operators and boundary conditions
    """
    ctx = does_not_raise() if should_match else pytest.raises(AssertionError)
    with ctx:
        compare_log_result(observed, expected, margin, operator)


@pytest.mark.category1
@pytest.mark.parametrize(
    "output_stream",
    ["stderr", "stdout"]
)
def test_create_run_process(monkeypatch, output_stream):
    proc = None

    def create_MockSubProcess(*args, **kwargs):
        # Get access to the closure's `proc` var
        nonlocal proc
        proc = MockSubProcess(output_stream)
        return proc

    def mock_handler(line, *args, **kwargs):
        assert line == iteration_log_single_loss

    monkeypatch.setattr(subprocess, "Popen", create_MockSubProcess)
    monkeypatch.setattr(convergence_harness,
                        "run_log_handler", mock_handler)

    mock_config = {
        "log_parsing": {},
        "log_output": output_stream,
        "metrics": {}
    }

    convergence_harness.create_run_process(
        None, mock_config, ["prog", "arg1"])

    # Check that the process has been correctly created
    assert proc is not None

    # Ensure we've output to the correct stream
    if output_stream == "stdout":
        assert proc._stdout_called
    elif output_stream == "stderr":
        assert proc._stderr_called

    # Ensure the process was killed
    assert proc._iskilled


@pytest.mark.category1
@pytest.mark.parametrize(
    "output_stream",
    ["stderr", "stdout"]
)
def test_create_gather_process(monkeypatch, output_stream):
    mock_subprocess = None
    mock_log_recorder = None

    manifest_name = "test_manifest.json"
    output_path = "/fake/path/to/"

    def create_MockLogRecorder(*args, **kwargs):
        nonlocal mock_log_recorder
        mock_log_recorder = MockLogRecorder()
        return mock_log_recorder

    def create_MockSubProcess(*args, **kwargs):
        nonlocal mock_subprocess
        mock_subprocess = MockSubProcess(output_stream)
        return mock_subprocess

    def mock_handler(line, *args, **kwargs):
        assert line == iteration_log_single_loss

    monkeypatch.setattr(subprocess, "Popen", create_MockSubProcess)
    monkeypatch.setattr(convergence_harness,
                        "LogRecorder", create_MockLogRecorder)
    monkeypatch.setattr(convergence_harness,
                        "gather_log_handler", mock_handler)

    mock_config = {
        "log_parsing": None,
        "log_output": output_stream,
        "recording": {
            "steps_to_record_after_save": random.randint(2, 20)
        },
        "metrics": {}
    }

    convergence_harness.create_gather_process(
        manifest_name, output_path, mock_config, ["prog", "arg1"], output_path)

    # Check that the mock process and recorder have been correctly created
    assert mock_subprocess is not None
    assert mock_log_recorder is not None

    # Ensure we've output to the correct stream
    if output_stream == "stdout":
        assert mock_subprocess._stdout_called
    elif output_stream == "stderr":
        assert mock_subprocess._stderr_called

    # Ensure the process was killed
    assert mock_subprocess._iskilled

    assert mock_log_recorder._stopped
    assert mock_log_recorder._saved


@pytest.mark.category1
@pytest.mark.parametrize(
    "do_gather, runner_fn_name",
    [
        (True, "gather"),
        (False, "run")
    ]
)
def test_do_convergence_wrappers(monkeypatch, tmp_path, do_gather, runner_fn_name):
    """
    The wrappers should run through all experiments in the YAML config file and generate
    a manifest name accordingly.

    Each item should be processed exactly once.
    """
    manifest_suffix = "manifest.json"
    experiment_names = [f"experiment_{i}" for i in range(5)]
    done_experiments = Counter()

    def mock_yaml_loader(*args, **kwargs):
        nonlocal experiment_names
        return {n: {"directory": ""} for n in experiment_names}

    def mock_runner(storage_path, checkpoint_path, manifest_name, config):
        nonlocal done_experiments
        done_experiments[manifest_name] += 1

    def mock_pass_method(*args, **kwargs):
        pass

    monkeypatch.setattr(yaml, "load", mock_yaml_loader)
    monkeypatch.setattr(os, "chdir", mock_pass_method)
    monkeypatch.setattr(convergence_harness, runner_fn_name, mock_runner)

    config = tmp_path / "tmp_config.json"
    config.write_text("{}")
    do_convergence_test("/mock/storage/path", manifest_suffix, str(config.absolute()), do_gather)

    assert len(experiment_names) == len(done_experiments)
    for name in experiment_names:
        full_manifest_name = f"{name}_{manifest_suffix}"
        assert done_experiments[full_manifest_name] == 1


if __name__ == "__main__":
    pytest.main(args=[__file__, '-vv'])
