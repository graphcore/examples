# Copyright 2020 Graphcore Ltd.

import pytest
import os
import sys
import argparse
import subprocess
import re
import yaml
import logging
import json
import operator
import math
from itertools import chain
from convergence_log_recorder import (
    LogRecorder
)

logger = logging.getLogger("convergence-harness")


class MissingConfigurationOptionException(KeyError):
    pass


class MissingMandatoryFlagException(KeyError):
    pass


def manifest_path(resource_path, manifest_name):
    return os.path.join(resource_path, manifest_name)


def flags_from_config(flags_config, gather_mode=False, flag_values={}):
    """There are four cases that need handling here:
        (1) Flags used in only one of `gather` and `run` operations, where this script sets the value (`config/flags/gather|run/*`)
        (2) Flags used in only one of `gather` and `run` operations, where the value is set in the config (`config/flags/gather|run/misc/*`)
        (3) Flags used in both `gather` and `run` operations, where the value is stored in the config (`config/flags/shared`)

        If this script sets the value, that flag's config should contain only the key entry and the "set_in_script" field, set to true.
        If it is to be loaded from the config, it must not contain the "set_in_script" field.

        This mechanism allows us to differentiate value-less flags from those requiring script input.

        See `demo_config.yml` for an example.
    """

    def flatten_misc_flags(misc_flags):
        return list(chain.from_iterable(flag.values() for flag in misc_flags.values() if "set_in_script" not in flag))

    def gather_specific_flags(gather_flags_config):
        return [
            gather_flags_config["checkpoint_output"]["key"],
            flag_values["checkpoint_output"]
        ]

    def run_specific_flags(run_flags_config):
        return [
            run_flags_config["checkpoint_input"]["key"], flag_values["checkpoint_input"],
            run_flags_config["start_step"]["key"], flag_values["start_step"],
            run_flags_config["run_for"]["key"], flag_values["run_for"]
        ]

    mode_key = "gather" if gather_mode else "run"

    specific_flag_fn = gather_specific_flags if gather_mode else run_specific_flags

    if "run" not in flags_config or "gather" not in flags_config:
        raise MissingConfigurationOptionException(
            "Both run and gather configurations must be supplied in the test config.")

    try:
        # (1)
        flags = specific_flag_fn(flags_config[mode_key])

        # (2)
        if "misc" in flags_config[mode_key]:
            flags += flatten_misc_flags(flags_config[mode_key]["misc"])

        # (3)
        if "common" in flags_config:
            flags += flatten_misc_flags(flags_config["common"])

        return [str(f) for f in flags]
    except KeyError as err:
        raise(MissingMandatoryFlagException(err))


def parse_log_regex(line, item):
    regex = item["regex"]
    labels = item.get("labels", None)

    m = re.search(regex, line)
    if m is None:
        return None

    matches = m.groups()
    if labels is not None and len(labels) != len(matches):
        # If a software change causes the loss groupings to differ, handle the mis-match
        # somewhat gracefully.
        logger.warning(
            "# labels != # matches, falling back to numeric indices")
        labels = None

    if labels is None:
        return dict(zip(range(len(matches)), matches))
    else:
        return dict(zip(labels, matches))


def gather_log_handler(line, logging_config, metrics_config, log_recorder):
    print(line.strip())

    # Check if we're saving a checkpoint, if so, we will need to trigger log recording mode
    checkpoint_save = parse_log_regex(line, logging_config["model_save"])
    if checkpoint_save is not None:
        log_recorder.start_recording(checkpoint_save[0])
        return

    step = parse_log_regex(line, logging_config["step_num"])
    if step is None:
        return

    step = int(step[0])
    log_recorder.update_step(step)

    if log_recorder.is_recording():
        for metric in metrics_config:
            if metric not in logging_config:
                logger.warning(
                    f"Cannot parse requested metric: {metric} - parser not supplied in the config. Skipping.")
                continue

            metric_value = parse_log_regex(line, logging_config[metric])
            log_recorder.record_step_metric(metric, metric_value)


def create_gather_process(output_path, manifest_name, full_config, process_cmd, ckpt_root):
    # Disable output buffering to make sure we get the process logs as they're recorded.
    os.environ['PYTHONUNBUFFERED'] = "1"
    proc = subprocess.Popen(process_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            )

    log_recorder = LogRecorder(full_config["recording"]["steps_to_record_after_save"],
                               manifest_path(output_path, manifest_name),
                               ckpt_root)
    try:
        while proc.poll() is None:
            if full_config["log_output"] == "stderr":
                line = proc.stderr.readline()
            else:
                line = proc.stdout.readline()
            gather_log_handler(
                line, full_config["log_parsing"], full_config["metrics"], log_recorder)
    finally:
        log_recorder.stop_recording()
        log_recorder.save()
        proc.kill()


def compare_log_result(test_values, expected_values, margin, comparator):
    # Check that labels match
    assert test_values.keys() == expected_values.keys(
    ), f"{test_values.keys()} != {expected_values.keys()}"

    # For each metric measurement, check it's within the fair bounds
    for k, tv in test_values.items():
        ev = float(expected_values[k])
        tv = float(tv)

        is_valid = False
        if "=" in comparator:
            fp_eq = math.isclose(ev, tv, rel_tol=margin)
            is_valid = not fp_eq if comparator == "!=" else fp_eq

            # If it's a straight (in)equality, we should test here and move on,
            # otherwise we need to check we're within
            if comparator in ("==", "!="):
                assert is_valid, f"Metric [{k}] was invalid"
                continue

            # If it's valid at this point, we don't check < or >
            if is_valid:
                continue

        # We know it's not equal (within margin), so now check it's less-than or
        # greater-than (depending on metric)
        comparator_op = operator.gt if ">" in comparator else operator.lt
        assert comparator_op(tv, ev), f"{tv} {comparator} {ev} is False"


def run_log_handler(line, logging_config, metrics_config, manifest_results):

    print(line.strip())

    # Check if we're saving a checkpoint, if so, we will need to trigger log recording mode
    step = parse_log_regex(line, logging_config["step_num"])
    if step is None:
        return

    # The manifest is loaded from JSON so all keys are strings
    step = str(step[0])
    if step in (str(k) for k in manifest_results):
        for metric in metrics_config:
            if metric not in logging_config:
                logger.warning(
                    f"Cannot parse requested metric: {metric} - parser not supplied in the config. Skipping.")
                continue

            if metric not in manifest_results[step]:
                logger.warning(
                    f"Cannot compare requested metric: {metric} - result not present in manifest. Skipping.")
                continue

            metric_value = parse_log_regex(line, logging_config[metric])
            # test_values, expected_values, margin, comparison
            margin = metrics_config[metric]["margin"]
            comparator = metrics_config[metric]["comparison"]
            compare_log_result(
                metric_value, manifest_results[step][metric], margin, comparator)


def create_run_process(manifest_results, full_config, process_cmd):
    os.environ['PYTHONUNBUFFERED'] = "1"
    print(f"Running command: {' '.join(process_cmd)}")
    proc = subprocess.Popen(process_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            )

    try:
        while proc.poll() is None:
            if full_config["log_output"] == "stderr":
                line = proc.stderr.readline()
            else:
                line = proc.stdout.readline()
            run_log_handler(
                line, full_config["log_parsing"], full_config["metrics"], manifest_results)
    finally:
        proc.kill()


def gather(output_path, checkpoint_path, manifest_name, test_config):

    flag_values = {
        "checkpoint_output": checkpoint_path
    }

    flags = flags_from_config(
        test_config["flags"], gather_mode=True, flag_values=flag_values)

    # All arguments MUST be strings for Popen
    process_args = [str(p) for p in test_config["executable"] + flags]
    create_gather_process(output_path, manifest_name,
                          test_config, process_args, checkpoint_path)


def run(resource_path, checkpoint_path, manifest_name, test_config):

    with open(manifest_path(resource_path, manifest_name)) as fh:
        manifest = json.load(fh)

    for step_at_save, entry in manifest["ckpt_logs"].items():
        ckpt_path = entry["checkpoint"]

        # Now need to run the application, supplying this path
        flag_values = {
            "checkpoint_input": os.path.join(checkpoint_path, ckpt_path),
            "start_step": step_at_save,
            "run_for": int(test_config["recording"]["steps_to_record_after_save"]) + int(step_at_save)
        }

        flags = flags_from_config(
            test_config["flags"], gather_mode=False, flag_values=flag_values)

        process_args = [str(p) for p in test_config["executable"] + flags]
        create_run_process(entry["results"], test_config, process_args)


def do_convergence_test(storage_path, manifest_suffix, test_config_path, do_gather=False):
    checkpoint_path = os.path.join(storage_path, "ckpts")

    test_fn = gather if do_gather else run
    with open(test_config_path) as fh:
        all_test_config = yaml.load(fh, Loader=yaml.FullLoader)
        for key in all_test_config:
            manifest_name = f"{key}_{manifest_suffix}"

            sys.path.append(all_test_config[key]["directory"])

            # Set the CWD to match the application location to avoid nasty path issues.
            os.chdir(all_test_config[key]["directory"])

            test_fn(storage_path, checkpoint_path,
                    manifest_name, all_test_config[key])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gather", action="store_true",
                        help="Gather test checkpoints for later testing.")
    parser.add_argument("-t", "--test-config-path", required=True,
                        help="Path containing the test configuration file.")
    parser.add_argument("-s", "--storage-path", required=True,
                        help="Path in which the gathered checkpoints and manifest are stored.")
    parser.add_argument("--manifest-name", type=str, default="convergence_harness_manifest.json",
                        help="The name of the manifest file. Used read-only for `run` mode "
                             "or write-only for `gather`.")

    args = parser.parse_args()

    # We change the cwd in the harness (to avoid issues in the target app), so we need to make the
    # output directory path absolute if it's provided as relative
    if not os.path.isabs(args.storage_path):
        args.storage_path = os.path.abspath(args.storage_path)
    return args


def main():
    args = parse_args()

    do_convergence_test(args.storage_path,
                        args.manifest_name,
                        args.test_config_path,
                        args.gather)


if __name__ == "__main__":
    main()
