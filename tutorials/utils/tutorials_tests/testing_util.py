# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import configparser
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from statistics import mean
from typing import Container, Dict, List, Union

"""Library of utility functions common between frameworks"""


DEFAULT_PROCESS_TIMEOUT_SECONDS = 40 * 60


class CalledProcessError(subprocess.CalledProcessError):
    """An error for subprocesses which captures stdout and stderr in the error message."""

    def __str__(self) -> str:
        original_message = super().__str__()
        return f"{original_message}\n" f"{self.stdout}\n" f"{self.stderr}"


def parse_results_for_speed(output, iter_tolerance, speed_tolerance):
    """Look for <iter number> sec/itr. <speed number> {other stuff}"""
    found_a_result = False

    for line in output.split("\n"):
        matches = re.match(r"([\d.]+) +sec/itr. +([\d.]+)", line)
        if matches:
            found_a_result = True
            iterations, speed = matches.groups()
            iterations = float(iterations)
            speed = float(speed)
            _verify_model_numbers(iter_tolerance, iterations, speed_tolerance, speed, line)

    if not found_a_result:
        raise AssertionError("No results detected in this run")


def parse_results_for_accuracy(output, expected_accuracies, acc_tolerance):
    """Look for Accuracy=<accuracy>%"""

    accuracies = []
    for line in output.split("\n"):
        if re.match(r" + Accuracy=+([\d.]+)%", line):
            accuracy = float(re.match(r" + Accuracy=+([\d.]+)%", line).groups()[0])
            accuracies.append(accuracy)
        elif re.search(r"Validation accuracy", line):
            accuracy_str = re.search(r"accuracy:\s(.*)", line).group(1)
            accuracy = float(accuracy_str[: accuracy_str.rfind("%")])
            accuracies.append(accuracy)

    if len(accuracies) == 0:
        raise AssertionError("No results detected in this run")
    elif len(accuracies) != len(expected_accuracies):
        raise AssertionError("Expected accuracies and parsed accuracies have" " different lengths")

    verify_model_accuracies(accuracies, expected_accuracies, acc_tolerance)


def _verify_model_numbers(iter_tolerance, iterations, speed_tolerance, speed, line):
    iter_error = ""
    speed_error = ""

    # Verify iteration speed
    if iterations > iter_tolerance[1]:
        iter_error = "The time per iteration has regressed above" " the tolerance maximum: " + str(iter_tolerance[1])
    elif iterations < iter_tolerance[0]:
        iter_error = (
            "Time taken to compete an iteration was "
            "suspiciously fast. Please verify the model"
            " is operating correctly and tune tolerances"
            " accordingly."
        )

    # Verify item processing speed
    if speed < speed_tolerance[0]:
        speed_error = "The number of items processed per second" " has regressed below the tolerance: " + str(
            speed_tolerance[0]
        )
    elif speed > speed_tolerance[1]:
        speed_error = (
            "The number of items processed per second"
            " was suspiciously high. Please verify the"
            " model is behaving correctly and tune"
            " tolerances accordingly."
        )

    if iter_error and speed_error:
        sys.stderr.write("\n".join([line, iter_error, speed_error]))
        raise AssertionError("Timings out of tolerance range")
    elif iter_error or speed_error:
        sys.stderr.write(line)
        raise AssertionError(iter_error + speed_error)


def verify_model_accuracies(accuracies, expected_accuracy, acc_tolerance):
    """Asserts a list of accuracies is within a list of expected accuracies
       with a tolerance applied.

    Args:
        accuracies: A list of floats representing the accuracies (%) produced
            by the model at each step.
        expected_accuracy: A list of floats representing the expected
            accuracies (%) produced by the model at each step.
        acc_tolerance: A float representing a percentage tolerance applied on
            top of the expected accuracies that the accuracies produced by
            the model should sit within.

    Raises:
        Assertion Error: Accuracy produced by the model are not within
            the expected limits.
    """

    for iter_num in range(len(accuracies)):
        exp_acc = expected_accuracy[iter_num]
        acc = accuracies[iter_num]
        acc_str = (
            f"{'Accuracy (%)':<22} = {acc:.2f}\n"
            f"{'Expected accuracy (%)':<22} = {exp_acc} +- {acc_tolerance}"
            f" = [{exp_acc - acc_tolerance:.2f}, {exp_acc + acc_tolerance:.2f}]"
        )
        if acc < exp_acc - acc_tolerance:
            raise AssertionError(
                f"After iteration {iter_num + 1}, the model is less accurate" " than expected.\n" f"{acc_str}"
            )
        elif acc > exp_acc + acc_tolerance:
            raise AssertionError(
                f"After iteration {iter_num + 1}, the model is producing an accuracy"
                " that is suspiciously high and should be reviewed.\n"
                f"{acc_str}"
            )


def parse_results_for_ipus_used(output):
    """Finds the number of IPUs used in the model by looking for
    string with format ' On 2 IPUs.' in output"""
    shards_regex = r" On ([\d.]+) IPUs."
    for line in output.split("\n"):
        matches = re.match(shards_regex, line)
        if matches:
            shards = matches.group(1)
            return int(shards)
    raise AssertionError("Expecting line detailing IPU use " "for example ' On 2 IPUs.'")


def assert_shards(output, expected_shards):
    """Verify the expected number of shards used were actually
    used"""
    actual_shards = parse_results_for_ipus_used(output)
    assert actual_shards == expected_shards


def get_final_accuracy(output):
    """Find and return the accuracy reported in a test's output."""
    result_regex = r"Accuracy=([\d.]+)\%"
    result_list = parse_results_with_regex(output, result_regex)
    result = result_list[0]
    return result[-1]


def get_final_loss(output):
    """Find and return the loss reported in a test's output."""
    result_regex = r"Loss=([\d.]+)"
    result_list = parse_results_with_regex(output, result_regex)
    result = result_list[0]
    return result[-1]


def get_average_speeds(output):
    """Finds the average seconds/iteration and tokens/second

    Args:
        output: String representing the output of a test.

    Returns:
        A tuple where the first element is a float representing
        the average iterations per second and the second the
        average tokens processed per second
    """

    result_regex = r"([\d.]+) +sec/itr. +([\d.]+)"
    results = parse_results_with_regex(output, result_regex)

    itr_sec_list = results[0]
    tokens_sec_list = results[1]

    return mean(itr_sec_list), mean(tokens_sec_list)


def parse_results_with_regex(output, regex):
    """Find and returns the regex matching results in output

    Looks through the output line by line looking for a matching regex.
    The function assembles a list of lists where each parent list is
    the results for that position in the regex string and each item in
    the child lists represents an order of the results found in the output

    Args:
        output: String representing the output of a test.
        regex: Regex of result to find.

    Returns:
        A list of lists of floats.  Parent list represents the result at each
        position in the regex.  Child list contains results received in the
        order they were output.

    Raises:
        AssertionError: a line matching the regex could not be found in the
            output
    """

    results = []

    for line in output.split("\n"):
        matches = re.search(regex, line)
        if matches:
            number_of_results = matches.lastindex
            if results == []:
                results = [None] * number_of_results
            for match_index in range(0, number_of_results):
                result = float(matches.group(match_index + 1))
                if results[match_index]:
                    results[match_index].append(result)
                    continue
                results[match_index] = [result]

    if results == []:
        raise AssertionError(f"Regex {regex} not found in result")

    return results


def get_total_epochs(output):
    """Finds the number of epochs model has run through by looking for
    string with format 'Epoch #3' in the models raw output"""
    epochs = None
    for line in output.split("\n"):
        epoch_match = re.search(r"Epoch #([\d.]+)", line)
        if epoch_match:
            epochs = int(epoch_match.group(1))
    if not epochs:
        raise AssertionError("Epochs not found in output, for example " "Epoch #3")
    return epochs


def assert_total_run_time(total_time, time_range):
    """Checks total run time is within the required range

    Args:
        total_time: float representing number of seconds the test took to
            run
        time_range: a tuple of floats where the first element is the minimum
            time the test should run in in seconds and the second the
            maximum

    Raises:
        AssertionError: if the total_time is not between the minimum time
            and maximum time
    """
    minimum_time = time_range[0]
    maximum_time = time_range[1]
    assert total_time >= minimum_time
    assert total_time <= maximum_time


def assert_final_accuracy(output, minimum, maximum):
    """Gets the final accuracy given a raw model output and checks its value
       is between the minimum and maximum

    Args:
        output: String representing the raw output of a model
        minimum: a float representing a percentage (between 0.0% and 100%)
            that is the minimum accuracy for the model after running
        maximum: a float representing a percentage (between 0.0% and 100%)
            that is the maximum accuracy for the model after running

    Raises:
        AssertionError: if the final accuracy is not between the maximum and
            minimum percentages
    """
    accuracy = get_final_accuracy(output)
    assert accuracy >= minimum
    assert accuracy <= maximum


def run_python_script_helper(
    cwd: str,
    script: Union[str, List[str]],
    want_std_err: bool = False,
    env=None,
    **kwargs,
):
    """A function that given a path and python script name, runs the script
      with kwargs as the command line arguments

    Args:
        cwd: string representing the directory of the python script
        script: string representing the full name of the python script
                can be a list of strings, which will be passed to the python
                command-line. for example ['-c', 'print("Hello")']
        want_std_err: optional - set True to include stderr trace in the output
        env : Optionally pass in the Environment variables to use
        kwargs: dictionary of string key and values that form the command
            line arguments when the script is run.

    Returns:
        A string representing the raw output of the python script run
    """
    versioned_python = f"python{sys.version_info[0]}"

    # Make script param a list if it isn't already, so we can concatenate with
    # versioned_python
    if isinstance(script, str):
        script = [script]

    cmd = [versioned_python] + script

    err = subprocess.STDOUT if want_std_err else subprocess.PIPE
    if kwargs:
        args = [str(item) for sublist in kwargs.items() for item in sublist if item != ""]
        cmd.extend(args)
    out = run_command_fail_explicitly(cmd, stderr=err, cwd=cwd, env=env)
    print(out)
    return out


def run_test_helper(subprocess_function, total_run_time=None, total_run_time_tolerance=0.1, **kwargs):
    """Checks that a function executes within a given time tolerance

    Takes in test keyword parameters, runs the test and checks that the
    test function executed within a tolerance of the specified duration.

    Note:
        If `total_run_time` is not specified this function does nothing.

    Args:
        subprocess_function: the function that runs a subprocess of
            the model in question
        total_run_time: float or None, the expected run time of the
            `subprocess_function` callable.
        total_run_time_tolerance: The range away from `total_run_time` which
            is considered to be acceptable.

    Returns:
        A String representing the raw output of the models subprocess.

    Raises:
        AssertionError: If time taken is not within the expected bounds.
    """

    start_time = time.time()

    out = subprocess_function(**kwargs)

    total_time = time.time() - start_time

    if total_run_time:
        total_run_time_range = range_from_tolerances(total_run_time, total_run_time_tolerance)
        assert_total_run_time(total_time, total_run_time_range)

    return out


def range_from_tolerances(value, tolerance):
    """Helper function that takes a value and applies the tolerance

    Args:
        value: a float representing the mean value to which the tolerance
            will be applied
        tolerance: a float representing a percentage (between 0.0 and 1.0)
            which is applied symmetrically across the value argument

    Returns:
        A tuple of floats, the first element representing the tolerance
        applied below the value (minimum) and the second above (maximum)
    """
    return (
        get_minimum_with_tolerance(value, tolerance),
        get_maximum_with_tolerance(value, tolerance),
    )


def get_minimum_with_tolerance(value, tolerance):
    """Helper function that takes a value and applies the tolerance
       below the value

    Args:
        value: a float representing the mean value to which the tolerance
            will be applied
        tolerance: a float representing a percentage (between 0.0 and 1.0)
            which is applied to the value argument

    Returns:
        A float representing the tolerance applied below the value (maximum)
    """
    return value * (1 - tolerance)


def get_maximum_with_tolerance(value, tolerance):
    """Helper function that takes a value and applies the tolerance
       above the value

    Args:
        value: a float representing the mean value to which the tolerance
            will be applied
        tolerance: a float representing a percentage (between 0.0 and 1.0)
            which is applied to the value argument

    Returns:
        A float representing the tolerance applied above the value (minimum)
    """
    return value * (1 + tolerance)


def check_data_exists(data_path, expected_files_list):
    """Helper function that checks the expected data exists in a directory

    Args:
        data_path: A string representing the directory of where the
            data is expected to be
        expected_files_list: a list of strings representing the expected
            file names in the data_path directory

    Returns:
        A boolean which represents whether the expected files are found in
        the data_path directory
    """

    if os.path.exists(data_path):
        for filename in expected_files_list:
            if not os.path.isfile(os.path.join(data_path, filename)):
                return False
        return True

    return False


def run_command(
    cmd: Union[str, List[str]],
    cwd: str,
    expected_strings: List[str] = [],
    **kwargs,
):
    """Run a command using subprocess, check it ran successfully, and
    check its output for specific strings or regexps.

    Consider using :func:`run_command_fail_explicitly`
    """

    # Allow option of calling with space separated string, but split into
    # List[str] to be consistent below
    if isinstance(cmd, str) and " " in cmd:
        cmd = cmd.split()

    output = run_command_fail_explicitly(cmd, cwd, **kwargs)
    missing_matches = find_missing_patterns(output, expected_strings)
    assert not missing_matches, (
        f"Not all strings were found in the output of command {cmd}, the "
        f"following expected strings were missing: {missing_matches}. "
        f"The following output was produced: {output}"
    )
    return output


def run_command_fail_explicitly(
    command: Union[str, List[str]],
    cwd: str,
    *,
    suppress_warnings: bool = False,
    **kwargs,
) -> str:
    """Runs a command returning the output or failing with useful information

    Args:
        command: The command to execute, can also be a space separated string.
        cwd: The directory in which the command should be
            launched. If called by a pytest test function or method, this
            probably should be a `tmp_path` fixture.
        suppress_warnings: Do not include warnings in stdout, so it can be
                           parsed more reliably. Will still be captured if
                           command raises an exception.
        **kwargs: Additional keyword arguments are passed to
            `subprocess.check_output`.

    Returns:
        The standard output and error of the command if successfully executed.

    Raises:
        RuntimeError: If the subprocess command executes with a non-zero output.
    """

    if suppress_warnings:
        # Warn if parameters contradict
        if "stderr" in kwargs and kwargs["stderr"] != subprocess.PIPE:
            warnings.warn(
                "`run_command_fail_explicitly` parameter `suppress_warnings` will"
                " override other specified parameter `stderr`. Using"
                " `stderr=subprocess.PIPE`",
                stacklevel=2,
            )

        # PIPE rather None, so we can still access from exceptions below
        kwargs["stderr"] = subprocess.PIPE

    DEFAULT_KWARGS = {
        "shell": isinstance(command, str) and " " in command,
        "stderr": subprocess.STDOUT,
        "timeout": DEFAULT_PROCESS_TIMEOUT_SECONDS,
        "universal_newlines": True,
    }

    try:
        merged_kwargs = {**DEFAULT_KWARGS, **kwargs}
        out = subprocess.check_output(
            command,
            cwd=cwd,
            **merged_kwargs,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stdout = e.stdout
        stderr = e.stderr
        # type of the stdout stream will depend on the subprocess.
        # The python docs say decoding is to be handled at
        # application level.
        if hasattr(stdout, "decode"):
            stdout = stdout.decode("utf-8", errors="ignore")
        if hasattr(stderr, "decode"):
            stderr = stderr.decode("utf-8", errors="ignore")
        raise CalledProcessError(1, cmd=command, output=stdout, stderr=stderr) from e
    return out


def find_missing_patterns(string: str, expected_patterns: List[str]) -> List[str]:
    """Finds patterns which are not in a string.

    This function is used to search through the output of commands for
    specific expected patterns.

    Args:
        string: A string which needs to contain the given patterns.
        expected_patterns: regular expression patterns that are expected
            in the string.

    Returns:
        A list with the expected_patterns which were not matched.
    """
    if not expected_patterns:
        return
    # If a string is passed as an argument convert it to a list
    if isinstance(expected_patterns, str):
        expected_patterns = [expected_patterns]

    missing_matches = [expected for expected in expected_patterns if not re.search(expected, string)]

    return missing_matches


def add_args(cmd: List[str], args: Dict) -> List[str]:
    """Takes a command formatted for subprocess and adds arguments from a dictionary.

    Args:
        cmd (List[str]): The command onto which to append arguments.
        args (Dict): The arguments to append. If the value of a given key is `None`, then the argument shall be treated as a flag.

    Returns:
        List[str]: The fully constructed command.
    """
    for k, v in args.items():
        if v is None:
            cmd.append(str(k))
        else:
            cmd.extend([str(k), str(v)])
    return cmd


def get_file_list(root_path: Path, required_types: Container[str], force_full_build: bool = False) -> List[Path]:
    """
    Get list of files, either by reading `diff_file_list.txt` (diff build mode),
    or by walking all sub-folders of `root_path` (full build mode).
    """
    diff_filename = root_path / "diff_file_list.txt"
    if diff_filename.exists() and not force_full_build:
        with open(diff_filename, "r", encoding="utf-8") as diff_file:
            print("Diff builder mode")
            file_list = [
                root_path / file_name
                for file_name in diff_file.read().splitlines()
                if is_required_type(Path(file_name), required_types)
            ]
    else:
        print("Full build mode")
        file_list = [path for path in root_path.rglob("*") if is_required_type(path, required_types)]

    return file_list


def is_required_type(file_path: Path, required_types: Container[str]) -> bool:
    """Is file of one of the required types and also not a special OS file to be
    ignored."""
    return file_path.suffix in required_types and not file_path.name.startswith("._")


def read_git_submodule_paths():
    try:
        config = configparser.ConfigParser()
        config.read(".gitmodules")
        module_paths = [config[k]["path"] for k in config.sections()]
        print(f"Git submodule paths: {module_paths}")
        return module_paths
    except:
        print("No Git submodules found.")
        return []
