# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

from statistics import mean
import os
import re
import subprocess
import sys
import json
import time
import unittest
from typing import List


"""Library of utility functions common between frameworks"""


def remote_buffers_available():
    output = subprocess.check_output(["gc-inventory -j"], shell=True)
    try:
        return json.loads(output)["devices"][0]["remote buffers"] == "1"
    except (KeyError, IndexError):
        return False


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
            _verify_model_numbers(
                iter_tolerance, iterations, speed_tolerance, speed, line
            )

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
            accuracy = float(accuracy_str[:accuracy_str.rfind("%")])
            accuracies.append(accuracy)

    if len(accuracies) == 0:
        raise AssertionError("No results detected in this run")
    elif len(accuracies) != len(expected_accuracies):
        raise AssertionError("Expected accuracies and parsed accuracies have"
                             " different lengths")

    verify_model_accuracies(accuracies, expected_accuracies, acc_tolerance)


def _verify_model_numbers(iter_tolerance, iterations,
                          speed_tolerance, speed, line):
    iter_error = ""
    speed_error = ""

    # Verify iteration speed
    if iterations > iter_tolerance[1]:
        iter_error = ("The time per iteration has regressed above"
                      " the tolerance maximum: " +
                      str(iter_tolerance[1]))
    elif iterations < iter_tolerance[0]:
        iter_error = ("Time taken to compete an iteration was "
                      "suspiciously fast. Please verify the model"
                      " is operating correctly and tune tolerances"
                      " accordingly.")

    # Verify item processing speed
    if speed < speed_tolerance[0]:
        speed_error = ("The number of items processed per second"
                       " has regressed below the tolerance: " +
                       str(speed_tolerance[0]))
    elif speed > speed_tolerance[1]:
        speed_error = ("The number of items processed per second"
                       " was suspiciously high. Please verify the"
                       " model is behaving correctly and tune"
                       " tolerances accordingly.")

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
        exp_acc_str = (
            "{0} = {1} +- {2} = [{3:.{5}f}, {4:.{5}f}]".format(
                "Expected accuracy (%)".ljust(22),
                exp_acc,
                acc_tolerance,
                exp_acc - acc_tolerance,
                exp_acc + acc_tolerance,
                2
            )
        )
        acc = accuracies[iter_num]
        acc_str = "{} = {:.{}f}".format(
            "Accuracy (%)".ljust(22),
            acc,
            2
        )
        full_acc_str = "{}\n{}".format(acc_str, exp_acc_str)
        if acc < exp_acc - acc_tolerance:
            raise AssertionError(
                "After iteration {}, the model is less accurate"
                " than expected.\n"
                "{}".format(iter_num + 1, full_acc_str)
            )
        elif acc > exp_acc + acc_tolerance:
            raise AssertionError(
                "After iteration {}, the model is producing an accuracy"
                " that is suspiciously high and should be reviewed.\n"
                "{}".format(iter_num + 1, full_acc_str)
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
    raise AssertionError("Expecting line detailing IPU usage "
                         "eg. ' On 2 IPUs.'")


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
        raise AssertionError("Regex {} not found in result".format(regex))

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
        raise AssertionError("Epochs not found in output, eg. "
                             "Epoch #3")
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


def run_python_script_helper(cwd, script, want_std_err=False, env=None, **kwargs):
    """A function that given a path and python script name, runs the script
      with kwargs as the command line arguments

    Args:
        cwd: string representing the directory of the python script
        script: string representing the full name of the python script
        want_std_err: optional - set True to include stderr trace in the output
        env : Optionally pass in the Environment variables to use
        kwargs: dictionary of string key and values that form the command
            line arguments when the script is run.

    Returns:
        A string representing the raw output of the python script run
    """
    py_version = "python{}".format(sys.version_info[0])
    cmd = [py_version, script]
    err = subprocess.STDOUT if want_std_err else subprocess.PIPE
    if kwargs:
        args = [
            str(item) for sublist in kwargs.items() for item in sublist if item != ""
        ]
        cmd.extend(args)
    out = subprocess.check_output(cmd, stderr=err, cwd=cwd, env=env, universal_newlines=True)
    print(out)
    return out


def run_test_helper(subprocess_function, total_run_time=None,
                    total_run_time_tolerance=0.1, **kwargs):
    """Helper function for running tests

    Takes in testable parameters, runs the test and checks the relevant
    parameters against test results

    Args:
        subprocess_function: the function that runs a subprocess of
            the model in question
        total_run_time_range: tuple float representing the expected
            upper and lower bounds for the total time taken to run
            the test

    Returns:
        A String representing the raw output of the models subprocess

    Raises:
        AssertionError: If the accuracy, time taken etc. are not within
            the expected bounds
    """

    start_time = time.time()

    out = subprocess_function(**kwargs)

    total_time = time.time() - start_time

    if total_run_time:
        total_run_time_range = range_from_tolerances(
            total_run_time, total_run_time_tolerance
        )
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


class SubProcessChecker(unittest.TestCase):
    """
    Utility Module for building tests that reliably check if a
    sub-process ran successfully.

    Commonly with an integration/system test you want to check
    a command can be run successfully and gives some expected
    output.

    How to use:
    1. Make a test case in the normal way but inherit from
    test_util.SubProcessChecker instead of unitteset.TestCase.
    2. Define a test method in your derived class in the normal way.
    3. Have the test method call self.run_command(...) and the output
    will be checked automatically.
    """


    def _check_output(self, cmd, output: str, must_contain: List[str]):
        """
        Internal utility used by run_command(...) to check output
        (Should not need to call this directly from test cases).
        """
        if not must_contain:
            return
        # If a string is passed in convert it to a list
        if isinstance(must_contain, str):
            must_contain = [must_contain]
        # Build a list of regexes then apply them all.
        # Each must have at least one result:
        regexes = [re.compile(s) for s in must_contain]
        for i, r in enumerate(regexes):
            match = r.search(output)
            if not match:
                self.fail(f"Output of command: '{cmd}' contained no match for: '{must_contain[i]}'\nOutput was:\n{output}")


    def run_command(self, cmd, working_path, expected_strings, env=None, timeout=None):
        """
        Run a command using subprocess, check it ran successfully, and
        check its output.

        Args:
            cmd:
                Command string. It will be split into args internallly.
            working_path:
                The working directory in which to run the command.
            expected_strings:
                List of strings that must appear in the output at least once.
            env:
                Optionally pass in the Environment variables to use
            timeout:
                Optionally pass in the timeout for running the command
            Returns:
                Output of the command (combined stderr and stdout).
        """
        if isinstance(cmd, list):
            cmd_list = cmd
        else:
            cmd_list = cmd.split()

        if env is None:
            completed = subprocess.run(args=cmd_list,
                                       cwd=working_path,
                                       shell=False,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       timeout=timeout)
        else:
            completed = subprocess.run(args=cmd_list, cwd=working_path,
                                       shell=False, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       env=env,
                                       timeout=timeout)
        combined_output = str(completed.stdout, 'utf-8')
        try:
            completed.check_returncode()
            return_code_ok = True
        except subprocess.CalledProcessError:
            return_code_ok = False

        if not return_code_ok:
            self.fail(f"The following command failed: {cmd}\nWorking path: {working_path}\nOutput of failed command:\n{combined_output}")

        self._check_output(cmd, combined_output, expected_strings)
        return combined_output
