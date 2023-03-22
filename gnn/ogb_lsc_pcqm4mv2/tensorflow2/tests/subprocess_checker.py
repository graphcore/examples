# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import re
import subprocess
import unittest
from typing import List


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
                self.fail(
                    f"Output of command: '{cmd}' contained no match " f"for: '{must_contain[i]}'\nOutput was:\n{output}"
                )

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
            completed = subprocess.run(
                args=cmd_list,
                cwd=working_path,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        else:
            completed = subprocess.run(
                args=cmd_list,
                cwd=working_path,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
        combined_output = str(completed.stdout, "utf-8")
        try:
            completed.check_returncode()
            return_code_ok = True
        except subprocess.CalledProcessError:
            return_code_ok = False

        if not return_code_ok:
            self.fail(
                f"The following command failed: {cmd}\n"
                f"Working path: {working_path}\n"
                f"Output of failed command:\n{combined_output}"
            )

        self._check_output(cmd, combined_output, expected_strings)
        return combined_output

    def parse_result_for_metrics(self, output):
        losses, throughputs = [], []

        match_loss_re = re.compile(r"loss: ([\d.]+)")
        throughput_re = re.compile(r"^Throughput: ([0-9.,]+) samples")

        for line in output.split("\n"):
            match_loss = match_loss_re.search(line)
            throughput = throughput_re.match(line)

            if match_loss:
                losses.append(float(match_loss.groups()[0]))
            elif throughput:
                throughputs.append(float(throughput.groups()[0]))

        # Print output if it failed to run (no losses or accuracies found)
        if not losses:
            self.fail("No losses were detected at all.")
        return losses, throughputs

    def loss_seems_reasonable(self, losses, min_expected_loss, max_expected_loss):
        # Test that loss at end is less than loss at start
        if not (losses[-1] < losses[0] or (min_expected_loss < losses[0] < max_expected_loss)):
            self.fail(f"The losses do not seem reasonable.")
        # the loss should converge to ln(2) (coinflip)
        if not min_expected_loss < losses[-1] < max_expected_loss:
            self.fail(
                f"The loss is not converging to the desired value.\n"
                f"Achieved loss {losses[-1]},"
                f" expected between {min_expected_loss} and {max_expected_loss}"
            )
