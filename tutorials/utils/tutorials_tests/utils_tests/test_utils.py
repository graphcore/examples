# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
from tutorials_tests.testing_util import (
    run_python_script_helper,
    run_command_fail_explicitly,
)
import subprocess
import sys


def test_run_python_script_helper_capture_stdout():
    output = run_python_script_helper(cwd=".", script=["-c", "print('Hello stdout')"])
    assert output == "Hello stdout\n"


def test_run_python_script_helper_dont_capture_stderr():
    output = run_python_script_helper(
        cwd=".",
        script=["-c", "import sys; print('Hello stderr', file=sys.stderr)"],
        want_std_err=False,
    )
    assert output == ""


def test_run_python_script_helper_capture_stderr():
    output = run_python_script_helper(
        cwd=".",
        script=["-c", "import sys; print('Hello stderr', file=sys.stderr)"],
        want_std_err=True,
    )
    assert output == "Hello stderr\n"


def test_run_command_fail_explicitly_fail_capture():
    """A test to ensure that if an error occurs run_command_fail_explicitly returns a subprocess.CalledProcessError containing the error output"""
    # throw an error by calling run_command_fail_explicitely
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command_fail_explicitly([sys.executable, "non_existant_file.py"], ".")
    # Capture error subprocess.CalledProcessError
    assert "can't open file 'non_existant_file.py': [Errno 2] No such file or directory" in str(error.value)


if __name__ == "__main__":
    pytest.main()
