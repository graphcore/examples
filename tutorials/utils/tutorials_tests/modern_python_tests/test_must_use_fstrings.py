# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
from typing import List, Optional

from ..testing_util import get_file_list, read_git_submodule_paths

PYTHON_FILE_EXTS = [".py"]

EXCLUDED = ["utils/tutorials_tests/modern_python_tests/test_must_use_fstrings.py"]


def file_uses_string_format(path):
    with open(path, "r") as f:
        file_contents = f.read()

    return ".format(" in file_contents


def test_for_format(file_list: Optional[List[Path]] = None):
    """Test that we are not using old style .format() strings"""

    root_path = Path(__file__).resolve().parents[3]

    if file_list is None:
        file_list = get_file_list(root_path, PYTHON_FILE_EXTS)

    excluded = [root_path / p for p in EXCLUDED]
    git_module_paths = read_git_submodule_paths()

    filtered_file_list = [
        file_path
        for file_path in file_list
        if file_path not in excluded and file_path not in git_module_paths and "/CMakeFiles/" not in str(file_path)
    ]

    bad_files = []
    for file_path in filtered_file_list:

        if file_path.suffix in PYTHON_FILE_EXTS:
            if file_uses_string_format(file_path):
                bad_files.append(file_path)

        else:
            # If we get here then the test script is broken
            raise NotImplementedError(f"Unexpected file type: {file_path}")

    no_failures = not bad_files
    assert no_failures, (
        "Please modernise the code to use f-strings instead of `'{}'.format(val)`:"
        " https://docs.python.org/3/tutorial/inputoutput.html#tut-f-strings\n"
        + "You can use https://pypi.org/project/flynt/ to automate the update\n"
        + "\n".join(str(p) for p in bad_files)
    )
