# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
from typing import List

from ..testing_util import get_file_list, read_git_submodule_paths

PYTHON_FILE_EXTS = [".py"]

EXCLUDED = [
    "utils/tutorials_tests/modern_python_tests/test_must_use_pytest.py",
]


def old_style_test_file(path):
    with open(path, "r") as f:
        file_contents = f.read()

    return "SubProcessChecker" in file_contents or "unittest" in file_contents


def test_must_use_pytest(file_list: List[Path] = None):
    """Test that we are not using old style SubProcessChecker or unittest"""

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
            if old_style_test_file(file_path):
                bad_files.append(file_path)
        else:
            # If we get here then the test script is broken
            raise NotImplementedError(f"Unexpected file type: {file_path}")

    no_failures = not bad_files
    assert no_failures, (
        "Please upgrade the test scripts. The use of unittest and SubProcessChecker"
        " have been deprecated in favour of pytest or helpers like"
        " testing_util.run_command\nSee D71447 as an example of how to upgrade\n" + "\n".join(str(p) for p in bad_files)
    )
