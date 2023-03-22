# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import argparse
import datetime
import fileinput
from pathlib import Path
import re
import sys

from tutorials_tests.testing_util import get_file_list, read_git_submodule_paths

PYTHON_FILE_EXTS = [".py"]
C_FILE_EXTS = [".c", ".cpp", ".C", ".cxx", ".c++", ".h", ".hpp"]

EXCLUDED = []


def check_file(path, language, amend):
    comment = "#" if language == "python" else "//"
    found_copyright = False
    first_line_index = 0
    empty_file = False
    with open(path, "r", encoding="utf-8") as file:
        first_line = file.readline()
        # if the first line is encoding, then read the second line
        if first_line.startswith(f"{comment} coding=utf-8"):
            first_line = file.readline()

        if first_line == "":
            empty_file = True

        if language == "python" and first_line.startswith("#!"):
            first_line_index += 1
            first_line = file.readline()
        # if the file is for jupyter notebook conversions
        if language == "python" and first_line.startswith('"""'):
            first_line_index += 1
            first_line = file.readline()
            regexp = r"Copyright \(c\) \d+ Graphcore Ltd. All (r|R)ights (r|R)eserved."
            if re.match(regexp, first_line):
                found_copyright = True

        regexp = rf"{comment} Copyright \(c\) \d+ Graphcore Ltd. All (r|R)ights" r" (r|R)eserved."

        if re.match(regexp, first_line):
            found_copyright = True

    if not empty_file and not found_copyright:
        if amend:
            now = datetime.datetime.now()
            year = now.year
            copyright_msg = f"{comment} Copyright (c) {year} Graphcore Ltd. All rights reserved."
            index = 0
            for line in fileinput.FileInput(str(path), inplace=True):
                if index == first_line_index:
                    line = copyright_msg + line
                print(line[:-1])
                index += 1

        return False

    return True


def test_copyrights(amend=False):
    """A test to ensure that every source file has the correct Copyright"""

    root_path = Path(__file__).resolve().parents[3]
    file_list = get_file_list(root_path, PYTHON_FILE_EXTS + C_FILE_EXTS)

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
            if not check_file(file_path, "python", amend):
                bad_files.append(file_path)

        elif file_path.suffix in C_FILE_EXTS:
            if not check_file(file_path, "c", amend):
                bad_files.append(file_path)

        else:
            # If we get here then the test script is broken
            raise NotImplementedError(f"Unexpected file type: {file_path}")

    assert not bad_files, f"{len(bad_files)} files do not have copyright notices: {bad_files}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copyright header test")
    parser.add_argument("--amend", action="store_true", help="Amend copyright headers in files.")

    opts = parser.parse_args()
    try:
        test_copyrights(opts.amend)
    except AssertionError:
        sys.exit(1)
