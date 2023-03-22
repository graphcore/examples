#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import contextlib
import io
from pathlib import Path
from typing import List, Optional, Sequence

from ...tutorials_tests.modern_python_tests import test_must_use_fstrings


def apply_lint_function(changed_files: List[Path]) -> int:
    """Lint function to be called by pre-commit. Simply calls the test script
    does nothing itself.

    Return 0 on success, 1 on failure.
    """
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            test_must_use_fstrings.test_for_format(changed_files)
        return 0
    except AssertionError as err:
        print("test_must_use_fstrings failed with:")
        print(str(err))
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    root_path = Path(__file__).resolve().parents[3]
    path_list = [root_path / file for file in args.filenames]

    return apply_lint_function(path_list)


if __name__ == "__main__":
    raise SystemExit(main())
