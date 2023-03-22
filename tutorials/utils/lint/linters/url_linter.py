#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
from typing import Optional
from typing import Sequence
import contextlib
import io

from ...tutorials_tests.urls_test import test_urls


def apply_lint_function() -> int:
    """Lint function to be called by pre-commit. Simply calls the URL tests
    does nothing itself.

    Returns:
        int: If there is no modification to the source file the function returns 0,
            else it will rewrite the file and return 1
    """
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            test_urls.test_all_internal_links()
            test_urls.test_links_are_pegged()
        return 0
    except AssertionError as err:
        print("Link test failed on:")
        print(str(err))
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)
    return apply_lint_function()


if __name__ == "__main__":
    raise SystemExit(main())
