#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
from subprocess import run
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    ret_val = 0
    for filename in args.filenames:
        if ".ipynb" in filename:
            ret_val |= run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to=notebook",
                    "--ClearMetadataPreprocessor.enabled=True",
                    "--inplace",
                    filename,
                ]
            ).returncode

    return ret_val


if __name__ == "__main__":
    raise SystemExit(main())
