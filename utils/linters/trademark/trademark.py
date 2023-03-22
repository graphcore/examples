#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import argparse
from bisect import bisect
from typing import Optional
from typing import Sequence
from pathlib import Path

__all__ = ["TrademarkLinter"]

# {"incorrect": "correct"}
TRADEMARK_DICT = {
    "Tensorflow": "TensorFlow",
    "Pytorch": "PyTorch",
    "Poptorch": "PopTorch",
    **{f"Pop{a}{r}{t}": "PopART" for a in "aA" for r in "rR" for t in "tT" if f"{a}{r}{t}" != "ART"},
}
# Characters which indicate a match is not code
VALID_START_END_CHAR = re.compile("[\s'\"\[\]]")
# Patterns that should always be skipped
SKIP_PATTERNS = [
    "utils/linters/trademark",
    "requirements.txt",
    ".jpeg",
    ".jpg",
    ".png",
    ".pth",
    ".tfrecord",
    ".gz",
    ".pt",
]


def apply_lint_function(file_path: str) -> int:
    """Lint function to be called by pre-commit.

    Args:
        file_path (str): The path to the file to be linted.

    Returns:
        int: If there is no modification to the source file the function returns 0,
            else it will rewrite the file and return 1
    """
    linter_message = ""
    path = Path(file_path)
    file_contents = path.read_text(encoding="utf-8")
    new_contents = file_contents
    matches = list(re.finditer("|".join(TRADEMARK_DICT.keys()), file_contents))
    if len(matches) > 0:
        offset = 0
        corrections = []
        lines = [m.end() for m in re.finditer(".*\n", file_contents)]
        for m in matches:
            start, end = m.start(0) + offset, m.end(0) + offset
            correction = TRADEMARK_DICT[m[0]]
            # Skip matches which may be in code.
            start_is_separator = start <= 0 or VALID_START_END_CHAR.match(new_contents[start - 1])
            end_is_separator = VALID_START_END_CHAR.match(new_contents[end])
            if not start_is_separator or not end_is_separator:
                continue
            new_contents = new_contents[:start] + correction + new_contents[end:]
            offset += len(correction) - len(m[0])
            corrections.append(
                f" Fixing ERROR in {file_path}:{bisect(lines, start)+1} - Replacing '{m[0]}' with '{correction}'"
            )
        linter_message = "\n".join(corrections)

    if linter_message == "":
        return 0
    else:
        print(linter_message)
        path.write_text(new_contents, encoding="utf-8")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    ret_val = 0
    for filename in args.filenames:
        if any(skip_pattern in filename for skip_pattern in SKIP_PATTERNS):
            continue
        cur_ret_val = apply_lint_function(filename)
        ret_val |= cur_ret_val

    return ret_val


if __name__ == "__main__":
    raise SystemExit(main())
