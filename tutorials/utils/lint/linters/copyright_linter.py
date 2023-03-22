#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Match, Optional, Sequence, Tuple

import git  # type: ignore

__all__ = ["CopyrightLinter"]

LINES_CHECKED = 5
GC_COPYRIGHT_NOTICE_PATTERN = r"[\/*# ]*Copyright (\xa9|\(c\)) (?P<year>\d{4}) Graphcore Ltd\. All rights reserved\."


def generate_copyright_notice(year: int) -> str:
    return f"Copyright (c) {year} Graphcore Ltd. All rights reserved."


class CopyrightLinter:
    """Linter which inserts a Graphcore copyright if non-existent.

    The format of the notice is determined from the source file extension.

    The file contents are processed in one of three ways:
    - The file contains a valid notice already, so it is not modified.
    - The file contains no string matching a notice exactly or partially,
      so a new notice is inserted at the top of the file (unless the first
      line is a shebang, in which case we insert one immediately after the
      first line) with the year it was added to the git repository.
    - The file contains a string which matches a copyright notice partially,
      but not exactly. This might happen is the notice contains a typo, or
      some syntactical differences for example. In order to avoid two notices
      which are technically different, but appear almost identical to the eye
      we replace the partially-matching notice with a correct one.
    """

    def __init__(self) -> None:
        self.linter_message = ""
        self._git = git.Git(".")

    def apply_lint_function(self, file_path: str) -> int:
        """Lint function to be called by pre-commit.

        Args:
            file_path (str): The path to the file to be linted.

        Returns:
            int: If there is no modification to the source file the function returns 0,
              else it will rewrite the file and return 1
        """
        path = Path(file_path)
        file_contents = path.read_text(encoding="utf-8")
        new_contents = self._determine_linter_message(path, file_contents)

        if not self.linter_message:
            return 0
        else:
            print(f"Fixing ERROR in {file_path}: {self.linter_message}")
            path.write_text(new_contents, encoding="utf-8")
            return 1

    def _determine_linter_message(self, file_path: Path, file_contents: str) -> str:
        """Determine the linter message (if any) and return the (possibly modified) file content.

        Args:
            file_path: The path to the file to be linted.
            file_contents: The content of the file.

        Returns:
            The file contents which has been modified in the case of a missing copyright notice
        """
        lines = file_contents.splitlines(keepends=True)
        # We only search for the copyright notice in the first
        # n lines in a file
        new_contents = file_contents
        target_lines = lines[:LINES_CHECKED]
        match, index = self._match_copyright(target_lines)
        partial_index = self._partial_match(target_lines)
        if match:
            year = int(match.group("year"))
            # The copyright notice must be after graphcore
            # was founded and can't be in the future
            current_year = datetime.now().year
            if year < 2016 or year > current_year:
                self.linter_message = f"Invalid year in copyright notice. Should be <={current_year} and >=2016."
                new_contents = self._insert_copyright_notice(file_path, lines, index=index, replace_notice=True)
        elif partial_index != -1:
            self.linter_message = "Copyright notice has errors."
            new_contents = self._insert_copyright_notice(file_path, lines, index=partial_index, replace_notice=True)

        elif lines:
            self.linter_message = "No copyright notice in file."
            new_contents = self._insert_copyright_notice(file_path, lines)
        # If its an empty file then we just include the notice
        else:
            self.linter_message = "No copyright notice in file."
            new_contents = self._determine_notice_from_name(file_path)
        return new_contents

    def _match_copyright(self, lines: List[str]) -> Tuple[Optional[Match[str]], int]:
        for i, line in enumerate(lines):
            m = re.search(GC_COPYRIGHT_NOTICE_PATTERN, line)
            if m:
                return m, i
        return None, -1

    def _insert_copyright_notice(
        self,
        file_path: Path,
        lines: List[str],
        index: int = 0,
        replace_notice: bool = False,
    ) -> str:
        notice = self._determine_notice_from_name(file_path)
        self.linter_message += f" It should be: '{notice.strip()}'"
        if replace_notice:
            lines[index] = notice
        # If the first line is a shebang
        # we insert the notice just after it
        elif ("#!" in lines[0] or "<?php" in lines[0]) and index == 0:
            lines.insert(1, notice)
        else:
            lines.insert(index, notice)
        return "".join(lines)

    def _determine_notice_from_name(self, path: Path) -> str:
        """Determine how the copyright notice comment should appear in a given file.
        This depends on the filename extension of the file because we use this to
        find out what the comment delimiter should be. For example, if a filename
        ends in .py we know the comment delimiter should be '#'.

        If we cannot find a filename extension pattern in notice_options which
        matches the filename, raise an error. Ideally the exclude and include
        filters should guarantee that the file we are linting has an extension
        which we support.
        """
        git_log_args = ["--follow", "--format=%ad", "--date=format:%Y", path]
        creation_year = self._git.log(*git_log_args).split("\n")[-1]
        creation_year = creation_year if creation_year else datetime.now().year
        notice = generate_copyright_notice(creation_year)
        notice_options = {
            f"# {notice}\n": r"\.(capnp|cmake|py|sh|txt|yaml|yml)",
            f"// {notice}\n": r"\.(c|C|cc|cpp|cxx|h|hpp|php)",
            f".. {notice}\n": r"\.(rst)",
            f"<!-- {notice} -->\n": r"\.(md)",
        }
        notice_comment = None
        for notice_option, extensions in notice_options.items():
            notice_comment = notice_option if re.match(extensions, path.suffix) is not None else notice_comment
        if notice_comment is None:
            raise RuntimeError(
                f"Unsupported file type passed to linter: {path}\n" "Please check your linter configuration.\n"
            )
        return notice_comment

    def _partial_match(self, lines: List[str]) -> int:
        """
        Check the lines of the file for a comment which is a close match to the copyright notice.

        This is often useful for files which do contain copyright notices, but they
        have some syntactical or format errors which cause them not to match the
        notice regular expression. Instead of  inserting a new notice, and creating
        a file which contains two notices that appear to be basically the same we
        instead replace any line in the file which reasonably looks like a good
        copyright notice to the eye, avoiding writing two notices.
        """
        for i, line in enumerate(lines):
            s = SequenceMatcher(
                None,
                generate_copyright_notice(datetime.now().year).upper().strip(),
                line.upper().strip(),
            )
            # According to the python documentation, a ratio() value
            # over 0.6 means the sequences are close matches.
            # https://docs.python.org/3/library/difflib.html#sequencematcher-examples
            if s.ratio() > 0.6:
                return i
        return -1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    ret_val = 0
    for filename in args.filenames:
        copyright_linter = CopyrightLinter()
        cur_ret_val = copyright_linter.apply_lint_function(filename)
        ret_val |= cur_ret_val

    return ret_val


if __name__ == "__main__":
    raise SystemExit(main())
