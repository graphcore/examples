# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
from trademark import apply_lint_function

TEST_FILE_CONTENTS = """PopArT # test start of file (should change)
_Popart  # this is valid code (should not change)

Test links: Should change in the alt-text not the URL
[PopaRt is in a link](but-also-Popart/-inaURL)

PopartOptions should stay as is as it is code.
"""
GROUND_TRUTH = """PopART # test start of file (should change)
_Popart  # this is valid code (should not change)

Test links: Should change in the alt-text not the URL
[PopART is in a link](but-also-Popart/-inaURL)

PopartOptions should stay as is as it is code.
"""


def test_trademark_linter(tmp_path: Path):
    """Checks that the TM Linter works as expected on a file with
    a few PopART mis-capitalisations"""
    test_file = tmp_path / "test.md"
    test_file.write_text(TEST_FILE_CONTENTS)
    apply_lint_function(test_file)
    assert test_file.read_text() == GROUND_TRUTH
