# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import json
import os
from pathlib import Path

import pytest

os.chdir(Path(__file__).parent.parent)
from ipu_utils.arguments import (
    StoreFalseOverridable,
    StoreTrueOverridable,
)

EXPECTED = {
    "true": True,
    "t": True,
    "1": True,
    "yes": True,
    "y": True,
    "false": False,
    "f": False,
    "0": False,
    "no": False,
    "n": False,
}


def generate_default_cases(first_item):
    return [(first_item,) + x for x in list(itertools.combinations_with_replacement([None, False, True], 2))]


@pytest.mark.parametrize(
    "input_args, default_true, default_false",
    generate_default_cases([])
    + generate_default_cases(["--true-item"])
    + generate_default_cases(["--true-item", "True"])
    + generate_default_cases(["--true-item", "False"])
    + generate_default_cases(["--false-item"])
    + generate_default_cases(["--false-item", "True"])
    + generate_default_cases(["--false-item", "False"])
    + generate_default_cases(["--false-item", "False", "--true-item", "False"]),
)
def test_all(default_true, default_false, input_args):
    print(input_args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-item", action=StoreTrueOverridable)
    parser.add_argument("--false-item", action=StoreFalseOverridable)

    defaults = {}
    if default_true is not None:
        defaults["true_item"] = default_true
    if default_false is not None:
        defaults["false_item"] = default_false

    parser.set_defaults(**defaults)

    result = vars(parser.parse_args(input_args))
    print(json.dumps(result, indent=2))

    def check(key, expected_if_provided):
        idx = input_args.index(key)
        if idx + 1 < len(input_args) and not input_args[idx + 1].startswith("--"):
            expected = EXPECTED[input_args[idx + 1].lower()]
        else:
            expected = expected_if_provided

        res_key = key[2:].replace("-", "_")

        assert result[res_key] == expected

    exp_default_true = False if default_true is None else default_true
    exp_default_false = True if default_false is None else default_false

    if len(input_args) == 0:
        assert result["true_item"] == exp_default_true
        assert result["false_item"] == exp_default_false

    if "--true-item" in input_args:
        check("--true-item", expected_if_provided=True)
    if "--false-item" in input_args:
        check("--false-item", expected_if_provided=False)

    print("OK")
