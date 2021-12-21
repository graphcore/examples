# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
import pytest
import subprocess
import re


@pytest.mark.parametrize('optim', ['sgd',
                                   'momentum',
                                   'adamw',
                                   'lamb'])
def test_regression_optimiser(optim):

    cmd = f"python3 regression_optimiser.py --optimiser {optim}"
    build_dir = Path(__file__).parent
    completed = subprocess.run(args=cmd.split(),
                               cwd=build_dir,
                               shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               )
    try:
        completed.check_returncode()
        return_code_ok = True
    except subprocess.CalledProcessError:
        return_code_ok = False
    """
    Print the output with error messages if the success message is not found
    """
    output = str(completed.stdout, 'utf-8')
    must_contain = ["Losses, grads and weights match."]
    regexes = [re.compile(s) for s in must_contain]
    for i, r in enumerate(regexes):
        match = r.search(output)
        if not match:
            print(f"Output of command: {cmd} contained no match for: '+\
                  {must_contain[i]}'\nOutput was:\n{output}")

    assert return_code_ok is True
