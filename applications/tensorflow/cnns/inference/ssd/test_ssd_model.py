# Copyright (C) 2020 Graphcore Ltd.
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

import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_ssd_model(self):
        self.run_command("pip3 install -r requirements.txt",
                         working_path,
                         [])
        self.run_command("python ssd_model.py",
                         working_path,
                         ["Mean TFLOPs/S is"])
        self.run_command("python ssd_single_image.py",
                         working_path,
                         ["Done running inference."])
