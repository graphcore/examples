# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

import os

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestResnet18Inference(SubProcessChecker):
    """High-level integration tests for ResNet-18 inference"""

    def setUp(self):
        # download files only if not downloaded already
        images_path = os.path.join(working_path, "images")
        weights_path = os.path.join(working_path, "weights")
        if not os.path.exists(images_path) or not os.path.exists(weights_path):
            self.run_command("./get_images_and_weights.sh",
                             working_path,
                             ["Fetching images224.tar.gz",
                              "Unpacking ResNet18.tar.gz"],
                             env=None,
                             timeout=5*60)

    def test_help(self):
        self.run_command("python3 classify_images.py -h",
                         working_path,
                         "usage: classify_images.py")

    def test_single_image(self):
        self.run_command("python3 classify_images.py images/zebra.jpg",
                         working_path,
                         "Class 340: zebra")

    def test_directory_of_images(self):
        self.run_command("python3 classify_images.py images/",
                         working_path,
                         ["Class 409: analog clock",
                          "Average images per second:",
                          "Filename : gondola.jpg"])

    def test_ipu_model(self):
        test_env = os.environ.copy()
        test_env["TF_POPLAR_FLAGS"] = "--use_ipu_model"
        self.run_command('python3 classify_images.py images/pelican.jpg',
                         working_path,
                         ["Class 144: pelican"],
                         env=test_env)
