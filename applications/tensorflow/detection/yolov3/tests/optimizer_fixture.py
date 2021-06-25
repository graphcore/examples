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

import pytest


@pytest.fixture()
def expected_losses():
    return {"momentum": 0.0005,
            "adamw": 0.002,
            }


@pytest.fixture()
def expected_grads():
    return {"momentum": [[-0.00119,   0.000475, - 0.002478],
                         [-0.00119,   0.000475, - 0.002478],
                         [0., - 0.,       0.],
                         [-0.,        0., - 0.]],
            "adamw": [[-0.001462,   0.0002267, - 0.001893],
                      [-0.002926,   0.000454, - 0.00379],
                      [0.00585, - 0.000908,   0.00758],
                      [-0.,         0., - 0.]]
            }


@pytest.fixture()
def expetced_weights():
    return {"momentum": [[2.438],
                         [-0.9727],
                         [5.074]],
            "adamw": [[2.994],
                      [-0.4644],
                      [3.877]],
            }
