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


expected_losses = {'sgd': 0.1,
                   'momentum': 0.0005,
                   'adamw': 0.002,
                   'lamb': 0.001}

expected_grads = {'sgd': [[-0.0666,     0.01865, - 0.11523],
                          [-0.08264,    0.02315, - 0.143],
                          [0.0421, - 0.011795,   0.0729],
                          [-0.000555,   0.0001554, - 0.0009604]],
                  'momentum': [[-0.00119,   0.000475, - 0.002478],
                               [-0.00119,   0.000475, - 0.002478],
                               [0., - 0.,       0.],
                               [-0.,        0., - 0.]],
                  'adamw': [[-0.001462,   0.0002267, - 0.001893],
                            [-0.002926,   0.000454, - 0.00379],
                            [0.00585, - 0.000908,   0.00758],
                            [-0.,         0., - 0.]],
                  'lamb': [[-0.,  0., - 0.],
                           [-0.,  0., - 0.],
                           [0., - 0.,  0.],
                           [-0.,  0., - 0.0]]}

expetced_weights = {'sgd': [[1.141],
                            [-0.319],
                            [1.972]],
                    'momentum': [[2.438],
                                 [-0.9727],
                                 [5.074]],
                    'adamw': [[2.994],
                              [-0.4644],
                              [3.877]],
                    'lamb': [[5.664], [-0.835], [7.59]]}
