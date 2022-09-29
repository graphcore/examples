# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2017-present, Facebook, Inc. All rights reserved.
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


import numpy as np
import poptorch
import torch

# Code below derived from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py


def sample_mixup_coefficients(config, random_generator):
    # need to generate num weight updates lambda coefficients
    coefficients = np.repeat(
        # 1 lambda per weight update
        random_generator.beta(config.alpha, config.alpha,
                              size=config.device_iterations),
        # repeat each lambda so there is one per micro batch
        int(config.samples_per_step / (config.device_iterations * config.micro_batch_size)))
    coefficients = coefficients.astype(np.float32, copy=False)
    coefficients = torch.from_numpy(coefficients.astype(
        np.float16 if config.precision[:3] == "16." else np.float))
    return coefficients
