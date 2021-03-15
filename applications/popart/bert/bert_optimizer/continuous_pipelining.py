# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

from typing import Mapping


class ContinuousPipeliningOptimizerTransform:
    '''A transformation class that takes optimizer parameters for a standard network
        and returns them with scaling added for continuous updates pipelining.

        Due to there being updates between the forward and backward pass for a single micro batch
        the weights/activations/gradients will be inconsistent. A practical solution to this
        is to increase the Momentum parameters and decrease the Learning Rate on earlier pipeline
        stages.

        Earlier pipeline stages have a larger delay (more updates) between the forward
        and backward passes so the additional smoothing from higher momentum and lower
        learning rate should reduce the impact of this additional noise.'''

    def __init__(self,
                 total_stages: int,
                 lr_scaling: bool = True,
                 lr_offset: float = 0.25,
                 momentum_scaling: bool = True,
                 momentum_offset: float = 0.1):
        if total_stages < 1:
            raise RuntimeError("Must specify total_stages > 0")
        self._total_stages = total_stages

        self.lr_scaling = lr_scaling
        self.lr_offset = lr_offset
        self.lr_scale_factor = (1 - lr_offset) / (total_stages)

        self.momentum_scaling = momentum_scaling
        self.momentum_offset = momentum_offset
        self.momentum_scale_factor = (1 - momentum_offset) / (total_stages)

    def _lr_value(self, val: float, stage: int):
        '''Proprotional to stage'''
        return val * abs(self.lr_scale_factor * stage + self.lr_offset)

    def _momentum_value(self, val: float, stage: int):
        '''Inversly proprotional to stage'''
        if val != 0:
            return 1 - ((1 - val) * abs(self.momentum_scale_factor * stage + self.momentum_offset))
        return 0

    def __call__(self,
                 tensor_id: str,
                 parameters: Mapping[str, float],
                 stage: int):
        transformed = parameters.copy()

        if self._total_stages == 1:
            return transformed

        def get_param(name):
            default_name = "default" + name[:1].capitalize() + name[1:]
            return parameters.get(name, parameters.get(default_name, None))

        lr = get_param("learningRate")
        if self.lr_scaling and lr is not None:
            transformed["learningRate"] = self._lr_value(lr, stage)

        momentum = get_param("momentum")
        if self.momentum_scaling and momentum is not None:
            transformed["momentum"] = self._momentum_value(momentum, stage)

        dampening = get_param("dampening")
        if self.momentum_scaling and dampening is not None:
            transformed["dampening"] = self._momentum_value(dampening, stage)

        return transformed
