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


class LayerScalingOptimizerTransform:
    '''Typically when finetuning the final layer has not been pretrained.
        It can be advantageous to have a larger the learning rate on this new layer
        when compared to the rest of the pretrained network.'''
    def __init__(self,
                 name: str,
                 scale: float):
        self.layer_name = name
        self.scale = scale

    def __call__(self,
                 tensor_id: str,
                 parameters: Mapping[str, float],
                 stage: int):
        transformed = parameters.copy()

        def get_param(name):
            default_name = "default" + name[:1].capitalize() + name[1:]
            return parameters.get(name, parameters.get(default_name, None))

        if self.layer_name in tensor_id:
            lr = get_param("learningRate")
            if lr is not None:
                transformed["learningRate"] = lr * self.scale

        return transformed
