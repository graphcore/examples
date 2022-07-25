# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2018 Philip Popien
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

import random
from randaugment import ImageNetPolicy
from randaugment.randaugment import SubPolicy


def _init(self):
    self.policies = [
        # probability1, operation1, magnitude1, probability2, operation2, magnitude2, fillcolor
        (0.4, "posterize", 8, 0.6, "rotate", 9, (128, 128, 128)),
        (0.6, "solarize", 5, 0.6, "autocontrast", 5, (128, 128, 128)),
        (0.8, "equalize", 8, 0.6, "equalize", 3, (128, 128, 128)),
        (0.6, "posterize", 7, 0.6, "posterize", 6, (128, 128, 128)),
        (0.4, "equalize", 7, 0.2, "solarize", 4, (128, 128, 128)),
        (0.4, "equalize", 4, 0.8, "rotate", 8, (128, 128, 128)),
        (0.6, "solarize", 3, 0.6, "equalize", 7, (128, 128, 128)),
        (0.8, "posterize", 5, 1.0, "equalize", 2, (128, 128, 128)),
        (0.2, "rotate", 3, 0.6, "solarize", 8, (128, 128, 128)),
        (0.6, "equalize", 8, 0.4, "posterize", 6, (128, 128, 128)),
        (0.8, "rotate", 8, 0.4, "color", 0, (128, 128, 128)),
        (0.4, "rotate", 9, 0.6, "equalize", 2, (128, 128, 128)),
        (0.0, "equalize", 7, 0.8, "equalize", 8, (128, 128, 128)),
        (0.6, "invert", 4, 1.0, "equalize", 8, (128, 128, 128)),
        (0.6, "color", 4, 1.0, "contrast", 8, (128, 128, 128)),
        (0.8, "rotate", 8, 1.0, "color", 2, (128, 128, 128)),
        (0.8, "color", 8, 0.8, "solarize", 7, (128, 128, 128)),
        (0.4, "sharpness", 7, 0.6, "invert", 8, (128, 128, 128)),
        (0.6, "shearX", 5, 1.0, "equalize", 9, (128, 128, 128)),
        (0.4, "color", 0, 0.6, "equalize", 3, (128, 128, 128)),
        (0.4, "equalize", 7, 0.2, "solarize", 4, (128, 128, 128)),
        (0.6, "solarize", 5, 0.6, "autocontrast", 5, (128, 128, 128)),
        (0.6, "invert", 4, 1.0, "equalize", 8, (128, 128, 128)),
        (0.6, "color", 4, 1.0, "contrast", 8, (128, 128, 128)),
        (0.8, "equalize", 8, 0.6, "equalize", 3, (128, 128, 128)),
    ]


def _call(self, img):
    policy_idx = random.randint(0, len(self.policies) - 1)
    policy = SubPolicy(*self.policies[policy_idx])
    return policy(img)


ImageNetPolicy.__init__ = _init
ImageNetPolicy.__call__ = _call
