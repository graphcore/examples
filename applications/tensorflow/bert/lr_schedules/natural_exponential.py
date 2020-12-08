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


from math import exp, floor


class LearningRate:
    def __init__(self, base_learning_rate, warmup_steps, decay_steps, decay_rate, total_steps):
        """Exponential decay learning rate.
        Args:
            base_learning_rate: base learning rate, this is the value reached at the peak of the learning rate schedule.
            warmup_steps: warm-up period in steps.
            decay_steps: attenuation period in steps.
            decay_rate: rate of decay.
        """
        self.base_lr = base_learning_rate
        self.decay_step = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps


    def feed_dict_lr(self, step):
        if step < self.warmup_steps:
            lr = self.base_lr*(step/self.warmup_steps)
        else:
            lr = self.base_lr * exp(- self.decay_rate * floor((step - self.warmup_steps) / self.decay_step))
        return lr
