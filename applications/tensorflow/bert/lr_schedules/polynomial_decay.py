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


class LearningRate:
    def __init__(self, base_learning_rate, warmup_steps, total_steps, power):
        """Polynomial decay learning rate.
        Args:
            base_learning_rate: base learning rate value, this is the value reached at the peak of the learning rate schedule.
            warmup_steps: The number of warmup steps we want to use for the ramp up of the learning rate from 0 to the base_learning_rate.
            total_steps: total number of steps to run.
        """
        self.max_learning_rate = base_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        if isinstance(power, int) or isinstance(power, float):
            self.power = [power]
        else:
            self.power = power

    def get_at_step(self, step):
        if step <= self.warmup_steps:
            lr = (step ** self.power[0]) * self.max_learning_rate / (self.warmup_steps ** self.power[0])
        else:
            if self.power[0] == 1 and len(self.power) == 1:
                lr = self.max_learning_rate * ((self.total_steps - step - 1) / (self.total_steps - self.warmup_steps))
            else:
                end_learning_rate = 1e-7
                decay_steps = self.total_steps - self.warmup_steps
                lr = ((self.max_learning_rate - end_learning_rate) * (
                    (1 - (step - self.warmup_steps) / decay_steps) ** (self.power[-1]))) + end_learning_rate
        # In order to avoid the learning rate to be exactly 0 we put a minimal learning rate of 1e-7
        lr = max(1e-7, lr)
        return lr
