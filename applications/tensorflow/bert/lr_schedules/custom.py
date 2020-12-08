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


import logging

logger = logging.getLogger("custom_lr")


class LearningRate:
    def __init__(self, base_learning_rate, lr_schedule_by_step):
        """Custom learning rate schedule.
        Args:
            base_learning_rate: base learning rate.
            lr_schedule_by_step: dictionary mapping step -> learning rate.
        """

        self.base_lr = base_learning_rate
        self.init_lr = self.base_lr
        self._cur_lr = self.init_lr
        self.lr_schedule_by_step = lr_schedule_by_step

    def feed_dict_lr(self, step):
        diffs = {
            step - int(k): int(k)
            for k in self.lr_schedule_by_step.keys()
            if int(k) <= step
        }
        closest = str(diffs[min(diffs)])
        self._cur_lr = self.lr_schedule_by_step[closest]
        return self._cur_lr

    def get_current_lr(self):
        return self._cur_lr
