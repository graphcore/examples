# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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

logger = logging.getLogger("custom_loss_scaling")


class LossScalingScheduler:
    def __init__(self, base_loss_scaling, loss_scaling_by_step=None):
        """Custom loss scale scheduler.
        """

        self.base_loss_scaling = base_loss_scaling
        self._cur_loss_scaling = self.base_loss_scaling

        if loss_scaling_by_step:
            logger.info("Using variable loss scaling.")
            # Convert json strings to integers
            self.loss_scaling_by_step = {
                int(k): v for k, v in loss_scaling_by_step.items()
            }
        else:
            self.loss_scaling_by_step = None
            logger.info("Using static loss scaling.")

    def get_at_step(self, step):
        if self.loss_scaling_by_step:
            diffs = {
                step - k: k for k in self.loss_scaling_by_step.keys() if k <= step
            }
            closest = diffs[min(diffs)]
            self._cur_loss_scaling = self.loss_scaling_by_step[closest]
            return self._cur_loss_scaling
        else:
            return self.base_loss_scaling
