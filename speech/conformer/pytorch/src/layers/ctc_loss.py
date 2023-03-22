# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import poptorch


class CtcLoss(torch.nn.Module):
    def forward(self, log_probs, targets, input_lengths, target_lengths, dtype):
        dummy_shape = torch.zeros_like(
            torch.tensor(58.0, dtype=dtype), dtype=dtype, requires_grad=True
        )  # the dummy_shape's value is not import, which is used to put the shape and dtype to the custom_op
        loss = poptorch.custom_op(
            [log_probs, targets, input_lengths, target_lengths - 1],
            "Ctc",
            "ai.graphcore",
            1,
            example_outputs=[dummy_shape, log_probs],
            attributes={
                "enableReducedClassesInLabel": self.training,
                "reduction": "Sum",
                "blank": 0,
                "outDataType": "UNDIFINED",
            },
        )[0]
        loss = loss / torch.tensor(log_probs.shape[1], dtype=dtype)
        return loss
