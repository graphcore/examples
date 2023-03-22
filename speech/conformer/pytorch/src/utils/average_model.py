# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2021 Mobvoi Inc. All Rights Reserved.
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

import torch


def average_epoch(average_num, epoch_list, src_path):
    avg = None
    assert average_num == len(epoch_list)
    for path in epoch_list:
        states = torch.load(path, map_location=torch.device("cpu"))["model_weight"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k], average_num)
    torch.save({"model_weight": avg}, src_path + "/average_" + str(average_num) + "/average.pt")
