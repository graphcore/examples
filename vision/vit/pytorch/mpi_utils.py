# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import horovod.torch as hvd
import torch


def mpi_reduce(output, average=True):
    """
    Perform MPI reduction across MPI instances
    """
    if type(output) in [int, float]:
        output = float(hvd.allreduce(
            torch.Tensor([output]), average=average).item())
    else:
        raise NotImplementedError(
            f'parameter output has type: {type(output)} which is not supported')

    return output
