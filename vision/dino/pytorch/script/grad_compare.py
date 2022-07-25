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
import sys
import torch
import numpy as np

cpu_grad = torch.load(sys.argv[1])
ipu_grad = torch.load(sys.argv[2])

rtols = {}
for key in cpu_grad.keys():
    if key in ipu_grad.keys():
        grad_cpu = cpu_grad[key]
        grad_ipu = ipu_grad[key]
        diff = grad_cpu - grad_ipu
        relative = diff.abs() / (grad_ipu.abs().mean())

        rtols[key] = relative

        diff_sum = torch.sum(torch.abs(diff))

        print(f'{key} {grad_ipu.shape} {relative.max()}')
        try:
            np.testing.assert_allclose(
                grad_cpu, grad_ipu, atol=1e-4, rtol=1e-6)
        except Exception as e:
            relative = e.args[0].split('\n')[5]
            abs_diff = e.args[0].split('\n')[4]
            print(f'{abs_diff}\t{relative}\n')
            raise e
