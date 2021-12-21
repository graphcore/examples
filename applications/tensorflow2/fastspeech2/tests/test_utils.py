# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numpy as np


class TestFailureError(Exception):
    __test__ = False


def getTensorError(tA, pA):
    """Get the error between two tensors."""
    # pA, tA are corresponding tensors from two models
    pA_shape = np.shape(pA)
    tA_shape = np.shape(tA)
    assert (pA_shape == tA_shape), "Arrays must be same shape"
    tA = tA.astype(np.float32)
    pA = pA.astype(np.float32)
    ss_err = np.sum((np.array(pA) - np.array(tA))**2)
    ss_pA = np.sum(np.array(pA)**2)
    ss_tA = np.sum(np.array(tA)**2)
    return ss_err / (math.sqrt(ss_pA * ss_tA) + 1.0e-8)


def getTensorRelativError(tA, pA):
    """Get the relative error between two tensors."""
    pA_shape = np.shape(pA)
    tA_shape = np.shape(tA)
    assert (pA_shape == tA_shape), "Arrays must be same shape"

    err = np.mean(np.abs(np.array(pA)-np.array(tA)))
    return err


def reportTensorError(result):
    reportStr = " |pA - tA|^2 / (|pA||tA| + 1e-8)  = " + str(result) + "\n"
    return reportStr


def checkResult(result, margin):
    if np.isnan(result):
        raise TestFailureError(str(result) + " is NaN")
    elif (result > margin):
        print(reportTensorError(result))
        raise TestFailureError(str(result) + " is greater than " + str(margin))
    else:
        print(reportTensorError(result))


def check_tensor(A, B, margin=1.5e-8):
    """Check if the error between two tensors is bigger than setting margin or not."""
    result = getTensorError(A, B)
    print(f"Results: {result}")
    checkResult(result, margin)


def check_tensor_relative(A, B, margin=1.5e-8):
    """Check if the relative error between two tensors is bigger than setting margin or not."""
    result = getTensorRelativError(A, B)
    checkResult(result, margin)
