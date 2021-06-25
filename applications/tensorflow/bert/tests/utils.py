# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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
from functools import reduce
from itertools import chain
from typing import Any, Callable, Iterable, Mapping, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn


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

    err = np.max(np.abs(np.array(pA)-np.array(tA)))
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


def make_tuple(something: Any) -> Tuple:
    if isinstance(something, tuple) or isinstance(something, list):
        def concat(accl: Iterable, s: Any) -> Iterable:
            return chain(accl, make_tuple(s))

        return tuple(reduce(concat, something, ()))
    return (something, )


def copy_torch_weights_to_tf(torch_model: nn.Module,
                             tf_model: object,
                             tf_to_torch: Mapping[str, str],
                             transform: Mapping[str, Callable[[np.ndarray], np.ndarray]],
                             sess: tf.Session):
    tensors = [v for v in tf.trainable_variables()]
    variable_name = [v.name for v in tf.trainable_variables()]
    torch_weight = {name: w.data.numpy()
                    for name, w in torch_model.named_parameters()}

    weights = []
    weights_tensor = []
    for name, tensor in zip(variable_name, tensors):
        if name in tf_to_torch.keys():
            torch_name = tf_to_torch[name]
            print(
                f"{name} ==> {torch_name}\n{torch_weight[torch_name]}")
            # tf.assign(tensor, torch_weight[torch_name])
            weights.append(tf.assign(tensor, torch_weight[torch_name]))
            weights_tensor.append(tensor)
    print(f"Weights = {weights}")
    return weights


def check_tensors(torch_outputs: Iterable[np.ndarray],
                  onnx_outputs: Iterable[np.ndarray],
                  margin: float = 1.5e-8):
    for t_torch, t_onnx in zip(torch_outputs, onnx_outputs):
        check_oom_failures(t_torch, t_onnx)
        check_tensor(t_onnx.reshape(t_torch.shape), t_torch, margin=margin)


def check_oom_failures(torch_output: np.ndarray, onnx_output: np.ndarray):
    failed_methods = []
    # Produce an error indicating which implementation ran out of memory during
    # compilation. Both could fail, so we won't print exclusively.
    if type(torch_output) == float and np.isnan(torch_output):
        failed_methods.append("Custom Operation")

    if type(onnx_output) == float and np.isnan(onnx_output):
        failed_methods.append("ONNX")

    if len(failed_methods) > 0:
        msg = "OOM in the following implementations: " + \
            ", ".join(failed_methods)

        raise TestFailureError(msg)


def run_fwd_model(inputs: Union[Iterable[np.ndarray], Mapping[str, np.ndarray]], torch_model: nn.Module) -> Iterable[np.ndarray]:
    def np_to_torch(np_arr: np.ndarray):
        if np_arr.dtype == np.uint32:
            np_arr = np_arr.astype(np.int32)

        torch_input = torch.from_numpy(np_arr)
        if np_arr.dtype == np.int32:
            torch_input = torch_input.long()
        elif np_arr.dtype == np.float:
            torch_input = torch_input.float()
        return torch_input
    if isinstance(inputs, dict):
        torch_inputs = {k: np_to_torch(v) for k, v in inputs.items()}
        torch_outputs = torch_model(**torch_inputs)
    else:
        torch_inputs = map(np_to_torch, inputs)
        torch_outputs = torch_model(*torch_inputs)

    return (t_torch.detach().numpy() for t_torch in make_tuple(torch_outputs))


def check_tf_torch_model(sess: tf.Session,
                         torch_model: nn.Module,
                         tf_to_torch: Mapping[str, str],
                         margin: float = 1.5e-8,
                         relative: bool = False):

    tensors = [v for v in tf.trainable_variables()]
    variable_name = [v.name for v in tf.trainable_variables()]
    torch_vars = {}
    for name, w in reversed(list(torch_model.named_parameters())):
        torch_vars[name] = w

    for name, w in zip(variable_name, tensors):
        if name in tf_to_torch.keys():
            torch_name = tf_to_torch[name]
            torch_w = torch_vars[torch_name]
            tf_w = sess.run(w)
            torch_w = torch_w.detach().numpy()
            try:
                if relative:
                    check_tensor_relative(tf_w, torch_w, margin)
                else:
                    check_tensor(tf_w, torch_w, margin)
            except TestFailureError as e:
                print("For weight: ", name)
                raise e
        else:
            raise TestFailureError('tf name not in torch:{}'.format(name))
