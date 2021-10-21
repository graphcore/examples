# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import os
import torch
import popdist
import popdist.poptorch
import horovod.torch as hvd

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def cycle(iterator):
    """
    Loop `iterator` forever
    """
    while True:
        for item in iterator:
            yield item


def logger(msg):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(msg)


def sync_metrics(outputs, factor=1, average=True):
    if popdist.isPopdistEnvSet():
        if isinstance(outputs, float):
            return float(hvd.allreduce(torch.Tensor([outputs]), average=average).item())
        else:
            return [hvd.allreduce(output.div(factor), average=average).mean().item() for output in outputs]
    else:
        if isinstance(outputs, float):
            return outputs
        else:
            return [output.div(factor).mean().item() for output in outputs]


def get_sdk_version():
    sdk_path = os.environ.get("POPLAR_SDK_ENABLED", None)
    if sdk_path:
        return os.path.split(os.path.split(sdk_path)[0])[1]
    return "Unknown"
