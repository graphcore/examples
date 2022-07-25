# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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


import torch


def _accuracy(pred, targ):
    """
    Compute accuracy score of predicted labels `pred` and target labels `targ`
    """
    return (pred == targ).float().mean()


def accuracy(pred, targ, targ_b=None, lam=None):
    """
    Support accuracy calculation when mixup is enabled.
    """
    if targ_b is None:
        return _accuracy(pred, targ)
    lam = lam[0]
    return lam * _accuracy(pred, targ) + (1 - lam) * _accuracy(pred, targ_b)
