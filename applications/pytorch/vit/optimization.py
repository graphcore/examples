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

from torch import float16, float32
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_constant_schedule
from poptorch.optim import SGD


def get_lr_scheduler(optimizer,
                     scheduler_type,
                     warmup_steps=None,
                     num_steps=None,
                     last_epoch=-1):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps,
                                                    num_steps, last_epoch=last_epoch)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)
    return scheduler


def get_optimizer(config, model):
    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": config.weight_decay},
        {"params": non_regularized_params, "weight_decay": 0}
    ]

    first_order_type = float16 if config.enable_half_first_order_momentum else float32

    if config.optimizer == "SGD":
        optimizer = SGD(params,
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay,
                        loss_scaling=config.loss_scaling,
                        accum_type=float16,
                        velocity_accum_type=first_order_type,
                        use_combined_accum=False)
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)
    return optimizer
