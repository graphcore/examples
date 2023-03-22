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
from poptorch.optim import SGD, Adam, LAMB
from transformers import get_constant_schedule, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

data_types = {"fp16": float16, "fp32": float32}


def get_lr_scheduler(optimizer, scheduler_type, warmup_steps=None, num_steps=None, last_epoch=-1):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=last_epoch)
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
        {"params": non_regularized_params, "weight_decay": 0},
    ]
    if config.optimizer == "LAMB":
        params[0]["max_weight_norm"] = config.max_norm
        params[1]["max_weight_norm"] = config.max_norm_bias

    if config.optimizer == "SGD":
        optimizer = SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            loss_scaling=config.loss_scaling,
            accum_type=data_types[config.accum_type],
            velocity_accum_type=data_types[config.first_order_type],
            use_combined_accum=config.use_combined_accum,
        )
    elif config.optimizer == "Adam":
        optimizer = Adam(
            params,
            lr=config.learning_rate,
            betas=None if config.adam_betas is None else (config.adam_betas[0], config.adam_betas[1]),
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
            loss_scaling=config.loss_scaling,
            accum_type=data_types[config.accum_type],
            first_order_momentum_accum_type=data_types[config.first_order_type],
            second_order_momentum_accum_type=data_types[config.second_order_type],
        )
    elif config.optimizer == "LAMB":
        optimizer = LAMB(
            params,
            lr=config.learning_rate,
            betas=None if config.adam_betas is None else (config.adam_betas[0], config.adam_betas[1]),
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
            loss_scaling=config.loss_scaling,
            max_weight_norm=None,
            accum_type=data_types[config.accum_type],
            first_order_momentum_accum_type=data_types[config.first_order_type],
            second_order_momentum_accum_type=data_types[config.second_order_type],
            bias_correction=config.bias_correction,
        )
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)
    return optimizer
