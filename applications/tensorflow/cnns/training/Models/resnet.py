# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

import tensorflow as tf
from . import resnet_base as rb
from .model_base import ModelBase
from functools import partial


class ResNet(rb.ResNetBase):
    def __init__(self, opts, is_training=True):
        if opts['dataset'] == 'imagenet':
            definitions = {**rb.RESNETS_Imagenet, **rb.RESNETS_Bottleneck_Imagenet}
        else:
            definitions = {**rb.RESNETS_Cifar, **rb.RESNETS_Bottleneck_Cifar}
        definition = definitions[opts["model_size"]]
        super().__init__(opts, definition, is_training)


def Model(opts, training, image):
    return ResNet(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = ResNet(opts, True)
    if splits is None or (
            len(splits) != opts['shards'] - 1 and opts['shards'] > 1):
        possible_splits = [
            s.keywords['name'] for s in x._build_function_list()
            if 'relu' in s.keywords['name']
        ]
        raise ValueError(
            "--pipeline-splits not specified or wrong number of splits. Need {} of {}".format(
                opts['shards'] - 1, possible_splits))
    splits.append(None)
    stages = [partial(x.first_stage, first_split_name=splits[0])]
    for i in range(len(splits) - 1):
        stages.append(
            partial(x.later_stage,
                    prev_split_name=splits[i],
                    end_split_name=splits[i + 1]))
    return stages


def add_arguments(parser):
    group = parser.add_argument_group('ResNet')
    rb.add_resnet_base_arguments(group)
    return parser


def set_defaults(opts):
    opts['summary_str'] += "ResNet-{model_size}\n"
    if opts["dataset"] == "imagenet":
        opts["shortcut_type"] = "B"
    elif "cifar" in opts["dataset"]:
        opts["shortcut_type"] = "A"

    # set ImageNet specific defaults
    if opts["dataset"] == "imagenet":
        if opts.get("weight_decay") is None:
            # value taken from tf_official_resnet - may not be appropriate for
            # small batch sizes
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-4
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-4
        if not opts.get("base_learning_rate_exponent"):
            if opts["optimiser"] == "SGD":
                opts["base_learning_rate_exponent"] = -8
            elif opts["optimiser"] == "momentum":
                opts["base_learning_rate_exponent"] = -11
        if not opts.get("epochs") and not opts.get("iterations"):
            opts["epochs"] = 100
        if not opts.get("learning_rate_schedule"):
            opts["learning_rate_schedule"] = [0.3, 0.6, 0.8, 0.9]
        if not opts.get("learning_rate_decay"):
            opts["learning_rate_decay"] = [1.0, 0.1, 0.01, 0.001, 1e-4]
        if not (
            opts.get("group_norm") is True or opts.get("batch_norm") is True
        ):
            # set group norm as default for ImageNet
            opts["group_norm"] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts["groups"] = 32
        if not opts.get("model_size"):
            opts["model_size"] = 18
        if not opts.get("batch_size"):
            opts["batch_size"] = 4
        if opts.get("warmup") is None:
            # warmup on by default for ImageNet
            opts["warmup"] = True

        # exclude beta and gamma from weight decay calculation
        opts["wd_exclude"] = ["beta", "gamma"]

    # set CIFAR specific defaults
    elif "cifar" in opts["dataset"]:
        if opts.get("weight_decay") is None:
            # based on sweep with CIFAR-10
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-6
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-6
        if not opts.get("base_learning_rate_exponent"):
            opts["base_learning_rate_exponent"] = -6
        if not opts.get("epochs") and not opts.get("iterations"):
            opts["epochs"] = 160
        if not opts.get("learning_rate_schedule"):
            opts["learning_rate_schedule"] = [0.5, 0.75]
        if not opts.get("learning_rate_decay"):
            opts["learning_rate_decay"] = [1.0, 0.1, 0.01]
        if not (
            opts.get("group_norm") is True or opts.get("batch_norm") is True
        ):
            # set batch norm as default for CIFAR
            opts["batch_norm"] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts["groups"] = 16
        if not opts.get("model_size"):
            opts["model_size"] = 20
        if not opts.get("batch_size"):
            opts["batch_size"] = 32

    if not opts["BN_decay"]:
        opts["BN_decay"] = 0.97

    opts["name"] = "RN{}".format(opts["model_size"])

    opts["name"] += "_bs{}".format(opts["batch_size"])
    if opts.get("replicas") > 1:
        opts["name"] += "x{}r".format(opts["replicas"])
    if opts["pipeline"]:
        opts["name"] += "x{}p".format(opts["gradient_accumulation_count"])
    elif opts.get("gradient_accumulation_count") > 1:
        opts["name"] += "x{}a".format(opts["gradient_accumulation_count"])

    if not (opts["batch_norm"] or opts["group_norm"]):
        opts["name"] += "_noBN"
        opts["summary_str"] += " No Batch Norm\n"
    elif opts["group_norm"]:
        opts["name"] += "_GN{}".format(opts["groups"])
        opts["summary_str"] += " Group Norm\n" "  {groups} groups\n"
    else:
        opts["name"] += "_BN"
        opts["summary_str"] += " Batch Norm\n"
        if (
            opts["BN_decay"] and opts["BN_decay"] != 0.97
        ):  # defined and not default
            opts["summary_str"] += "  Decay: {}\n".format(opts["BN_decay"])

    opts["name"] += "_{}{}".format(
        opts["precision"], "_noSR" if opts["no_stochastic_rounding"] else ""
    )
