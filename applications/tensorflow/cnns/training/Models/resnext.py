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

"""
ResNeXt

Aggregated Residual Transformations for Deep Neural Networks
https://arxiv.org/pdf/1611.05431.pdf

In addition to the many parameters it shares with ResNet, you can also change the
cardinality with --convolution-groups and change the dimensions with --group-dim. A
--wide-net option can be used when training with CIFAR-sized data.

"""


import tensorflow as tf
from functools import partial
from . import resnet_base as rb
from .model_base import ModelBase


class ResNeXt(rb.ResNetBase):
    def __init__(self, opts, is_training=True):
        if opts['dataset'] == 'imagenet':
            definitions = rb.RESNETS_Bottleneck_Imagenet
        else:
            if opts['widenet']:
                definitions = rb.RESNETS_Bottleneck_Cifar_wide
            else:
                definitions = rb.RESNETS_Bottleneck_Cifar
        definition = definitions[opts["model_size"]]
        super().__init__(opts, definition, is_training)

        self.conv_groups = opts['convolution_groups']
        self.group_dim = opts['group_dim']
        mult = self.group_dim * self.conv_groups / self.initial_block_filters
        self.block_fn = partial(definition.block_fn,
                                shortcut_type=opts["shortcut_type"],
                                conv_groups=self.conv_groups,
                                filter_factor=mult,
                                conv=self.conv,
                                norm=self.norm_dict)

        # Apply dataset specific changes
        if opts["dataset"] == "imagenet":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=7, initial_downsample=True)
        elif opts["dataset"] == "cifar-10":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=3, initial_downsample=False)
        elif opts["dataset"] == "cifar-100":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=3, initial_downsample=False)


def Model(opts, training, image):
    return ResNeXt(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = ResNeXt(opts, True)
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
    group = parser.add_argument_group('ResNeXt')
    rb.add_resnet_base_arguments(group)
    group.add_argument('--convolution-groups', type=int,
                       help="Cardinality/Number of groups in convolution")
    group.add_argument('--group-dim', type=int,
                       help="Dimension of groups in convolution")
    group.add_argument("--widenet", action="store_true",
                       help="Use widenet with bottleneck on cifar.")
    return parser


def set_defaults(opts):
    if not opts['convolution_groups']:
        opts['convolution_groups'] = 32
    if not opts['group_dim']:
        opts['group_dim'] = 4
    sum = "ResNeXt-{model_size}_{convolution_groups}-{group_dim}d\n"
    opts['summary_str'] += sum
    if opts["dataset"] == "imagenet":
        opts["shortcut_type"] = "B"
    elif "cifar" in opts["dataset"]:
        opts["shortcut_type"] = "A"

    # set ImageNet specific defaults
    if opts["dataset"] == "imagenet":
        if opts.get("weight_decay") is None:
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
            opts["group_norm"] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts["groups"] = 32
        if not opts.get("model_size"):
            opts["model_size"] = 14
        if not opts.get("batch_size"):
            opts["batch_size"] = 1
        if opts.get("warmup") is None:
            opts["warmup"] = True
        # force stable norm on
        if not opts.get("stable_norm"):
            opts['stable_norm'] = True

        # exclude beta and gamma from weight decay calculation
        opts["wd_exclude"] = ["beta", "gamma"]

    # set CIFAR specific defaults
    elif "cifar" in opts["dataset"]:
        if opts.get("weight_decay") is None:
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
            opts["batch_norm"] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts["groups"] = 16
        if not opts.get("model_size"):
            opts["model_size"] = 11
        if not opts.get("batch_size"):
            opts["batch_size"] = 8

    if not opts["BN_decay"]:
        opts["BN_decay"] = 0.97

    opts["name"] = "RNX{}".format(opts["model_size"])

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
