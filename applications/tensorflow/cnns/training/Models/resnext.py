# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
from .resnet_base import *


class ResNeXt(ResNetBase):
    def __init__(self, opts, is_training=True):
        if opts['dataset'] == 'imagenet':
            definitions = RESNETS_Bottleneck_Imagenet
        else:
            if opts['widenet']:
                definitions = RESNETS_Bottleneck_Cifar_wide
            else:
                definitions = RESNETS_Bottleneck_Cifar
        definition = definitions[opts["model_size"]]
        super().__init__(opts, definition, conv, is_training)
        self.conv_groups = opts['convolution_groups']
        self.group_dim = opts['group_dim']
        mult = self.group_dim * self.conv_groups / self.initial_block_filters
        self.block_fn = partial(definition.block_fn,
                                shortcut_type=opts["shortcut_type"],
                                conv_groups=self.conv_groups,
                                filter_factor=mult,
                                conv=self.conv,
                                norm=self.norm)


def Model(opts, training, image):
    return ResNeXt(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = ResNeXt(opts, True)
    if splits is None:
        possible_splits = [
            s.keywords['name'] for s in x._build_function_list()
            if 'relu' in s.keywords['name']
        ]
        raise ValueError(
            "--pipeline-splits not specified. Need {} of {}".format(
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
    add_resnet_arguments(group)
    group.add_argument('--convolution-groups', type=int,
                       help="Cardinality/Number of groups in convolution")
    group.add_argument('--group-dim', type=int,
                       help="Dimension of groups in convolution")
    return parser


def set_defaults(opts):
    if not opts['convolution_groups']:
        opts['convolution_groups'] = 32
    if not opts['group_dim']:
        opts['group_dim'] = 4
    sum = "ResNeXt-{model_size}_{convolution_groups}-{group_dim}d\n"
    opts['summary_str'] += sum
    set_resnet_defaults(opts, "RNX")
