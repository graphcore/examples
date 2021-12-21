# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import re
import logging
import torch
import torchvision
import poptorch
import timm
from functools import partial
from collections import OrderedDict
from torch.fx import symbolic_trace
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.mobilenetv2 import model_urls as mobilenet_urls
import import_helper
from .mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small
import models


model_urls = dict(resnet_urls.items() | mobilenet_urls.items())
convert_model_names = {
    "resnext50": "resnext50_32x4d",
    "resnext101": "resnext101_32x8d",
    "mobilenet": "mobilenet_v2",
}


def create_mobilenetv3(model_name, pretrained=True, num_classes=1000, norm_layer=torch.nn.BatchNorm2d):
    """ Creates MobilenetV3 instance with the predefined parameters.
    Parameters:
    model_name: Name of the model, 'small' or 'large'.
    pretrained: if true the network is initialized with pretrained weights.
    norm_layer: The used normalization layer in the network. eg. torch.nn.Identity means no initialization.

    Returns:
    The initialized MobilenetV3 model.
    """
    if pretrained:
        logging.info("Pretrained MobileNet v3 not available. Random initialization is used.")
    if model_name == 'small':
        model = MobileNetV3_Small(num_classes=num_classes, norm_layer=norm_layer)
    else:
        model = MobileNetV3_Large(num_classes=num_classes, norm_layer=norm_layer)
    return model


def create_efficientnet(model_name, pretrained=False, num_classes=1000,
                        norm_layer=torch.nn.BatchNorm2d, expand_ratio=6,
                        group_dim=1):
    """Creates an initialized EfficientNet model.
    Parameters:
        model_name: variant of the model.
        pretrained: if true the network is initialized with pretrained weights.
        norm_layer: a function that creates normalization layers instances.
        expand_ratio: expansion ratio, official EfficientNet uses 6.
        group_dim: dimensionality of the group convolution, official EfficientNet
            uses 1 (i.e. depthwise convolutions)
    """
    if pretrained and _creating_modified_en_model(norm_layer, expand_ratio, group_dim):
        raise ValueError(
            "Pretrained EfficientNet model is not supported if "
            "the model is modified, please use batchnorm and other "
            "default parameters with pretrained=True.")

    model_variant = model_name.replace('-', '_')
    if pretrained:
        # Use weights ported from the original TF model.
        model_variant = "tf_" + model_variant
    model_config = models.available_models[model_name]

    architecture_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e{:d}_c24_se0.25'.format(expand_ratio)],
        ['ir_r2_k5_s2_e{:d}_c40_se0.25'.format(expand_ratio)],
        ['ir_r3_k3_s2_e{:d}_c80_se0.25'.format(expand_ratio)],
        ['ir_r3_k5_s1_e{:d}_c112_se0.25'.format(expand_ratio)],
        ['ir_r4_k5_s2_e{:d}_c192_se0.25'.format(expand_ratio)],
        ['ir_r1_k3_s1_e{:d}_c320_se0.25'.format(expand_ratio)],
    ]

    round_channels_fn = partial(
        timm.models.efficientnet.round_channels,
        multiplier=model_config['channel_multiplier'],
    )

    model = timm.models.helpers.build_model_with_cfg(
        model_cls=timm.models.EfficientNet,
        variant=model_variant,
        default_cfg=timm.models.efficientnet.default_cfgs[model_variant],
        pretrained=pretrained,
        block_args=timm.models.efficientnet.decode_arch_def(
            architecture_def,
            model_config['depth_multiplier']
        ),
        num_features=round_channels_fn(1280),
        stem_size=32,
        round_chs_fn=round_channels_fn,
        act_layer=timm.models.efficientnet.resolve_act_layer({'act_layer': 'swish'}),
        norm_layer=norm_layer,
        num_classes=num_classes,
        drop_rate=model_config['dropout_rate'],
        drop_path_rate=0.2,
    )

    if group_dim > 1:
        model = _depthwise_conv_to_group_conv(model, group_dim)
    model = _optimize_en_se_layers(model)
    return model


def _creating_modified_en_model(norm_layer, expand_ratio, group_dim):
    uses_batch_norm = False
    if isinstance(norm_layer, partial):
        uses_batch_norm = norm_layer.func == torch.nn.BatchNorm2d
    else:
        uses_batch_norm = norm_layer == torch.nn.BatchNorm2d
    return not (uses_batch_norm and expand_ratio == 6 and group_dim == 1)


def _depthwise_conv_to_group_conv(model, group_dim):
    for block in model.blocks:
        for layer in block:
            group_conv = timm.models.layers.create_conv2d(
                in_channels=layer.conv_dw.in_channels,
                out_channels=layer.conv_dw.out_channels,
                kernel_size=layer.conv_dw.kernel_size[0],
                stride=layer.conv_dw.stride[0],
                dilation=layer.conv_dw.dilation[0],
                groups=max(layer.conv_dw.in_channels // group_dim, 1),
            )
            replace_layer(layer, 'conv_dw', group_conv)
    return model


def _optimize_en_se_layers(model):
    for block in model.blocks:
        for layer in block:
            se_ipu = models.SqueezeExciteIPU(layer.se)
            replace_layer(layer, 'se', se_ipu)
    return model


def residual_normlayer_init(model):
    """
    The method initialize the norm layer's weight part to be zero before the residual connection.
    It mimics to the networks to be shallower.  It helps to converge in the early part of the training.
    Only works on ResNet and ResNext networks.
    """
    for layer_id in range(1, 5):
        layer = getattr(model, "layer" + str(layer_id))
        for block in layer:
            if hasattr(block, 'downsample') and block.downsample is not None:
                norm_layer = block.downsample[-1]
            else:
                if isinstance(block, torchvision.models.resnet.BasicBlock):
                    norm_layer = block.bn2
                elif isinstance(block, torchvision.models.resnet.Bottleneck):
                    norm_layer = block.bn3
            if hasattr(norm_layer, "weight"):
                torch.nn.init.zeros_(norm_layer.weight)


def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)


def get_module_and_parent_by_name(node, split_tokens):
    child_to_find = split_tokens[0]
    for name, child in node.named_children():
        if name == child_to_find:
            if len(split_tokens) == 1:
                return node, child, name
            else:
                return get_module_and_parent_by_name(child, split_tokens[1:])

    return None, None, None


def load_modified_model(model, model_name):
    if model_name in convert_model_names.keys():
        model_name = convert_model_names[model_name]

    model_url = model_urls[model_name]
    trained_state_dict = load_state_dict_from_url(model_url, progress=True)
    return load_modified_model_from_state(model, trained_state_dict)


def load_modified_model_from_state(model, pretrained_state_dict):
    default_state_dict = model.state_dict()

    def get_weight(layer):
        if layer in pretrained_state_dict.keys() and pretrained_state_dict[layer].size() == default_state_dict[layer].size():
            return pretrained_state_dict[layer]
        else:
            return default_state_dict[layer]
    corrected_state_dict = OrderedDict({layer: get_weight(layer) for layer in default_state_dict.keys()})
    model.load_state_dict(corrected_state_dict)
    return model


def recompute_model(model, recompute_checkpoints):
    # Put recomutation checkpoint if regular expression matches
    traced_model = symbolic_trace(model)
    for node in traced_model.graph.nodes:
        name = str(node).replace('_', '/')
        recompute_checkpoint = False
        for checkpoint_re in recompute_checkpoints:
            if re.fullmatch(checkpoint_re, name):
                logging.info(f"RECOMPUTE CHECKPOINT:{name}")
                recompute_checkpoint = True
                with traced_model.graph.inserting_after(node):
                    new_node = traced_model.graph.call_function(
                        poptorch.recomputationCheckpoint, args=(node,))
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node,)
                break
        if not recompute_checkpoint:
            logging.info(f"RECOMPUTE:{name}")

    traced_model.recompile()
    return traced_model


def pad_first_conv(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parent, node, field_or_idx_str = get_module_and_parent_by_name(model, name.split('/'))
            replace_layer(parent, field_or_idx_str, PaddedConv(module))
            return model
    return model


class PaddedConv(torch.nn.Conv2d):
    """
    This layer can be applied as the first Conv2d layer.
    Expects 3 input channel and converts it for 4 channel for the Conv2d layer.
    The inputs forth channel is padded with zeros.
    """
    def __init__(self, layer):
        super().__init__(4, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups, bias=hasattr(layer, "bias"), padding_mode=layer.padding_mode)
        self.extract_weights(layer)


    def forward(self, input):
        pad_shape = list(input.size())
        pad_shape[1] = 1
        padding = torch.zeros(pad_shape, dtype=input.dtype)
        padded_input = torch.cat((input, padding), 1)
        return super().forward(padded_input)

    def extract_weights(self, conv_layer):
        self.weight.data[:, :3, :, :] = conv_layer.weight.data
        if hasattr(conv_layer, "bias"):
            self.bias = conv_layer.bias
