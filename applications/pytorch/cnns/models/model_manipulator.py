# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import OrderedDict
import torch
import torchvision
import poptorch
from torchvision.models.utils import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet, get_model_params
from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.mobilenet import model_urls as mobilenet_urls
import sys
sys.path.append('..')
import datasets


model_urls = dict(resnet_urls.items() | mobilenet_urls.items())
convert_model_names = {"resnext50": "resnext50_32x4d",
                       "resnext101": "resnext101_32x8d",
                       "mobilenet": "mobilenet_v2"}


def create_efficientnet(model_name, pretrained=True, num_classes=1000, norm_layer=torch.nn.BatchNorm2d, expand_ratio=6, group_dim=1):
    """ Creates EfficientNet instance with the predefined parameters.
    Parameters:
    model_name: Name of the model.
    pretrained: if true the network is initialized with pretrained weights.
    norm_layer: The used normalization layer in the network. eg. torch.nn.Identity means no initialization.
    expand_ratio: The used expand ratio in the blocks. Official EfficientNet uses 6
    group_dim: Dimensionality of the depthwise convolution. Official EfficientNet uses 1.

    Returns:
    The initialized EfficientNet model.
    """
    EfficientNet._check_model_name_is_valid(model_name)
    blocks_args, global_params = get_model_params(model_name, {"num_classes": num_classes})
    # Change expand ratio
    for idx in range(1, len(blocks_args)):
            blocks_args[idx] = blocks_args[idx]._replace(expand_ratio = expand_ratio)
    model = EfficientNet(blocks_args, global_params)
    model.set_swish(memory_efficient=False)
    if group_dim > 1:
        replace_en_depthwise_conv(model, group_dim)
    if not isinstance(norm_layer, torch.nn.BatchNorm2d):
        replace_bn(model, norm_layer)
    init_efficientnet(model)
    if pretrained:
        pretrained_model = EfficientNet.from_name(model_name)
        load_modified_model_from_state(model, pretrained_model.state_dict())
    return model


def init_efficientnet(model):
    """
    The method optimize the EfficientNet initialization.
    """
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        if isinstance(node, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(node.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(node, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(node.weight)
        for name, child in node.named_children():
            stack.append(child)


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


def replace_en_depthwise_conv(model, group_dim=1):
    """
    Modify the depthwise convolutions in EfficientNet to have the given group dimensionality.
    """
    for block in model._blocks:
        groups = max(block._depthwise_conv.in_channels // group_dim, 1)
        custom_conv = type(block._depthwise_conv)
        new_conv_layer = custom_conv(in_channels=block._depthwise_conv.in_channels,
                                     out_channels=block._depthwise_conv.out_channels,
                                     groups=groups,
                                     kernel_size=block._depthwise_conv.kernel_size,
                                     stride=block._depthwise_conv.stride,
                                     bias=False,
                                     image_size=224)  # Use fake image size as it'll have no effect.
        new_conv_layer.static_padding = block._depthwise_conv.static_padding
        replace_layer(block, '_depthwise_conv', new_conv_layer)


def replace_bn(model, norm_layer):
    """Replaces the normalization layers to the given normalization layer.
    Parameters:
    model: The model.
    norm_layer: The inserted torch.nn.Module instance.
    """
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            stack.append(child)
            if isinstance(child, torch.nn.BatchNorm2d):
                new_layer = norm_layer(child.num_features)
                replace_layer(node, name, new_layer)


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


def full_precision_norm(model, norm_layer):
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            stack.append(child)
            if isinstance(child, norm_layer):
                child.float()
                replace_layer(node, name, torch.nn.Sequential(datasets.ToFloat(), child))


def recompute_model(model, recompute_checkpoints):
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            if ("conv" in recompute_checkpoints and isinstance(child, torch.nn.Conv2d)) or \
               ("norm" in recompute_checkpoints and (isinstance(child, torch.nn.GroupNorm) or isinstance(child, torch.nn.BatchNorm2d))):
                new_layer = RecomputationCheckpoint(child)
                replace_layer(node, name, new_layer)
            stack.append(child)


class RecomputationCheckpoint(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        y = self.layer(x)
        return poptorch.recomputationCheckpoint(y)
