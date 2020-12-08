# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import OrderedDict
from functools import partial
import torch
import torchvision
import poptorch
import logging
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.mobilenet import model_urls as mobilenet_urls
from torchvision.models.utils import load_state_dict_from_url

available_models = {'resnet18': {"model": torchvision.models.resnet18, "input_shape": (3, 224, 224)},
                    'resnet34': {"model": torchvision.models.resnet34, "input_shape": (3, 224, 224)},
                    'resnet50': {"model": torchvision.models.resnet50, "input_shape": (3, 224, 224)},
                    'resnet101': {"model": torchvision.models.resnet101, "input_shape": (3, 224, 224)},
                    'resnet152': {"model": torchvision.models.resnet152, "input_shape": (3, 224, 224)},
                    'resnext50': {"model": torchvision.models.resnext50_32x4d, "input_shape": (3, 224, 224)},
                    'resnext101': {"model": torchvision.models.resnext101_32x8d, "input_shape": (3, 224, 224)},
                    'mobilenet': {"model": torchvision.models.mobilenet_v2, "input_shape": (3, 224, 224)},
                    'efficientnet-b0': {"model": EfficientNet, "input_shape": (3, 224, 224)},
                    'efficientnet-b1': {"model": EfficientNet, "input_shape": (3, 240, 240)},
                    'efficientnet-b2': {"model": EfficientNet, "input_shape": (3, 260, 260)},
                    'efficientnet-b3': {"model": EfficientNet, "input_shape": (3, 300, 300)},
                    'efficientnet-b4': {"model": EfficientNet, "input_shape": (3, 380, 380)},
                    'efficientnet-b5': {"model": EfficientNet, "input_shape": (3, 456, 456)},
                    'efficientnet-b6': {"model": EfficientNet, "input_shape": (3, 528, 528)},
                    'efficientnet-b7': {"model": EfficientNet, "input_shape": (3, 600, 600)}
                    }

model_urls = dict(resnet_urls.items() | mobilenet_urls.items())
convert_model_names = {"resnext50": "resnext50_32x4d",
                       "resnext101": "resnext101_32x8d",
                       "mobilenet": "mobilenet_v2"}


def get_model(opts, data_info, pretrained=True):
    """
    params:
    opts: contains the user defined command line parameters
    data info: the input and the output shape of the data
    pretrain: if it is true the weights are loaded from a publicly available pretrained model
    """
    norm_layer = get_norm_layer(opts)

    if opts.model in available_models:
        if 'efficientnet' in opts.model:
            if pretrained:
                model = available_models[opts.model]["model"].from_pretrained(opts.model)
            else:
                model = available_models[opts.model]["model"].from_name(opts.model, {"num_classes": data_info["out"]})
            model.set_swish(memory_efficient=False)
            if opts.norm_type in ["group", "none"]:
                replace_bn(model, opts)
            init_efficientnet(model)
        else:
                model = available_models[opts.model]["model"](pretrained=False, num_classes=data_info["out"], norm_layer=norm_layer)
                if pretrained:
                    model = load_modified_model(model, opts.model)
                elif "resnet" in opts.model or "resnext" in opts.model:
                    residual_normlayer_init(model)

    if len(opts.pipeline_splits) > 0:
        pipeline_model(model, opts.pipeline_splits)

    if opts.precision[-3:] == ".16":
        model.half()

    logging.info(model)

    return model


def get_module_and_parent_by_name(node, split_tokens):

    child_to_find = split_tokens[0]
    for name, child in node.named_children():
        if name == child_to_find:
            if len(split_tokens) == 1:
                return node, child, name
            else:
                return get_module_and_parent_by_name(child, split_tokens[1:])

    return None, None, None


def pipeline_model(model, pipeline_splits):
    """
    Split the model into stages.
    """
    for name, modules in model.named_modules():
        name = name.replace('.', '/')
        if name in pipeline_splits:
            logging.info('--------')
        logging.info(name)

    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        logging.info(f'Processing pipeline split {split_tokens}')
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            logging.warn(f'Split {split} not found')
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx+1, layer_to_call=node))


def replace_bn(model, opts):
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            stack.append(child)
            if isinstance(child, torch.nn.BatchNorm2d):
                if opts.norm_type == "group":
                    new_layer = torch.nn.GroupNorm(opts.norm_num_groups, child.num_features, child.eps, child.affine)
                else:
                    new_layer = torch.nn.Identity()
                replace_layer(node, name, new_layer)


def load_modified_model(model, model_name):
    if model_name in convert_model_names.keys():
        model_name = convert_model_names[model_name]

    model_url = model_urls[model_name]
    trained_state_dict = load_state_dict_from_url(model_url, progress=True)
    default_state_dict = model.state_dict()

    def get_weight(layer):
        if layer in trained_state_dict.keys():
            return trained_state_dict[layer]
        else:
            return default_state_dict[layer]

    corrected_state_dict = OrderedDict({layer: get_weight(layer) for layer in default_state_dict.keys()})
    model.load_state_dict(corrected_state_dict)
    return model


def get_norm_layer(opts):
    if opts.norm_type == "none":
        return torch.nn.Identity
    elif opts.norm_type == "batch":
        return torch.nn.BatchNorm2d
    elif opts.norm_type == "group":
        return lambda x: torch.nn.GroupNorm(opts.norm_num_groups, x)


def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)


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
            torch.nn.init.zeros_(norm_layer.weight)


def init_efficientnet(model):
    stack = [model]
    # Optimize the initialization of the model.
    while len(stack) != 0:
        node = stack.pop()
        if isinstance(node, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(node.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(node, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(node.weight)
        for name, child in node.named_children():
            stack.append(child)
