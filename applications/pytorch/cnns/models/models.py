# Copyright 2020 Graphcore Ltd.
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

available_models = {'resnet18': torchvision.models.resnet18,
                    'resnet34': torchvision.models.resnet34,
                    'resnet50': torchvision.models.resnet50,
                    'resnet101': torchvision.models.resnet101,
                    'resnet152': torchvision.models.resnet152,
                    'resnext50': torchvision.models.resnext50_32x4d,
                    'resnext101': torchvision.models.resnext101_32x8d,
                    'mobilenet': torchvision.models.mobilenet_v2,
                    'efficientnet-b0': EfficientNet,
                    'efficientnet-b1': EfficientNet,
                    'efficientnet-b2': EfficientNet,
                    'efficientnet-b3': EfficientNet,
                    'efficientnet-b4': EfficientNet,
                    'efficientnet-b5': EfficientNet,
                    'efficientnet-b6': EfficientNet,
                    'efficientnet-b7': EfficientNet
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
                model = available_models[opts.model].from_pretrained(opts.model)
            else:
                model = available_models[opts.model].from_name(opts.model, {"num_classes": data_info["out"]})
            model.set_swish(memory_efficient=False)
            if opts.normlayer in ["group", "none"]:
                replace_bn(model, opts)
        else:
                model = available_models[opts.model](pretrained=False, num_classes=data_info["out"], norm_layer=norm_layer)
                if pretrained:
                    model = load_modified_model(model, opts.model)

    if len(opts.pipeline_splits) > 0:
        pipeline_model(model, opts.pipeline_splits)

    if opts.data == "synthetic":
        model = convert_to_syntetic(model, data_info["in"], len(opts.pipeline_splits) > 0)

    if opts.precision == "half":
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

    """
    for name, modules in model.named_modules():
        name = name.replace('.', '/')
        if name in pipeline_splits:
            logging.info('--------')
        logging.info(name)

    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        print(split_tokens)
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            logging.warn(f'Split {split} not found')
        else:
            replace_layer(parent, field_or_idx_str, poptorch.IPU(ipu_id=split_idx+1, layer_to_call=node))


def convert_to_syntetic(model, input_shape, pipelining):
    """

    """
    class SyntheticDataModel(torch.nn.Module):
        def __init__(self, model, input_shape, pipelining=False):
            super(SyntheticDataModel, self).__init__()
            self. model = model
            self.input_shape = input_shape
            self.pipelining = pipelining

        def forward(self, x):
            shape = x.size() + self.input_shape
            if self.pipelining:
                synt_data = torch.ones(shape) + x[0].float()
            else:
                synt_data = torch.ones(shape)
            return self.model(synt_data)
    return SyntheticDataModel(model, input_shape, pipelining)


def replace_bn(model, opts):
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            stack.append(child)
            if isinstance(child, torch.nn.BatchNorm2d):
                if opts.normlayer == "group":
                    new_layer = torch.nn.GroupNorm(opts.groupnorm_group_num, child.num_features, child.eps, child.affine)
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
    if opts.normlayer == "none":
        return torch.nn.Identity
    elif opts.normlayer == "batch":
        return torch.nn.BatchNorm2d
    elif opts.normlayer == "group":
        return lambda x: torch.nn.GroupNorm(opts.groupnorm_group_num, x)


def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)
