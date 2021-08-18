# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import OrderedDict
from functools import partial
import torch
import torchvision
import poptorch
import logging
from .model_manipulator import create_efficientnet, residual_normlayer_init, replace_layer, get_module_and_parent_by_name, load_modified_model, full_precision_norm, recompute_model
import sys
sys.path.append('..')
import datasets
import datasets.augmentations as augmentations

available_models = {'resnet18': {"model": torchvision.models.resnet18, "input_shape": (3, 224, 224)},
                    'resnet34': {"model": torchvision.models.resnet34, "input_shape": (3, 224, 224)},
                    'resnet50': {"model": torchvision.models.resnet50, "input_shape": (3, 224, 224)},
                    'resnet101': {"model": torchvision.models.resnet101, "input_shape": (3, 224, 224)},
                    'resnet152': {"model": torchvision.models.resnet152, "input_shape": (3, 224, 224)},
                    'resnext50': {"model": torchvision.models.resnext50_32x4d, "input_shape": (3, 224, 224)},
                    'resnext101': {"model": torchvision.models.resnext101_32x8d, "input_shape": (3, 224, 224)},
                    'mobilenet': {"model": torchvision.models.mobilenet_v2, "input_shape": (3, 224, 224)},
                    'efficientnet-b0': {"model": partial(create_efficientnet, 'efficientnet-b0'), "input_shape": (3, 224, 224)},
                    'efficientnet-b1': {"model": partial(create_efficientnet, 'efficientnet-b1'), "input_shape": (3, 240, 240)},
                    'efficientnet-b2': {"model": partial(create_efficientnet, 'efficientnet-b2'), "input_shape": (3, 260, 260)},
                    'efficientnet-b3': {"model": partial(create_efficientnet, 'efficientnet-b3'), "input_shape": (3, 300, 300)},
                    'efficientnet-b4': {"model": partial(create_efficientnet, 'efficientnet-b4'), "input_shape": (3, 380, 380)},
                    'efficientnet-b5': {"model": partial(create_efficientnet, 'efficientnet-b5'), "input_shape": (3, 456, 456)},
                    'efficientnet-b6': {"model": partial(create_efficientnet, 'efficientnet-b6'), "input_shape": (3, 528, 528)},
                    'efficientnet-b7': {"model": partial(create_efficientnet, 'efficientnet-b7'), "input_shape": (3, 600, 600)}
                    }


def get_model(opts, data_info, pretrained=True, mixup=False):
    """
    params:
    opts: contains the user defined command line parameters
    data info: the input and the output shape of the data
    pretrain: if it is true the weights are loaded from a publicly available pretrained model
    mixup: use on-device mixup augmentation
    """
    norm_layer = get_norm_layer(opts)

    if opts.model in available_models:
        if 'efficientnet' in opts.model:
            model = available_models[opts.model]["model"](pretrained=pretrained, num_classes=data_info["out"], norm_layer=norm_layer,
                                                          expand_ratio=opts.efficientnet_expand_ratio, group_dim=opts.efficientnet_group_dim)
        else:
            model = available_models[opts.model]["model"](pretrained=False, num_classes=data_info["out"], norm_layer=norm_layer)
            if "resnet" in opts.model or "resnext" in opts.model:
                residual_normlayer_init(model)  # Custom init for better training
            if pretrained:
                model = load_modified_model(model, opts.model)

    if len(opts.pipeline_splits) > 0:
        pipeline_model(model, opts.pipeline_splits)

    if opts.precision[-3:] == ".16":
        model.half()
        if opts.full_precision_norm:
            if opts.norm_type == "batch":
                norm_layer = torch.nn.BatchNorm2d
                full_precision_norm(model, norm_layer)
            elif opts.norm_type == "group":
                norm_layer = torch.nn.GroupNorm
                full_precision_norm(model, norm_layer)

    if hasattr(opts, "recompute_checkpoints") and len(opts.recompute_checkpoints) > 0:
        recompute_model(model, opts.recompute_checkpoints)

    if opts.normalization_location == "ipu":
        cast = "half" if opts.precision[:3] == "16." else "full"
        model = NormalizeInputModel(model, datasets.normalization_parameters["mean"], datasets.normalization_parameters["std"], output_cast=cast)

    if mixup:
        model = augmentations.MixupModel(model)

    logging.info(model)
    return model


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
            logging.error(f'Split {split} not found')
            sys.exit()
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx+1, layer_to_call=node))


def get_norm_layer(opts):
    if opts.norm_type == "none":
        return torch.nn.Identity
    elif opts.norm_type == "batch":
        return lambda x: torch.nn.BatchNorm2d(x, momentum=opts.batchnorm_momentum)
    elif opts.norm_type == "group":
        return lambda x: torch.nn.GroupNorm(opts.norm_num_groups, x)


class NormalizeInputModel(torch.nn.Module):
    def __init__(self, model, mean, std, output_cast=None):
        super().__init__()
        self.model = model
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mul = (1.0/(255.0 * std)).view(-1, 1, 1)
        self.sub = (mean / std).view(-1, 1, 1)
        self.output_cast = output_cast
        if output_cast == "full":
            self.mul, self.sub = self.mul.float(), self.sub.float()
        elif output_cast == "half":
            self.mul, self.sub = self.mul.half(), self.sub.half()


    def forward(self, img):
        if self.output_cast == "half":
            img = img.half()
        elif self.output_cast == "full":
            img = img.float()
        img = img.mul(self.mul)
        img = img.sub(self.sub)
        return self.model(img)
