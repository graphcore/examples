# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from functools import partial
import torch
import torchvision
import timm
import poptorch
import logging
from .model_manipulator import create_efficientnet, create_mobilenetv3, residual_normlayer_init, replace_layer, get_module_and_parent_by_name, \
                               load_modified_model, recompute_model, pad_first_conv, model_urls
import sys
import import_helper
import datasets
import datasets.augmentations as augmentations
from .mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small

available_models = {
    'resnet18': {
        "model": torchvision.models.resnet18,
        "input_shape": (3, 224, 224),
    },
    'resnet34': {
        "model": torchvision.models.resnet34,
        "input_shape": (3, 224, 224),
    },
    'resnet50': {
        "model": torchvision.models.resnet50,
        "input_shape": (3, 224, 224),
    },
    'resnet101': {
        "model": torchvision.models.resnet101,
        "input_shape": (3, 224, 224),
    },
    'resnet152': {
        "model": torchvision.models.resnet152,
        "input_shape": (3, 224, 224),
    },
    'resnext50': {
        "model": torchvision.models.resnext50_32x4d,
        "input_shape": (3, 224, 224),
    },
    'resnext101': {
        "model": torchvision.models.resnext101_32x8d,
        "input_shape": (3, 224, 224),
    },
    'mobilenet': {
        "model": torchvision.models.mobilenet_v2,
        "input_shape": (3, 224, 224),
    },
    'mobilenet-v3-small': {
        "model": partial(create_mobilenetv3, 'small'),
        "input_shape": (3, 224, 224),
    },
    'mobilenet-v3-large': {
        "model": partial(create_mobilenetv3, 'large'),
        "input_shape": (3, 224, 224),
    },
    'efficientnet-b0': {
        "model": partial(create_efficientnet, 'efficientnet-b0'),
        "input_shape": (3, 224, 224),
        'channel_multiplier': 1.0,
        'depth_multiplier': 1.0,
        'dropout_rate': 0.2,
    },
    'efficientnet-b4': {
        "model": partial(create_efficientnet, 'efficientnet-b4'),
        "input_shape": (3, 380, 380),
        'channel_multiplier': 1.4,
        'depth_multiplier': 1.8,
        'dropout_rate': 0.4,
    },
}

available_model_types = [
    torchvision.models.ResNet,
    torchvision.models.MobileNetV2,
    timm.models.EfficientNet,
    MobileNetV3_Large,
    MobileNetV3_Small
]

# Values from https://arxiv.org/abs/2106.03640.
original_to_half_resolution = {
    224: 160,
    380: 252,
}


def model_input_shape(args, train=True):
    input_shape = available_models[args.model]["input_shape"]
    if train and hasattr(args, 'half_res_training') and args.half_res_training:
        return (
            input_shape[0],
            original_to_half_resolution[input_shape[1]],
            original_to_half_resolution[input_shape[2]],
        )
    return input_shape


def get_model(args, data_info, pretrained=True, use_mixup=False, use_cutmix=False):
    """
    params:
    args: contains the user defined command line parameters
    data info: the input and the output shape of the data
    pretrain: if it is true the weights are loaded from a publicly available pretrained model
    use_mixup: use on-device mixup augmentation
    use_cutmix: use on-device cutmix augmentation
    """
    logging.info("Creating the model")
    norm_layer = get_norm_layer(args)

    if args.model in available_models:
        if 'efficientnet' in args.model:
            model = available_models[args.model]["model"](
                pretrained=pretrained,
                num_classes=data_info["out"],
                norm_layer=norm_layer,
                expand_ratio=args.efficientnet_expand_ratio,
                group_dim=args.efficientnet_group_dim,
            )
        else:
            model = available_models[args.model]["model"](
                pretrained=False,
                num_classes=data_info["out"],
                norm_layer=norm_layer,
            )
            if "resnet" in args.model or "resnext" in args.model:
                residual_normlayer_init(model)  # Custom init for better training
            if pretrained and args.model in model_urls.keys():
                model = load_modified_model(model, args.model)

    if args.precision[-3:] == ".16":
        model.half()

    if hasattr(args, "recompute_checkpoints") and len(args.recompute_checkpoints) > 0:
        model = recompute_model(model, args.recompute_checkpoints)

    if hasattr(args, "input_image_padding") and args.input_image_padding:
        pad_first_conv(model)

    if len(args.pipeline_splits) > 0:
        pipeline_model(model, args.pipeline_splits)

    if args.normalization_location == "ipu":
        cast = "half" if args.precision[:3] == "16." else "full"
        model = NormalizeInputModel(
            model,
            datasets.normalization_parameters["mean"],
            datasets.normalization_parameters["std"],
            output_cast=cast
        )

    if args.num_io_tiles > 0:
        model = OverlapModel(model)

    if use_mixup or use_cutmix:
        model = augmentations.AugmentationModel(model, use_mixup, use_cutmix, args)

    logging.info(model)
    total_num_params = sum(p.numel() for p in model.parameters())
    logging.info("Total number of parameters: {:d}".format(total_num_params))
    # Use human readable names for each layer
    hooks = NameScopeHook(model)
    return model


def get_model_state_dict(model):
    model = get_nested_model(model)
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    model = get_nested_model(model)
    model.load_state_dict(state_dict)


def get_nested_model(model):
    while not any(isinstance(model, mt) for mt in available_model_types) and not isinstance(model, torch.fx.GraphModule):
        if hasattr(model, 'model'):
            model = model.model
        elif hasattr(model, 'module'):
            model = model.module
        else:
            raise AttributeError(
                "The models._get_nested_model function encountered "
                "a non-expected nested model attribute. Maybe a new "
                "type needs to be added to models.available_model_types?")
    return model


def pipeline_model(model, pipeline_splits):
    """
    Split the model into stages.
    """
    for name, _ in model.named_modules():
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
            sys.exit(1)
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx+1, layer_to_call=node))


def get_norm_layer(args):
    if args.norm_type == "none":
        return torch.nn.Identity
    elif args.norm_type == "batch":
        return partial(torch.nn.BatchNorm2d, momentum=args.batchnorm_momentum, eps=args.norm_eps)
    elif args.norm_type == "group":
        return partial(torch.nn.GroupNorm, args.norm_num_groups, eps=args.norm_eps)


class OverlapModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, img):
        img = poptorch.set_overlap_for_input(img, poptorch.OverlapMode.OverlapAccumulationLoop)
        img = self.model(img)
        img = poptorch.set_overlap_for_output(img, poptorch.OverlapMode.OverlapAccumulationLoop)
        return img


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


class NameScopeHook():
    def __init__(self, module):
        self.hooks = []
        for name, m in module.named_modules():
            self.hooks.append(
                m.register_forward_pre_hook(
                    partial(self.enter_fn, name=name)))
            self.hooks.append(
                m.register_forward_hook(self.exit_fn))

    def enter_fn(self, module, input, name):
        torch.ops.poptorch.push_name_scope(name.split(".")[-1])

    def exit_fn(self, module, input, output):
        torch.ops.poptorch.pop_name_scope()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
