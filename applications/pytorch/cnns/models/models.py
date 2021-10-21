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
    'resnet18': {"model": torchvision.models.resnet18, "input_shape": (3, 224, 224)},
    'resnet34': {"model": torchvision.models.resnet34, "input_shape": (3, 224, 224)},
    'resnet50': {"model": torchvision.models.resnet50, "input_shape": (3, 224, 224)},
    'resnet101': {"model": torchvision.models.resnet101, "input_shape": (3, 224, 224)},
    'resnet152': {"model": torchvision.models.resnet152, "input_shape": (3, 224, 224)},
    'resnext50': {"model": torchvision.models.resnext50_32x4d, "input_shape": (3, 224, 224)},
    'resnext101': {"model": torchvision.models.resnext101_32x8d, "input_shape": (3, 224, 224)},
    'mobilenet': {"model": torchvision.models.mobilenet_v2, "input_shape": (3, 224, 224)},
    'mobilenet-v3-small': {"model": partial(create_mobilenetv3, 'small'), "input_shape": (3, 224, 224)},
    'mobilenet-v3-large': {"model": partial(create_mobilenetv3, 'large'), "input_shape": (3, 224, 224)},
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


def get_model(opts, data_info, pretrained=True, use_mixup=False, use_cutmix=False):
    """
    params:
    opts: contains the user defined command line parameters
    data info: the input and the output shape of the data
    pretrain: if it is true the weights are loaded from a publicly available pretrained model
    use_mixup: use on-device mixup augmentation
    use_cutmix: use on-device cutmix augmentation
    """
    norm_layer = get_norm_layer(opts)

    if opts.model in available_models:
        if 'efficientnet' in opts.model:
            model = available_models[opts.model]["model"](
                pretrained=pretrained,
                num_classes=data_info["out"],
                norm_layer=norm_layer,
                expand_ratio=opts.efficientnet_expand_ratio,
                group_dim=opts.efficientnet_group_dim,
            )
        else:
            model = available_models[opts.model]["model"](
                pretrained=False,
                num_classes=data_info["out"],
                norm_layer=norm_layer,
            )
            if "resnet" in opts.model or "resnext" in opts.model:
                residual_normlayer_init(model)  # Custom init for better training
            if pretrained and opts.model in model_urls.keys():
                model = load_modified_model(model, opts.model)

    if opts.precision[-3:] == ".16":
        model.half()

    if hasattr(opts, "recompute_checkpoints") and len(opts.recompute_checkpoints) > 0:
        model = recompute_model(model, opts.recompute_checkpoints)

    if hasattr(opts, "input_image_padding") and opts.input_image_padding:
        pad_first_conv(model)

    if len(opts.pipeline_splits) > 0:
        pipeline_model(model, opts.pipeline_splits)

    if opts.normalization_location == "ipu":
        cast = "half" if opts.precision[:3] == "16." else "full"
        model = NormalizeInputModel(model, datasets.normalization_parameters["mean"], datasets.normalization_parameters["std"], output_cast=cast)

    if use_mixup or use_cutmix:
        model = augmentations.AugmentationModel(model, use_mixup, use_cutmix, opts)

    logging.info(model)
    total_num_params = sum(p.numel() for p in model.parameters())
    logging.info("Total number of parameters: {:d}".format(total_num_params))
    # Use human readable names for each layer
    hooks = NameScopeHook(model)
    return model


def get_model_state_dict(model):
    model = _get_nested_model(model)
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    model = _get_nested_model(model)
    model.load_state_dict(state_dict)


def _get_nested_model(model):
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
            sys.exit()
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx+1, layer_to_call=node))


def get_norm_layer(opts):
    if opts.norm_type == "none":
        return torch.nn.Identity
    elif opts.norm_type == "batch":
        return partial(torch.nn.BatchNorm2d, momentum=opts.batchnorm_momentum, eps=opts.norm_eps)
    elif opts.norm_type == "group":
        return partial(torch.nn.GroupNorm, opts.norm_num_groups, eps=opts.norm_eps)


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
