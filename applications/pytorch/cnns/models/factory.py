# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torchvision
import logging
from functools import partial
import import_helper
from models.model_manipulator import ModelManipulator, name_match, get_module_from_node, next_node
from models.implementations.efficientnet import create_efficientnet
from models.implementations.mobilenet_v3 import MobileNetV3_Small, MobileNetV3_Large
from typing import Tuple
from inspect import getfullargspec

"""Collect the usable models. Following attributes are used:
   * model: a function, which provides the model
   * input_shape: The size of the input image
   * model_transformations: list of model transformation, each of the transformation defined as (match_fn, transformation_fn)
   * args_params_mapping: maps the args param as input of the creator function. Format: 'args.parameter': 'fn_input_param'
"""
available_models = {
    'resnet18': {
        'model': torchvision.models.resnet18,
        'input_shape': (3, 224, 224),
    },
    'resnet34': {
        'model': torchvision.models.resnet34,
        'input_shape': (3, 224, 224),
    },
    'resnet50': {
        'model': torchvision.models.resnet50,
        # Custom init for better training
        'model_transformations': [(lambda node: name_match(["layer./[^0]/bn.", "layer./0/downsample/1"])(node) and
                                  next_node(name_match("add.*"))(node) and
                                  torch.allclose(get_module_from_node(node).weight, torch.ones(1)),  # only change init if not pretrained
                                  lambda node: torch.nn.init.zeros_(get_module_from_node(node).weight), "BATCHNORM INIT")],
        'input_shape': (3, 224, 224),
    },
    'resnet101': {
        'model': torchvision.models.resnet101,
        'input_shape': (3, 224, 224),
    },
    'resnet152': {
        'model': torchvision.models.resnet152,
        'input_shape': (3, 224, 224),
    },
    'resnext50': {
        'model': torchvision.models.resnext50_32x4d,
        'input_shape': (3, 224, 224),
    },
    'resnext101': {
        'model': torchvision.models.resnext101_32x8d,
        'input_shape': (3, 224, 224),
    },
    'mobilenet': {
        'model': torchvision.models.mobilenet_v2,
        'input_shape': (3, 224, 224),
    },
    'efficientnet-b0': {
        'model': partial(create_efficientnet, 'efficientnet_b0'),
        'input_shape': (3, 224, 224),
        'args_params_mapping': {
            'efficientnet_expand_ratio': 'expand_ratio',
            'efficientnet_group_dim': 'group_dim',
        }
    },
    'efficientnet-b4': {
        'model': partial(create_efficientnet, 'efficientnet_b4'),
        'input_shape': (3, 380, 380),
        'args_params_mapping': {
            'efficientnet_expand_ratio': 'expand_ratio',
            'efficientnet_group_dim': 'group_dim',
        }
    },
    'mobilenet-v3-large': {
        'model': MobileNetV3_Large,
        'input_shape': (3, 224, 224)
    },
    'mobilenet-v3-small': {
        'model': MobileNetV3_Small,
        'input_shape': (3, 224, 224)
    }
}


def create_model(model_name: str, args, num_classes: int=1000, pretrained: bool=False, inference_mode: bool=False) -> torch.nn.Module:
    """Create model based on available_models"""
    if model_name not in available_models.keys():
        raise ValueError("Model {model_name} not found.")
    selected_model = available_models[model_name]
    if 'args_params_mapping' in selected_model.keys():
        # Add opt params
        params = {value: getattr(args, key) for key, value in selected_model.get('args_params_mapping', {}).items()}
    else:
        params = {}
    # if Model constructor has 'inference_mode' param add it
    if "inference_mode" in getfullargspec(selected_model['model']).args:
        params["inference_mode"] = inference_mode
    model = selected_model['model'](pretrained=pretrained,
                                    num_classes=num_classes,
                                    **params)
    # Set model in the right mode.
    if inference_mode:
        model.eval()
    else:
        model.train()
    # Apply manipulation on the created model
    model = ModelManipulator(model).transform_pipeline(selected_model.get('model_transformations', []))
    return model


# Values from https://arxiv.org/abs/2106.03640.
original_to_half_resolution = {
    224: 160,
    380: 252,
}


def model_input_shape(args, train: bool=True) -> Tuple:
    input_shape = available_models[args.model]["input_shape"]
    if train and hasattr(args, 'half_res_training') and args.half_res_training:
        return (
            input_shape[0],
            original_to_half_resolution[input_shape[1]],
            original_to_half_resolution[input_shape[2]],
        )
    return input_shape
