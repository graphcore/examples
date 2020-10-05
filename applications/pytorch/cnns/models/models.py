# Copyright 2020 Graphcore Ltd.
import torch
import torchvision
import poptorch
from efficientnet_pytorch import EfficientNet
import logging

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


def get_model(opts, data_shape):
    """
    Factory method that creates the requested model for both inference
    and training.
    """
    if opts.model in available_models:
        if 'efficientnet' in opts.model:
            model = available_models[opts.model].from_pretrained(opts.model)
            model.set_swish(memory_efficient=False)
        else:
            model = available_models[opts.model](pretrained=True)

    if len(opts.pipeline_splits) > 0:
        pipeline_model(model, opts.pipeline_splits)

    if opts.data == "synthetic":
        model = convert_to_syntetic(model, data_shape["in"])

    if opts.precision == "half":
        raise Exception("Half precision is not supported yet")

    logging.info(model)

    return model


def get_module_and_parent_by_name(node, split_tokens):
    """
    Auxiliary function for pipelining a model. It will look for
    the requested pipelining split in a recursive fashion and
    returns the first layer in the pipeline stage, its parent layer,
    and either the field or index of the module in the parent structure.
    """
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
    Function to pipeline a model into multiple stages. A pipeline split
    is represented by the layer name in the form layerA/layerB/layerC.
    Any Pytorch model layer can be chosen at any abitrary depth.

    """
    for name, modules in model.named_modules():
        name = name.replace('.', '/')
        if name in pipeline_splits:
            logging.info('--------')
        logging.info(name)

    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        logging.info(split_tokens)
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            logging.warn(f'Split {split} not found')
        elif isinstance(parent, torch.nn.Sequential):
            parent[int(field_or_idx_str)] = poptorch.IPU(ipu_id=split_idx+1, layer_to_call=node)
        else:
            setattr(parent, field_or_idx_str, poptorch.IPU(ipu_id=split_idx+1, layer_to_call=node))


def convert_to_syntetic(model, input_shape):
    """
    Currently Poptorch does not support synthetic mode (no host->device IO)
    This method tries to circumvent that limitation by creating a tensor with
    correct shape, at every forward pass. Note that this is note a true no IO
    mode, as there is still data flowing to and from the device, but an attempt
    at reducing the data exchange to a minimum.
    """
    class SyntheticDataModel(torch.nn.Module):
        def __init__(self, model, input_shape):
            super(SyntheticDataModel, self).__init__()
            self. model = model
            self.input_shape = input_shape

        def forward(self, x):
            shape = x.size() + self.input_shape
            synt_data = torch.ones(shape) + x[0].float()
            return self.model(synt_data)
    return SyntheticDataModel(model, input_shape)
