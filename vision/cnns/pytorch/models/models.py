# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import logging
import import_helper
from models.factory import create_model
from models.model_manipulator import (
    ModelManipulator,
    get_norm_layer,
    name_match,
    type_match,
    get_module_from_node,
    replace_module,
    get_node_name,
    insert_after,
)
from models.model_wrappers import NormalizeInputModel, OverlapModel
from models.implementations.optimisation import PaddedConv
from models.loss import LabelSmoothing, TrainingModelWithLoss, weighted_nll_loss
import datasets
import datasets.augmentations as augmentations
import utils


def get_model(
    args,
    data_info,
    pretrained: bool = True,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    with_loss: bool = False,
    inference_mode: bool = False,
) -> torch.nn.Module:
    """
    params:
    args: contains the user defined command line parameters
    data info: the input and the output shape of the data
    pretrain: if it is true the weights are loaded from a publicly available pretrained model
    use_mixup: use on-device mixup augmentation
    use_cutmix: use on-device cutmix augmentation
    inference_mode: create model in 'eval' model or 'train' mode
    """
    logging.info("Creating the model")
    model = create_model(
        args.model, args, num_classes=data_info["out"], pretrained=pretrained, inference_mode=inference_mode
    )

    model_manipulator = ModelManipulator(model)
    # Set norm layers
    if not args.norm_type == "batch":
        model = model_manipulator.transform(
            type_match(torch.nn.BatchNorm2d),
            replace_module(lambda node: get_norm_layer(args)(get_module_from_node(node).num_features)),
            "NORMLAYER REPLACE",
        )
    # Insert recompute checkpoints
    model = model_manipulator.transform(
        name_match(getattr(args, "recompute_checkpoints", [])),
        insert_after(lambda node: poptorch.recomputationCheckpoint),
        "RECOMPUTE CHECKPOINT",
    )

    # Handle pipelining
    model = model_manipulator.transform(
        name_match(args.pipeline_splits),
        replace_module(
            lambda node: poptorch.BeginBlock(
                ipu_id=1 + args.pipeline_splits.index(get_node_name(node)), layer_to_call=get_module_from_node(node)
            )
        ),
        "PIPELINE",
    )
    if hasattr(args, "input_image_padding") and args.input_image_padding:
        model = model_manipulator.transform(
            ModelManipulator.first_match(type_match(torch.nn.Conv2d)),
            replace_module(lambda node: PaddedConv(get_module_from_node(node))),
            "INPUT PADDING",
        )

    # Convert the model to half after all model manipulations have been made
    if args.precision[-3:] == ".16":
        model.half()

    nested_model = model
    if args.normalization_location == "ipu":
        cast = "half" if args.precision[:3] == "16." else "full"
        model = NormalizeInputModel(
            model, datasets.normalization_parameters["mean"], datasets.normalization_parameters["std"], output_cast=cast
        )
    if use_mixup or use_cutmix:
        model = augmentations.AugmentationModel(model, use_mixup, use_cutmix, args)

    model_summary(model)
    if with_loss:
        if use_mixup or use_cutmix:

            def mix_classification_loss(output, labels):
                inner_model = model
                # Find AugmentationModel
                while not isinstance(inner_model, augmentations.AugmentationModel):
                    inner_model = inner_model.model
                log_preds, coeffs = output
                all_labels, weights = inner_model.mix_labels(labels, coeffs)
                return weighted_nll_loss(log_preds, all_labels, weights)

            losses = LabelSmoothing(mix_classification_loss, label_smoothing=args.label_smoothing).get_losses_list()
        else:
            losses = LabelSmoothing(
                torch.nn.NLLLoss(reduction="mean"), label_smoothing=args.label_smoothing
            ).get_losses_list()
        model = TrainingModelWithLoss(model, losses, [utils.accuracy])
    name_scope_hook(model)  # Use human readable names for each layer

    if args.num_io_tiles > 0:
        model = OverlapModel(model)

    # Put it into tuple to avoid too many recursive call problems: if no wrapper applied, it can be infinite recursion.
    # PyTorch doesn't look into non Parameter / Module type instances.
    model.nested_model = (nested_model,)

    return model


def model_summary(model):
    """Log the summary of the model"""
    logging.info(model)
    total_num_params = sum(p.numel() for p in model.parameters())
    logging.info("Total number of parameters: {:d}".format(total_num_params))


def name_scope_hook(module: torch.nn.Module):
    """Provides human readable names in PopVision"""
    for name, m in module.named_modules():
        m.register_forward_pre_hook(lambda _module, _inp: torch.ops.poptorch.push_name_scope(name.split(".")[-1]))
        m.register_forward_hook(lambda _module, _inp, _out: torch.ops.poptorch.pop_name_scope())


def get_model_state_dict(model: torch.nn.Module):
    return model.nested_model[0].state_dict()


def load_model_state_dict(model: torch.nn.Module, state_dict):
    model.nested_model[0].load_state_dict(state_dict)
