# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import types
import logging
import torch
import timm
from functools import partial
import import_helper
from models.model_manipulator import ModelManipulator, get_module_from_node, replace_op, name_match, replace_module

en_params = {
    "efficientnet_b0": {"channel_multiplier": 1.0, "depth_multiplier": 1.0, "dropout_rate": 0.2},
    "efficientnet_b4": {"channel_multiplier": 1.4, "depth_multiplier": 1.8, "dropout_rate": 0.4},
}


def create_efficientnet(
    model_name: str,
    pretrained: bool = False,
    num_classes: int = 1000,
    expand_ratio: int = 6,
    group_dim: int = 1,
    inference_mode: bool = False,
):
    """Creates an initialized EfficientNet model.
    Parameters:
        model_name: variant of the model.
        pretrained: if true the network is initialized with pretrained weights.
        norm_layer: a function that creates normalization layers instances.
        expand_ratio: expansion ratio, official EfficientNet uses 6.
        group_dim: dimensionality of the group convolution, official EfficientNet
            uses 1 (i.e. depthwise convolutions)
        inference_mode: create model in inference mode
    """
    model_variant = "tf_" + model_name if pretrained else model_name  # use tf variant for pretrained inference
    if pretrained and _creating_modified_en_model(expand_ratio, group_dim):
        pretrained = False
        logging.warn(
            "Pretrained EfficientNet model is not supported if "
            "the model is modified, please use default parameters with pretrained=True."
        )

    model_config = en_params[model_name]

    architecture_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e{:d}_c24_se0.25".format(expand_ratio)],
        ["ir_r2_k5_s2_e{:d}_c40_se0.25".format(expand_ratio)],
        ["ir_r3_k3_s2_e{:d}_c80_se0.25".format(expand_ratio)],
        ["ir_r3_k5_s1_e{:d}_c112_se0.25".format(expand_ratio)],
        ["ir_r4_k5_s2_e{:d}_c192_se0.25".format(expand_ratio)],
        ["ir_r1_k3_s1_e{:d}_c320_se0.25".format(expand_ratio)],
    ]

    round_channels_fn = partial(
        timm.models.efficientnet.round_channels,
        multiplier=model_config["channel_multiplier"],
    )

    model = timm.models.helpers.build_model_with_cfg(
        model_cls=timm.models.EfficientNet,
        variant=model_variant,
        default_cfg=timm.models.efficientnet.default_cfgs[model_variant],
        pretrained=pretrained,
        block_args=timm.models.efficientnet.decode_arch_def(architecture_def, model_config["depth_multiplier"]),
        num_features=round_channels_fn(1280),
        stem_size=32,
        round_chs_fn=round_channels_fn,
        act_layer=timm.models.efficientnet.resolve_act_layer({"act_layer": "swish"}),
        num_classes=num_classes,
        drop_rate=model_config["dropout_rate"],
        drop_path_rate=0.2,
    )
    if inference_mode:
        model.eval()
    else:
        model.train()
    return _common_efficientnet_manipulation(model, group_dim)


def _creating_modified_en_model(expand_ratio, group_dim):
    return not (expand_ratio == 6 and group_dim == 1)


def _common_efficientnet_manipulation(model: torch.nn.Module, group_dim: int):
    """Modify the given model, the following transformation applied:
    * Change the depthwise convolutions group dimensionality
    * Use adaptive_avg_pool2d instead of mean operation (speeds up on IPU)
    """
    manipulated_model = ModelManipulator(model)
    if group_dim > 1:
        # Modify the group_dim
        def group_conv(node):
            node = get_module_from_node(node)
            return timm.models.layers.create_conv2d(
                in_channels=node.in_channels,
                out_channels=node.out_channels,
                kernel_size=node.kernel_size[0],
                stride=node.stride[0],
                dilation=node.dilation[0],
                groups=max(node.in_channels // group_dim, 1),
            )

        model = manipulated_model.transform(name_match(".*conv_dw"), replace_module(group_conv))

    # speed optimisation mean() replaced to adaptive_avg_pool2d()
    def adaptive_avg_pool2d(x):
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)

    model = manipulated_model.transform(name_match("mean.*"), replace_op(lambda _: adaptive_avg_pool2d))
    return model
