# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import torch
import poptorch
import import_helper
import models


class TestCustomEfficientNet:
    def test_expand_ratio(self):
        model = models.create_efficientnet("efficientnet-b0", expand_ratio=2)
        for i, block in enumerate(model.blocks):
            if i == 0:
                continue
            for layer in block:
                conv = layer.conv_pw
                assert conv.out_channels == conv.in_channels * 2

    def test_group_dim(self):
        model = models.create_efficientnet("efficientnet-b0", group_dim=2)
        for block in model.blocks:
            for layer in block:
                conv = layer.conv_dw
                assert conv.groups * 2 == conv.in_channels


class TestRecomputation:
    @classmethod
    def check_recompute(cls, traced_model, checkpoint_name):
        for node in traced_model.graph.nodes:
            if str(node) == checkpoint_name:
                assert "recomputation_checkpoint" in str(node.next), f"Layer conversion to recompute checkpoint doesn't work for{node}!"


    @classmethod
    def default_args(cls):
        class HelperClass: pass
        args = HelperClass()
        args.model = "resnet18"
        args.precision = "16.16"
        args.norm_type = "batch"
        args.norm_eps = 1e-5
        args.normalization_location = "none"
        args.pipeline_splits = []
        args.batchnorm_momentum = 0.9
        args.num_io_tiles = 0
        return args

    def test_recomputation_model_by_name(self):
        args = TestRecomputation.default_args()
        args.recompute_checkpoints = ["layer2/0/conv2"]
        model = models.get_model(args, {"out": 1000}, pretrained=False)
        TestRecomputation.check_recompute(model, "layer2/0/conv2")

    def test_recomputation_pipelined_model_by_name(self):
        args = TestRecomputation.default_args()
        args.pipeline_splits = ["layer2"]
        args.recompute_checkpoints = ["layer2/0/conv2"]
        model = models.get_model(args, {"out": 1000}, pretrained=False)
        TestRecomputation.check_recompute(model, "layer2/0/conv2")

    def test_recomputation_normalized_model_by_name(self):
        args = TestRecomputation.default_args()
        args.normalization_location = "ipu"
        args.recompute_checkpoints = ["layer2/0/conv2"]
        model = models.get_model(args, {"out": 1000}, pretrained=False)
        TestRecomputation.check_recompute(model.model, "layer2/0/conv2")


    def test_recomutation_regex_conv(self):
        args = TestRecomputation.default_args()
        args.recompute_checkpoints = [".*conv.*"]
        model = models.get_model(args, {"out": 1000}, pretrained=False)
        TestRecomputation.check_recompute(model, "conv1")
        TestRecomputation.check_recompute(model, "layer2/0/conv1")
        TestRecomputation.check_recompute(model, "layer2/0/conv2")


@pytest.mark.parametrize("precision", ["half", "full"])
@pytest.mark.parametrize("bias", [True, False])
def test_convpadding_3ch_vs_4ch_forward(precision, bias):
    torch.manual_seed(0)
    conv_3ch = torch.nn.Conv2d(3, 32, (3, 3), bias=bias)
    if precision == "half":
        conv_3ch.half()
    conv_4ch = models.PaddedConv(conv_3ch)
    sample_input = torch.rand(4, 3, 32, 32)
    if precision == "half":
        sample_input.half()
        conv_4ch.half()
    ipu_conv_3ch = poptorch.inferenceModel(conv_3ch, poptorch.Options())
    ipu_conv_4ch = poptorch.inferenceModel(conv_4ch, poptorch.Options())
    result3 = ipu_conv_3ch(sample_input)
    result4 = ipu_conv_4ch(sample_input)
    assert torch.allclose(result3, result4, rtol=1e-03, atol=1e-04)
