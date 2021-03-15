# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import torch
import sys
sys.path.append('..')
import models


class TestCustomEfficientNet:
    @pytest.mark.category1
    @pytest.mark.ipus(0)
    def test_expand_ratio(self):
        model = models.create_efficientnet("efficientnet-b0", expand_ratio=2)
        for idx, block in enumerate(model._blocks):
            if idx > 0:
                conv = block._expand_conv
                assert conv.out_channels == conv.in_channels * 2

    @pytest.mark.category1
    @pytest.mark.ipus(0)
    def test_group_dim(self):
        model = models.create_efficientnet("efficientnet-b0", group_dim=2)
        for block in model._blocks:
            conv = block._depthwise_conv
            assert conv.groups * 2 == conv.in_channels


class TestGroupNormConversion:
    @staticmethod
    def create_norm():
        return lambda x: torch.nn.GroupNorm(2, x)

    @pytest.mark.category1
    @pytest.mark.ipus(0)
    def test_single_element_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.bn = torch.nn.BatchNorm2d(1)

        model = Model()
        group_norm = TestGroupNormConversion.create_norm()
        models.replace_bn(model, group_norm)
        assert isinstance(model.bn, torch.nn.GroupNorm)

    @pytest.mark.category1
    @pytest.mark.ipus(0)
    def test_sequential_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 2, 3),
                    torch.nn.BatchNorm2d(1)
                )

        model = Model()
        group_norm = TestGroupNormConversion.create_norm()
        models.replace_bn(model, group_norm)

        assert isinstance(model.layers[1], torch.nn.GroupNorm) and isinstance(model.layers[0], torch.nn.Conv2d)

    @pytest.mark.category1
    @pytest.mark.ipus(0)
    def test_nested_model(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super(Block, self).__init__()
                self.conv = torch.nn.Conv2d(1, 2, 3)
                self.bn = torch.nn.BatchNorm2d(1)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layers = torch.nn.Sequential(
                    Block(),
                    torch.nn.Sequential(
                        Block(),
                        torch.nn.BatchNorm2d(1)
                    ),
                    torch.nn.BatchNorm2d(1)
                )

        model = Model()
        group_norm = TestGroupNormConversion.create_norm()
        models.replace_bn(model, group_norm)

        assert isinstance(model.layers[0].conv, torch.nn.Conv2d) and \
            isinstance(model.layers[0].bn, torch.nn.GroupNorm) and \
            isinstance(model.layers[1][0].conv, torch.nn.Conv2d) and \
            isinstance(model.layers[1][0].bn, torch.nn.GroupNorm) and \
            isinstance(model.layers[2], torch.nn.GroupNorm)
