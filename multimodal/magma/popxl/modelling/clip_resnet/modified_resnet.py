# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from clip.model import ModifiedResNet as ClipModifiedResNet

from popxl_addons import Module, NamedTensors, NamedVariableFactories
from popxl_addons.layers import Conv2D
from popxl import ops, Tensor
from popxl.utils import to_numpy
import popxl
from configs import ResNetConfig
from .stem import Stem
from .batch_norm import BatchNorm2D
from .attention_pool import AttentionPool
from .bottleneck import Bottleneck

__all__ = ["ModifiedResNet"]


class ModifiedResNet(Module):
    """
    Clip ModifiedResnet.
    Inference only: batch normalisation layers can work with a baked running mean and running vars,
    but these values won't be updated.
    Differences from torchvision ResNet
        - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
        - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, config: ResNetConfig, pool: bool = False, cache: bool = False):
        super().__init__(cache)
        self.config = config
        self.pool = pool
        self.stem = Stem(config)
        if pool:
            self.attnpool = AttentionPool(config)

        # this is a *mutable* variable used during construction.
        # Changes each time _make_layer is called
        self._inplanes = config.width
        self.added_layers = 0

    def build(self, x: Tensor):
        x = self.stem(x)
        x = self.layer(x, self.config.width, self.config.layers[0])
        x = self.layer(x, self.config.width * 2, self.config.layers[1], stride=2)
        x = self.layer(x, self.config.width * 4, self.config.layers[2], stride=2)
        x = self.layer(x, self.config.width * 8, self.config.layers[3], stride=2)
        if self.pool:
            x = self.attnpool(x)
        return x

    def layer(self, x, planes, blocks, stride=1):
        # group layer variables under the scope layer1, layer2, ...
        # to match CLIP naming convention
        self.added_layers += 1
        # first bottleneck of a layer can have stride > 1
        first_block = Bottleneck(self._inplanes, planes, stride)
        facts, first_block_graph = first_block.create_graph(x)
        # add variables under the scope
        ts = self.add_variable_inputs(f"layer.{self.added_layers}.{0}", facts, overwrite=True)
        # call
        (x,) = first_block_graph.bind(ts).call(x)

        # following layers are all the same, and don't use strides
        # we OUTLINE them
        self._inplanes = planes * Bottleneck.expansion
        block = Bottleneck(self._inplanes, planes)
        facts, block_graph = block.create_graph(x)
        for i in range(1, blocks):
            # add different variables for each block
            ts = self.add_variable_inputs(f"layer.{self.added_layers}.{i}", facts, overwrite=True)
            # call
            (x,) = block_graph.bind(ts).call(x)
        return x

    @staticmethod
    def _clip_layer_mapping(clip_model: ClipModifiedResNet, config: ResNetConfig, idx: int, variables: NamedTensors):
        clip_layer = getattr(clip_model, f"layer{idx}")
        layer_vars = variables.layer[idx]
        mapping = {}
        layers_blocks = config.layers[idx - 1]
        for block in range(layers_blocks):
            mapping.update(Bottleneck.clip_mapping(clip_layer[block], layer_vars[block]))
        return mapping

    @staticmethod
    def clip_mapping(clip_model: ClipModifiedResNet, config: ResNetConfig, variables: NamedTensors, pool: bool = False):
        state_dict = Stem.clip_mapping(clip_model, variables.stem)
        for i in range(4):
            state_dict.update(ModifiedResNet._clip_layer_mapping(clip_model, config, i + 1, variables))
        if pool:
            state_dict.update(AttentionPool.clip_mapping(clip_model.attnpool, variables.attnpool))
        return state_dict
