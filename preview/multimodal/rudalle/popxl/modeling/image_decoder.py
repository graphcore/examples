# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import math
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear
from popxl_addons.ops.replicated_all_reduce_TP import \
    replicated_all_reduce_identical_grad_inputs

import popxl
from popxl import ReplicaGrouping, ops
from popxl.utils import to_numpy
from utils import shard


class Conv2d(addons.Module):
    def __init__(self, n_in: int, n_out: int, kw: int, bias: bool = True, amp: List[float] = None,
                 replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kw = kw
        self.bias = bias
        self.amp = amp
        self.replica_grouping = replica_grouping

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_variable_input(
                "weight",
                partial(np.random.normal, 0, 1/math.sqrt(self.n_in*self.kw**2),
                        (self.n_out, self.n_in, self.kw, self.kw)),
                x.dtype,
                replica_grouping=self.replica_grouping
            )
        pad = (self.kw - 1) // 2
        y = ops.conv(x, w, padding=(pad, pad, pad, pad), enable_conv_dithering=[1],
                     available_memory_proportions=self.amp)

        if self.bias:
            b = self.add_variable_input(
                    "bias",
                    partial(np.zeros, self.n_out),
                    x.dtype,
                    replica_grouping=self.replica_grouping
                )
            y = y + b.reshape_((1, self.n_out, 1, 1))
        return y

    @staticmethod
    def hf_mapping(variables: NamedTensors, hf_model, n_shards=1, axis=0, bias=True) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.weight.dtype
        weights = {
            variables.weight: shard(to_numpy(hf_model.weight.data, dtype), n_shards, axis=axis)
        }

        if bias:
            weights.update(
                {variables.bias: shard(to_numpy(hf_model.bias.data, dtype), n_shards, axis=0)})
        return weights


class GroupNorm(addons.Module):
    def __init__(self, n_shards = 1, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.n_shards = n_shards
        self.replica_grouping = replica_grouping

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_variable_input(
                "weight",
                partial(np.ones, x.shape[1]),
                x.dtype,
                replica_grouping=self.replica_grouping
            )
        b = self.add_variable_input(
                "bias",
                partial(np.ones, x.shape[1]),
                x.dtype,
                replica_grouping=self.replica_grouping
            )

        return ops.group_norm(x, w, b, 32//self.n_shards, eps=1e-6)

    @staticmethod
    def hf_mapping(variables: NamedTensors, hf_model, n_shards=1) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.weight.dtype
        weights = {
            variables.weight: shard(to_numpy(hf_model.weight.data, dtype), n_shards, axis=0),
            variables.bias: shard(to_numpy(hf_model.bias.data, dtype), n_shards, axis=0),
        }

        return weights


class Upsample(addons.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.conv = Conv2d(n_in, n_in, kw=3, amp=[0.04])

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = ops.interpolate(x, scale_factor=(1, 1, 2.0, 2.0), mode="nearest")
        x = self.conv(x)
        return x

    @staticmethod
    def hf_mapping(variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        return Conv2d.hf_mapping(variables.conv, hf_model.conv)


class Block(addons.Module):
    def __init__(self, n_shards: int = 1, replica_grouping: Optional[ReplicaGrouping] = None,
                 n_in: int = None, n_out: int = None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_shards = n_shards
        self.replica_grouping = replica_grouping

        self.gn_1 = GroupNorm()
        self.conv_1 = Conv2d(n_in,
                             n_out//n_shards,
                             kw=3,
                             bias=True,
                             amp=([0.04] if n_out==128 or n_in*n_out==2**17 else None),
                             replica_grouping = replica_grouping)

        self.gn_2 = GroupNorm(n_shards, replica_grouping)
        self.conv_2 = Conv2d(n_out//n_shards,
                             n_out,
                             kw=3,
                             bias=(n_shards==1),
                             amp=([0.04] if n_out==128 else None),
                             replica_grouping = replica_grouping)

        if self.n_in != self.n_out:
            self.nin_shortcut = Conv2d(n_in, n_out, kw=1)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        out = self.gn_1(x)
        out = ops.swish_(out)
        out = self.conv_1(out)

        out = self.gn_2(out)
        out = ops.swish_(out)
        out = self.conv_2(out)

        if self.n_shards > 1:
            out = replicated_all_reduce_identical_grad_inputs(out)
            bias = self.add_variable_input('bias', lambda: np.zeros(out.shape[1]), out.dtype)
            out = out + bias.reshape_((1, out.shape[1], 1, 1))

        if self.n_in != self.n_out:
            x = self.nin_shortcut(x)

        return x + out

    @staticmethod
    def hf_mapping(variables: NamedTensors, hf_model, n_shards = 1) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.conv_1.weight.dtype
        weights = GroupNorm.hf_mapping(variables.gn_1, hf_model.norm1)
        weights.update(Conv2d.hf_mapping(variables.conv_1, hf_model.conv1, n_shards))
        weights.update(GroupNorm.hf_mapping(variables.gn_2, hf_model.norm2, n_shards))
        if n_shards > 1:
            weights.update(Conv2d.hf_mapping(variables.conv_2, hf_model.conv2, n_shards, 1, False))
            weights.update({variables.bias: to_numpy(hf_model.conv2.bias.data, dtype)})
        else:
            weights.update(Conv2d.hf_mapping(variables.conv_2, hf_model.conv2, True))

        return weights


class Attention(addons.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.gn = GroupNorm()
        self.c_attn = Conv2d(n_in, n_in*3, kw=1)
        self.c_proj = Conv2d(n_in, n_in, kw=1)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x_shape = x.shape
        norm = self.gn(x)

        qkv_act = self.c_attn(norm).reshape_((x_shape[0], x_shape[1]*3, -1))
        query, key, value = ops.split(qkv_act, 3, axis=1)

        query = query.transpose_((0, 2, 1))
        attn_weights = query @ key
        attn_weights /= math.sqrt(x_shape[1])
        attn_scores = ops.softmax(attn_weights, axis=2)

        attn_scores = attn_scores.transpose_((0, 2, 1))
        attn_output = value @ attn_scores
        attn_output = attn_output.reshape_(x_shape)

        attn_output = self.c_proj(attn_output)

        return x + attn_output

    @staticmethod
    def hf_mapping(variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.c_attn.weight.dtype
        weights = {
            variables.c_attn.weight: np.concatenate([to_numpy(hf_model.q.weight.data, dtype),
                                                     to_numpy(hf_model.k.weight.data, dtype),
                                                     to_numpy(hf_model.v.weight.data, dtype)]),
            variables.c_attn.bias: np.concatenate([to_numpy(hf_model.q.bias.data, dtype),
                                                   to_numpy(hf_model.k.bias.data, dtype),
                                                   to_numpy(hf_model.v.bias.data, dtype)]),
        }
        weights.update(GroupNorm.hf_mapping(variables.gn, hf_model.norm))
        weights.update(Conv2d.hf_mapping(variables.c_proj, hf_model.proj_out, True))
        return weights


class Decoder(addons.Module):
    def __init__(self, config, replica_grouping: Optional[ReplicaGrouping]=None):
        super().__init__()
        self.channels = config.channels
        self.n_resolutions = len(self.channels)
        self.n_blocks = config.n_blocks
        self.image_size = config.image_size
        self.n_shards = config.ipus
        self.replica_grouping = replica_grouping

        n_in = self.channels[0]
        self.conv_in = Conv2d(config.z_channel, n_in, kw=3)
        upsampling_list = []
        feature_size = self.image_size // 2**(self.n_resolutions-2)
        for i in range(self.n_resolutions):
            block = []
            if i > 0:
                for _ in range(self.n_blocks):
                    block.append(Block(n_shards=1, replica_grouping=None,
                                       n_in=n_in, n_out=self.channels[i]))
                    n_in = self.channels[i]
            n_in = self.channels[i]

            upsampling = addons.Module()
            if block:
                upsampling.block = addons.Module.from_list(block)
            if i != self.n_resolutions-1:
                upsampling.upsample = Upsample(n_in)
                feature_size = feature_size * 2
            upsampling_list.append(upsampling)
        self.up = addons.Module.from_list(upsampling_list)

        self.gn_out = GroupNorm()
        self.conv_out = Conv2d(n_in, 3, kw=3)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = self.conv_in(x)

        args_blk, graph_blk = Block(n_shards=self.n_shards,
                                    replica_grouping=self.replica_grouping,
                                    n_in=self.channels[0], n_out=self.channels[0]).create_graph(x)
        args_attn, graph_attn = Attention(n_in=self.channels[0]).create_graph(x)

        count = 0
        for i in range(2+self.n_blocks):
            args = self.add_variable_inputs(count, args_blk)
            x, = graph_blk.bind(args).call(x)
            count += 1

            if i != 1:
                args = self.add_variable_inputs(count, args_attn)
                x, = graph_attn.bind(args).call(x)
                count += 1

        for i in range(self.n_resolutions):
            if i > 0:
                for j in range(self.n_blocks):
                    x = self.up[i].block[j](x)
            if i != self.n_resolutions-1:
                x = self.up[i].upsample(x)

        x = self.gn_out(x)
        x = ops.swish_(x)
        x = self.conv_out(x)
        return x

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        weights = Conv2d.hf_mapping(variables.conv_in, hf_model.conv_in)
        weights.update(Block.hf_mapping(variables[0], hf_model.mid.block_1, config.ipus))
        weights.update(Attention.hf_mapping(variables[1], hf_model.mid.attn_1))
        weights.update(Block.hf_mapping(variables[2], hf_model.mid.block_2, config.ipus))
        for j in range(config.n_blocks):
            weights.update(Block.hf_mapping(
                variables[3+2*j], hf_model.up[3].block[j], config.ipus))
            weights.update(Attention.hf_mapping(variables[4+2*j], hf_model.up[3].attn[j]))

        for i in range(len(config.channels)-1):
            variables_ = variables.up[len(config.channels)-1-i]
            hf_model_ = hf_model.up[i]
            for j in range(config.n_blocks):
                weights.update(Block.hf_mapping(variables_.block[j], hf_model_.block[j]))
            if i != 0:
                weights.update(Conv2d.hf_mapping(
                    variables_.block[0].nin_shortcut, hf_model_.block[0].nin_shortcut))
            weights.update(Upsample.hf_mapping(
                variables.up[len(config.channels)-2-i].upsample, hf_model.up[i+1].upsample))

        weights.update(GroupNorm.hf_mapping(variables.gn_out, hf_model.norm_out))
        weights.update(Conv2d.hf_mapping(variables.conv_out, hf_model.conv_out))
        return weights


class GumbelVQ(addons.Module):
    def __init__(self, config, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.post_quant_conv = Linear(config.z_channel, bias=False, replica_grouping=replica_grouping)
        self.decoder = Decoder(config, replica_grouping=replica_grouping)

    def build(self, quant: popxl.Tensor) -> popxl.Tensor:
        quant = self.post_quant_conv(quant)
        quant = replicated_all_reduce_identical_grad_inputs(quant)
        self.bias = self.add_variable_input('bias', lambda: np.zeros(quant.shape[-1]), quant.dtype)
        quant = quant + self.bias
        b, n, _ = quant.shape
        quant = quant.transpose_((0, 2, 1)).reshape_((b, -1, int(math.sqrt(n)), int(math.sqrt(n))))
        dec = self.decoder(quant)
        return dec

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.bias.dtype
        weights = {
            variables.post_quant_conv.weight: shard(
                to_numpy(hf_model.post_quant_conv.weight.data.reshape(256, 256).T, dtype),
                config.ipus,
                axis=0),
            variables.bias: to_numpy(hf_model.post_quant_conv.bias.data, dtype),
        }
        weights.update(Decoder.hf_mapping(config, variables.decoder, hf_model.decoder))
        return weights


class VQGanGumbelVAE(addons.Module):
    def __init__(self, config, n_shards: int = 1, replica_grouping: Optional[ReplicaGrouping] = None, use_float16 = True):
        super().__init__()

        self.num_tokens = config.image_vocab_size
        self.z_channel = config.z_channel
        self.model = GumbelVQ(config=config, replica_grouping=replica_grouping)
        self.replica_grouping = replica_grouping
        self.use_float16 = use_float16
        self.n_shards = n_shards

    def build(self, img_seq: popxl.Tensor) -> popxl.Tensor:
        if self.use_float16:
            dtype = popxl.float16
        else:
            dtype = popxl.float32

        self.weight = self.add_variable_input(
                        "weight",
                        partial(np.random.normal, 0, 1/self.num_tokens,
                                (self.num_tokens, self.z_channel//self.n_shards)),
                        dtype=dtype,
                        replica_grouping=self.replica_grouping)

        num_classes = popxl.constant(self.num_tokens, popxl.int32)
        values = popxl.constant(np.array([0., 1.]), dtype)
        onehot_indices = ops.onehot(img_seq, num_classes=num_classes, values=values, axis=-1)
        z = onehot_indices @ self.weight
        img = self.model(z)
        # Clamp to [-1, 1] and then normalize to [0, 1]
        img_norm = (ops.relu_(img+1) - ops.relu_(img-1)) * 0.5

        return img_norm

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = variables.weight.dtype
        weights = {
            variables.weight: shard(to_numpy(hf_model.model.quantize.embed.weight.data, dtype),
                                    config.ipus, axis=-1)
        }
        weights.update(GumbelVQ.hf_mapping(config, variables.model, hf_model.model))

        return weights
