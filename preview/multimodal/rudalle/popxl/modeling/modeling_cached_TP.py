# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import popxl_addons as addons
import torch
from popxl_addons import NamedTensors
from popxl_addons.layers import Embedding, LayerNorm, Linear
from popxl_addons.ops.replicated_all_reduce_TP import \
    replicated_all_reduce_identical_grad_inputs
from rudalle.dalle.image_attention import (get_col_mask, get_conv_mask,
                                           get_row_mask)

import popxl
from popxl import ReplicaGrouping, ops
from popxl.tensor import HostTensor
from popxl.utils import to_numpy
from utils import repeat, shard


def generate_positions(config):
    pos = np.repeat(np.arange(0, config.text_seq_len).reshape(1, -1),
                    config.micro_batch_size, axis=0).flatten()
    return pos


class EmbeddingsTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.replica_grouping = replica_grouping
        assert self.config.ipus == self.replica_grouping.num_groups
        # Reserve unique padding tokens for each position (text seq len)
        self.num_text_tokens = config.vocab_size + config.text_seq_len
        config.total_vocab_size = self.num_text_tokens + config.image_vocab_size
        config.n_positions = config.text_seq_len + 1 + config.image_tokens_per_dim**2
        self.text_image_emb = Embedding(config.dtype, config.total_vocab_size, config.n_embd,
                                        replica_grouping=self.replica_grouping)
        self.image_tokens_per_dim = int(config.image_seq_len**0.5)
        self.total_pos_len = config.text_seq_len + 1 + self.image_tokens_per_dim**2
        self.pos_emb = Embedding(self.config.dtype, self.total_pos_len, self.config.n_embd,
                                 replica_grouping=self.replica_grouping)

    def build(self,
              input_ids: popxl.Tensor,
              position_ids: popxl.Tensor = None)  -> List[popxl.Tensor]:
        if (len(input_ids) > 1):  # stage 0, the input is text tokens
            text_pos = self.pos_emb(position_ids)
            embeddings = self.text_image_emb(input_ids) + text_pos
        else:  # stage 1, the input is image token
            input_ids = input_ids + self.num_text_tokens
            image_embeddings = self.text_image_emb(input_ids)
            image_pos = self.pos_emb(position_ids)
            embeddings = image_embeddings + image_pos

        x = replicated_all_reduce_identical_grad_inputs(embeddings)
        return x

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.ipus

        text_image_emb = torch.cat((hf_model.text_embeddings.weight.data, hf_model.image_embeddings.weight.data))
        image_pos_emb = torch.zeros((config.image_seq_len, config.n_embd), requires_grad=False)
        for i in range(config.image_tokens_per_dim):
            for j in range(config.image_tokens_per_dim):
                image_pos_emb[i*config.image_tokens_per_dim+j] = hf_model.image_row_embeddings(torch.tensor(i)) + \
                     hf_model.image_col_embeddings(torch.tensor(j))
        pos_emb = torch.cat((hf_model.text_pos_embeddings.weight.data, image_pos_emb), 0)

        word_shard_size, pos_shard_size = EmbeddingsTP.get_vocab_shard_sizes(config)
        word_pad = word_shard_size * n_shards - config.total_vocab_size
        pos_pad = pos_shard_size * n_shards - config.n_positions

        pad = lambda x, n_pad: np.pad(x, ((0, n_pad), (0, 0)), 'constant')  # Pad only first axis in one direction

        return {
            variables.text_image_emb.weight: shard(pad(to_numpy(text_image_emb, dtype), word_pad), n_shards, axis=0),
            variables.pos_emb.weight: shard(pad(to_numpy(pos_emb, dtype), pos_pad), n_shards, axis=0),
        }

    @staticmethod
    def get_offsets(config) -> (np.ndarray, np.ndarray):
        n_shards = config.ipus

        word_offsets = Embedding.get_offsets(config.total_vocab_size, n_shards)
        pos_offsets = Embedding.get_offsets(config.n_positions, n_shards)
        return word_offsets, pos_offsets

    @staticmethod
    def get_vocab_shard_sizes(config) -> (int, int):
        n_shards = config.ipus

        word_shard_size = Embedding.get_vocab_shard_size(config.total_vocab_size, n_shards)
        pos_shard_size = Embedding.get_vocab_shard_size(config.n_positions, n_shards)
        return word_shard_size, pos_shard_size

    @staticmethod
    def offset_inputs(config, words: Optional[HostTensor] = None, axis=0):
        n_shards = config.ipus
        positions = generate_positions(config).flatten()
        word_offsets, pos_offsets = EmbeddingsTP.get_offsets(config)

        repeat_ = lambda x: repeat(x, n_shards, axis)

        def bc_shape(t):
            # Shape for broadcasting. `slice(None, None)` represents all like `array[:]`
            shape = [np.newaxis] * len(t.shape)
            shape.insert(axis, slice(None, None))
            return tuple(shape)

        pos_offsetted = repeat_(positions) - pos_offsets[bc_shape(positions)]

        if words is not None:
            words_offsetted = repeat_(words) - word_offsets[bc_shape(words)]
            return words_offsetted, pos_offsetted
        else:
            return pos_offsetted


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape_((micro_batch_size, sequence_length, heads, head_hidden_size))


def transpose_for_scores(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
    assert len(x.shape) == 4
    perm = (0, 2, 1, 3) if not is_key else (0, 2, 3, 1)
    x = x.transpose_(perm)
    return x


class AttentionHeads(addons.Module):
    def __init__(self, config, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()

        self.config = config
        self.replica_grouping = replica_grouping

        if self.replica_grouping:
            self.n_heads_groups = self.replica_grouping.num_groups
        else:
            self.n_heads_groups = 1

        assert self.config.n_head % self.n_heads_groups == 0

        self.embed_dim = self.config.n_embd
        self.num_heads = self.config.n_head // self.n_heads_groups
        self.scale_attn_weights = config.scale_attn_weights

        self.c_attn = Linear(3 * self.embed_dim // self.n_heads_groups, replica_grouping=replica_grouping)

    def transform_heads(self, x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
        return transpose_for_scores(reshape_for_scores(x, x.shape[0]//self.config.micro_batch_size, self.num_heads), is_key)

    def build(self, x: popxl.Tensor,
              past_k: popxl.Tensor = None,
              past_v: popxl.Tensor = None,
              mask: popxl.Tensor = None,
              onehot_index: popxl.Tensor = None):
        qkv_act = self.c_attn(x)
        query, key, value = ops.split(qkv_act, 3, axis=-1)

        query = self.transform_heads(query, False)
        key = self.transform_heads(key, is_key=True)
        value = self.transform_heads(value, False)

        if past_k and past_v:
            key = key @ onehot_index
            value = onehot_index.transpose_() @ value
            past_k = past_k + key
            past_v = past_v + value

        if self.scale_attn_weights:
            query /= math.sqrt(self.config.n_embd//self.config.n_head)

        if not past_k:
            past_k_ = key
            past_v_ = value
        else:
            past_k_ = past_k
            past_v_ = past_v
        attn_weights = query @ past_k_

        if mask:
            causal_mask = mask.reshape_((1, -1))
        else:
            query_length, key_length = attn_weights.shape[-2], attn_weights.shape[-1]
            causal_mask = self.add_variable_input("mask", partial(np.tril, np.ones((query_length, key_length))),
                                                  attn_weights.dtype)  # (128, 128)

        attn_weights = attn_weights + 10000.0 * (causal_mask - 1)
        attn_scores = ops.softmax(attn_weights, axis=-1)
        attn_output = attn_scores @ past_v_

        x_part_shape = list(x.shape)
        x_part_shape[-1] = x_part_shape[-1] // self.n_heads_groups
        attn_output = attn_output.transpose_((0, 2, 1, 3)).reshape_(x_part_shape)
        return attn_output, past_k_, past_v_


class AttentionTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()

        self.config = config
        self.replica_grouping = replica_grouping
        assert self.config.ipus == self.replica_grouping.num_groups

        # Sharded across devices
        self.heads = AttentionHeads(config=config, replica_grouping=replica_grouping)

        # Sharded across devices (bias applied separately)
        self.c_proj = Linear(self.config.n_embd, bias=False, replica_grouping=replica_grouping)

    def build(self,
              x: popxl.Tensor,
              layer_past_k: popxl.Tensor = None,
              layer_past_v: popxl.Tensor = None,
              mask: popxl.Tensor = None,
              onehot_index: popxl.Tensor = None
              ) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        if layer_past_k:
            z, k, v = self.heads(x, past_k=layer_past_k, past_v=layer_past_v, mask=mask, onehot_index=onehot_index)
        else:
            z, k, v = self.heads(x, mask=mask)

        z = self.c_proj(z)
        z = replicated_all_reduce_identical_grad_inputs(z)

        self.bias = self.add_variable_input('bias', lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        return z, k, v

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.ipus
        hf_query_w, hf_key_w, hf_value_w = np.split(hf_model.query_key_value.weight.data.T.numpy(), 3, axis=-1)
        hf_query_b, hf_key_b, hf_value_b = np.split(hf_model.query_key_value.bias.data.numpy(), 3, axis=-1)

        query_w = np.split(to_numpy(hf_query_w, dtype), n_shards, axis=-1)
        key_w = np.split(to_numpy(hf_key_w, dtype), n_shards, axis=-1)
        value_w = np.split(to_numpy(hf_value_w, dtype), n_shards, axis=-1)
        query_b = np.split(to_numpy(hf_query_b, dtype), n_shards, axis=-1)
        key_b = np.split(to_numpy(hf_key_b, dtype), n_shards, axis=-1)
        value_b = np.split(to_numpy(hf_value_b, dtype), n_shards, axis=-1)
        c_proj_w = to_numpy(hf_model.dense.weight.data.T.numpy(), dtype)
        c_proj_b = to_numpy(hf_model.dense.bias.data, dtype)

        return {
            variables.heads.c_attn.weight: np.ascontiguousarray(np.concatenate(
                [np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...]
                 for i in range(n_shards)]
            )),
            variables.heads.c_attn.bias: np.ascontiguousarray(np.concatenate(
                [np.concatenate([query_b[i], key_b[i], value_b[i]], axis=-1)[np.newaxis, ...]
                 for i in range(n_shards)]
            )),
            variables.c_proj.weight: shard(c_proj_w, n_shards, axis=0),
            variables.bias: c_proj_b,
        }


def gelu(x: popxl.Tensor) -> popxl.Tensor:
    # The self-defined gelu is closer to OpenAI's gelu than popxl.ops.gelu when using fp16
    return 0.5 * x * (1.0 + ops.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


class MLPTP(addons.Module):
    def __init__(self, intermediate_size, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.intermediate_size = 4 * config.n_embd if intermediate_size is None else intermediate_size

        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping
        assert self.n_shards == self.replica_grouping.num_groups

        assert self.intermediate_size % self.n_shards == 0

        self.c_fc = Linear(self.intermediate_size // self.n_shards, replica_grouping=self.replica_grouping)

        # Sharded across devices (bias applied separately)
        self.c_proj = Linear(config.n_embd, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """Identical input x and identical output across shards."""

        y = self.c_fc(x)
        y = gelu(y)
        y = self.c_proj(y)

        y = replicated_all_reduce_identical_grad_inputs(y)

        # Identical computation
        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input('bias', lambda: np.zeros(y.shape[-1]), y.dtype)
        y = y + self.bias

        return y

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.ipus

        return {
            variables.c_fc.weight: shard(to_numpy(hf_model.dense_h_to_4h.weight.data.T, dtype), n_shards, axis=-1),
            variables.c_fc.bias: shard(to_numpy(hf_model.dense_h_to_4h.bias.data, dtype), n_shards, axis=-1),
            variables.c_proj.weight: shard(to_numpy(hf_model.dense_4h_to_h.weight.data.T, dtype), n_shards, axis=0),
            variables.bias: to_numpy(hf_model.dense_4h_to_h.bias.data, dtype)
        }


class BlockTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config

        hidden_size = self.config.n_embd
        inner_dim = 4 * hidden_size

        self.ln_1 = LayerNorm()
        self.attn = AttentionTP(self.config, replica_grouping)
        self.ln_2 = LayerNorm()
        self.ln_3 = LayerNorm()
        self.mlp = MLPTP(inner_dim, self.config, replica_grouping)
        self.ln_4 = LayerNorm()

    def build(self, x: popxl.Tensor,
              layer_past_k: popxl.Tensor = None,
              layer_past_v: popxl.Tensor = None,
              mask: popxl.Tensor = None,
              onehot_index: popxl.Tensor = None):
        """Identical inputs and identical outputs."""

        a, present_k, present_v = self.attn(self.ln_1(x), layer_past_k=layer_past_k, layer_past_v=layer_past_v, mask=mask, onehot_index=onehot_index)
        a = self.ln_2(a)
        x = x + a
        m = self.mlp(self.ln_3(x))
        m = self.ln_4(m)
        x = x + m
        return x, present_k, present_v

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype

        weights = {
            variables.ln_1.weight: to_numpy(hf_model.input_layernorm.weight.data, dtype),
            variables.ln_1.bias: to_numpy(hf_model.input_layernorm.bias.data, dtype),
            variables.ln_2.weight: to_numpy(hf_model.before_first_addition_layernorm.weight.data, dtype),
            variables.ln_2.bias: to_numpy(hf_model.before_first_addition_layernorm.bias.data, dtype),
            variables.ln_3.weight: to_numpy(hf_model.post_attention_layernorm.weight.data, dtype),
            variables.ln_3.bias: to_numpy(hf_model.post_attention_layernorm.bias.data, dtype),
            variables.ln_4.weight: to_numpy(hf_model.before_second_addition_layernorm.weight.data, dtype),
            variables.ln_4.bias: to_numpy(hf_model.before_second_addition_layernorm.bias.data, dtype),
        }
        weights.update(AttentionTP.hf_mapping(config, variables.attn, hf_model.attention))
        weights.update(MLPTP.hf_mapping(config, variables.mlp, hf_model.mlp))

        return weights


class DecoderTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping
        assert self.n_shards == self.replica_grouping.num_groups

    def build(self, x: popxl.Tensor,
              past_k: popxl.Tensor = None,
              past_v: popxl.Tensor = None,
              col_mask: popxl.Tensor = None,
              row_mask: popxl.Tensor = None,
              conv_mask: popxl.Tensor = None,
              update_index: popxl.Tensor = None,
              ):
        """Identical input x and identical output across shards."""

        if past_k:
            layers_past_k = ops.split(past_k, self.config.layers, axis=0)
            layers_past_v = ops.split(past_v, self.config.layers, axis=0)
            if x.dtype == popxl.float16:
                _values = popxl.constant(np.array([0., 1.]).astype(np.float16))
            else:
                _values = popxl.constant(np.array([0., 1.]).astype(np.float32))
            _num_classes = popxl.constant(self.config.total_seq_len-1, popxl.int32)
            onehot_index = ops.onehot(update_index, num_classes=_num_classes, values=_values, axis=0).reshape_((1, -1))
            args, graph = BlockTP(self.config, self.replica_grouping).create_graph(x,
                                                                                   layer_past_k=layers_past_k[0],
                                                                                   layer_past_v=layers_past_v[0],
                                                                                   mask=col_mask,
                                                                                   onehot_index=onehot_index,)
        else:
            args, graph = BlockTP(self.config, self.replica_grouping).create_graph(x)

        presents_k = []
        presents_v = []

        for i in range(self.config.layers):
            args_nt = self.add_variable_inputs(i, args)
            if i % 4 == 1:
                mask = col_mask
            elif i != self.config.layers - 1:
                mask = row_mask
            else:
                mask = conv_mask

            if past_k:
                x, present_k, present_v = graph.bind(args_nt).call(x, layers_past_k[i], layers_past_v[i], mask, onehot_index)
            else:
                x, present_k, present_v = graph.bind(args_nt).call(x)
            presents_k.append(present_k)
            presents_v.append(present_v)

        presents_k = ops.concat_(presents_k, axis=0)
        presents_v = ops.concat_(presents_v, axis=0)
        return x, presents_k, presents_v


class ModelTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping
        assert self.n_shards == self.replica_grouping.num_groups

        self.embeddings = EmbeddingsTP(self.config, self.replica_grouping)
        self.decoder = DecoderTP(self.config, self.replica_grouping)
        # Final layer norm before output.
        self.final_layernorm = LayerNorm()

    def build(self, input_ids: popxl.Tensor, position_ids=None,
              past_k: popxl.Tensor = None,
              past_v: popxl.Tensor = None,
              col_mask: popxl.Tensor = None,
              row_mask: popxl.Tensor = None,
              conv_mask: popxl.Tensor = None,
              update_index: popxl.Tensor = None,
              ):
        x = self.embeddings(input_ids, position_ids)

        x, past_k, past_v = self.decoder(x, past_k, past_v, col_mask, row_mask, conv_mask, update_index)
        y = self.final_layernorm(x)
        return y, past_k, past_v

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        weights = {}
        weights.update(EmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.layers):
            weights.update(BlockTP.hf_mapping(config, variables.decoder[l], hf_model.transformer.layers[l]))

        weights.update({variables.final_layernorm.weight: to_numpy(hf_model.transformer.final_layernorm.weight.data, config.dtype)})
        weights.update({variables.final_layernorm.bias: to_numpy(hf_model.transformer.final_layernorm.bias.data, config.dtype)})

        return weights


class Mask(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping = None):
        super().__init__()
        self.config = config
        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping

    def build(self, indices) -> popxl.Tensor:
        self.mask = self.add_variable_input(
            "weight",
            lambda: np.zeros((self.config.image_seq_len, (self.config.total_seq_len)//self.n_shards), np.float16),
            self.config.dtype,
            replica_grouping=self.replica_grouping,
        )

        return ops.gather(self.mask, indices, axis=0, zero_OOR=True)


def gumbel_noise(t, seed):
    noise = ops.random_uniform(seed, t.shape, dtype=popxl.float32)
    return -ops.log(-ops.log(noise+1e-20) + 1e-20)


def top_k_top_p_sampling(logits, seed, top_k = 2048, top_p = 0.99):
    # top k filtering
    num_logits = logits.shape[-1]
    k = min(max(top_k, 1), num_logits)
    val, ind = ops.topk(logits, k, axis=1, largest=True, sorted=True)

    # top p filtering
    if top_p < 1.0:
        val = ops.cast(val, popxl.float32)
        p = ops.softmax(val, axis=-1)
        cumulative_probs = ops.cumsum(p, dim=-1)  # ops.cumsum only support fp32
        # filtering logit according to cumulative probs
        const = popxl.constant(np.array(len(val)*[top_p]), dtype=val.dtype)
        top_p_mask = ops.cast(ops.greater(cumulative_probs, const), val.dtype)  # bool to float
        val = val + top_p_mask * -10000.0

    # gumbel sampling
    p = ops.softmax(val, axis=-1)
    t = ops.log(p+1e-20) + gumbel_noise(p, seed)
    max_ind = ops.argmax(t, dim = -1)
    return ops.dynamic_slice(ind, max_ind, axes=[1], sizes=[1], no_overlap=False).reshape_((-1,))


class LMModelTP(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping
        assert self.n_shards == self.replica_grouping.num_groups

        self.col_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.row_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.conv_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.transformer = ModelTP(self.config, replica_grouping=self.replica_grouping)

        word_shard_size = config.image_vocab_size // self.n_shards
        self.ln_f = LayerNorm()
        self.to_logits = Linear(word_shard_size, replica_grouping=self.replica_grouping)

    def build(self, input_ids: popxl.Tensor, position_ids=None,
              past_k: popxl.Tensor = None,
              past_v: popxl.Tensor = None,
              seed: popxl.Tensor = None,
              word_offset: popxl.Tensor = None,
              ):

        # Prevent mask parameters that will be used in LMModelTP2 from being pruning
        index = popxl.constant(0)
        self.col_mask(index)
        self.row_mask(index)
        self.conv_mask(index)

        hidden_states, presents_k, presents_v = self.transformer(input_ids, position_ids, past_k, past_v)
        logits = self.to_logits(self.ln_f(hidden_states))
        next_token_logits = ops.slice_(logits, start=-1, axis=0)
        next_token_logits = ops.collectives.replicated_all_gather(next_token_logits).reshape_((-1, self.config.image_vocab_size))
        next_token = top_k_top_p_sampling(next_token_logits, seed, self.config.top_k, self.config.top_p) - word_offset

        return next_token, presents_k, presents_v

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.ipus

        assert config.image_vocab_size % n_shards == 0

        weights = {
            variables.ln_f.weight: to_numpy(hf_model.to_logits[0].weight.data, dtype),
            variables.ln_f.bias: to_numpy(hf_model.to_logits[0].bias.data, dtype),
            variables.to_logits.weight: shard(to_numpy(hf_model.to_logits[1].weight.data[-config.image_vocab_size:,:].T, dtype), n_shards, axis=-1),
            variables.to_logits.bias: shard(to_numpy(hf_model.to_logits[1].bias.data[-config.image_vocab_size:], dtype), n_shards, axis=-1),
        }

        weights.update({variables.col_mask.weight: shard(to_numpy(get_col_mask(128)[config.text_seq_len-1:-1], dtype), n_shards, axis=1)})
        weights.update({variables.row_mask.weight: shard(to_numpy(get_row_mask(128)[config.text_seq_len-1:-1], dtype), n_shards, axis=1)})
        weights.update({variables.conv_mask.weight: shard(to_numpy(get_conv_mask(128)[config.text_seq_len-1:-1], dtype), n_shards, axis=1)})

        weights.update(ModelTP.hf_mapping(config, variables.transformer, hf_model))
        return weights


class LMModelTP2(addons.Module):
    def __init__(self, config, replica_grouping: ReplicaGrouping):
        super().__init__()
        self.config = config
        self.n_shards = self.config.ipus
        self.replica_grouping = replica_grouping
        assert self.n_shards == self.replica_grouping.num_groups

        self.col_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.row_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.conv_mask = Mask(self.config, replica_grouping=self.replica_grouping)
        self.transformer = ModelTP(self.config, replica_grouping=self.replica_grouping)

        word_shard_size = config.image_vocab_size // self.n_shards
        self.ln_f = LayerNorm()
        self.to_logits = Linear(word_shard_size, replica_grouping=self.replica_grouping)

    def build(self, inputs: List[popxl.Tensor]):
        input_token, position_ids, past_k, past_v, update_index, word_offset, seed, output = inputs
        increment = popxl.constant(1)
        update_index = ops.add(update_index, increment)
        position_ids = ops.add(position_ids, increment)

        col_mask = self.col_mask(update_index-self.config.text_seq_len+1)
        row_mask = self.row_mask(update_index-self.config.text_seq_len+1)
        conv_mask = self.conv_mask(update_index-self.config.text_seq_len+1)
        col_mask = ops.collectives.replicated_all_gather(col_mask).reshape_((-1,))[:-1]
        row_mask = ops.collectives.replicated_all_gather(row_mask).reshape_((-1,))[:-1]
        conv_mask = ops.collectives.replicated_all_gather(conv_mask).reshape_((-1,))[:-1]

        hidden_states, presents_k, presents_v = self.transformer(input_token, position_ids, past_k, past_v, col_mask, row_mask, conv_mask, update_index)
        logits = self.to_logits(self.ln_f(hidden_states))
        next_token_logits = ops.collectives.replicated_all_gather(logits).reshape_((-1, self.config.image_vocab_size))

        seed, _ = ops.split_random_seed(seed)
        next_token = top_k_top_p_sampling(next_token_logits, seed, self.config.top_k, self.config.top_p) - word_offset
        output = ops.dynamic_update(output, update_index-self.config.text_seq_len+1, next_token, axes=[0], sizes=[1], no_overlap=False)
        return next_token, position_ids, presents_k, presents_v, update_index, word_offset, seed, output


    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model) -> Dict[popxl.Tensor, np.ndarray]:
        # LMModelTP2 has the same parameters to LMModelTP
        return LMModelTP.hf_mapping(config, variables, hf_model)
