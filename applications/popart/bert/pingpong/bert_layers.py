# Copyright 2020 Graphcore Ltd.
"""BERT layers."""
import ctypes
import logging
import os
from functools import reduce
from typing import List

import numpy as np

import popart
from bert_model import ExecutionMode
from pingpong.layers import Add, Dense, Dropout, Embedding, Norm, Split
from pingpong.nn import Block, Parameter, Scope
from pingpong.scope_manager import Scope

__all__ = [
    "Attention", "FeedForward", "MaskLM", "NextSentencePred", "SquadProjection"
]


class Attention(Block):
    def __init__(self,
                 name: str,
                 input_size,
                 hidden_size,
                 num_heads,
                 serialize_matmul,
                 avail_mem_prop,
                 epsilon,
                 dropout,
                 dropout_prob,
                 attn_dropout,
                 attn_dropout_prob,
                 batch_size,
                 sequence_length,
                 dtype,
                 task,
                 num_mask_tokens,
                 residual=True,
                 prefetch_masks=True,
                 use_default_mem_proportion=True,
                 mask=None,
                 **kwargs):
        params = [
            Parameter(name='QKV',
                      shape=[input_size, 3 * hidden_size],
                      value=None),
            Parameter(name='Out', shape=[hidden_size, input_size], value=None)
        ]
        scope_provider = kwargs['scope_provider']
        super(Attention, self).__init__(params=params,
                                        scope=scope_provider.get_scope(
                                            name, 'next'),
                                        dtype=dtype,
                                        **kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.serialize_matmul = serialize_matmul
        self.available_memory_proportion = avail_mem_prop
        self.use_default_mem_proportion = use_default_mem_proportion
        self.batch_size = batch_size
        self.seq_len = sequence_length
        if hidden_size % num_heads != 0:
            raise ValueError('Hidden size must be a multiple of num_heads')
        self.qkv_length = hidden_size // num_heads
        self.dtype = dtype
        self.residual = residual
        self.task = task
        self.num_mask_tokens = num_mask_tokens
        self.mask = mask
        self.prefetch_masks = prefetch_masks
        if prefetch_masks:
            additional_scopes = [self.builder.recomputeOutput(popart.RecomputeType.Checkpoint),
                                 self.builder.outputTensorLocation(popart.TensorLocation(popart.TensorStorage.OnChip))]
            self.mask_execution_phase = scope_provider.get_scope('Mask', 'prev').execution_phase % 2
            self.mask_scope = scope_provider.get_scope('Mask',
                                                       self.mask_execution_phase,
                                                       additional_scopes=additional_scopes)
        else:
            self.mask_scope = scope_provider.get_scope('Mask', 'prev')

        if self.residual:
            self.norm = Norm(scope_provider.get_scope('Norm', 'prev'), hidden_size,
                             epsilon, dtype, **kwargs)
        if dropout:
            self.dropout = Dropout(scope_provider.get_scope('Dropout', 'prev'),
                                   dropout_prob, **kwargs)
        else:
            self.dropout = lambda x: x

        if attn_dropout:
            self.attn_dropout = Dropout(scope_provider.get_scope('AttnDropout', 'prev'),
                                        attn_dropout_prob, **kwargs)
        else:
            self.attn_dropout = lambda x: x

        self.total_execution_phases = self.total_phases()

    def attention_mask(self, masks):
        if self.prefetch_masks and (self.mask is not None):
            return self.mask

        with self.scope_provider(self.builder, self.mask_scope):
            all_indices_np = np.arange(self.seq_len, dtype=np.uint32)
            all_indices = self.builder.aiOnnx.constant(all_indices_np,
                                                       "mask_sequence")
            if self.task == "PRETRAINING":
                # Mask tokens mask
                indices_less_than_maskidx = self.builder.aiOnnx.less(
                    [all_indices, masks[0]])

                indices_greater_than_num_mask_token_np = np.greater_equal(
                    all_indices_np, self.num_mask_tokens).astype(np.bool)
                indices_greater_than_num_mask_token = self.builder.aiOnnx.constant(
                    indices_greater_than_num_mask_token_np)

                mask_tokens_mask = self.builder.aiOnnx.logical_or(
                    [indices_less_than_maskidx, indices_greater_than_num_mask_token])

                # Sequence mask
                sequence_mask = self.builder.aiOnnx.less(
                    [all_indices, masks[1]])

                final_mask = self.builder.aiOnnx.logical_and(
                    [mask_tokens_mask, sequence_mask])
            else:
                final_mask = self.builder.aiOnnx.less([all_indices, masks[0]])

            final_mask = self.builder.aiOnnx.cast(
                [final_mask],
                'FLOAT' if self.dtype == np.float32 else 'FLOAT16')
            final_mask = self.builder.aiOnnx.sub([
                final_mask,
                self.builder.aiOnnx.constant(np.array(1.0, self.dtype))
            ])
            final_mask = self.builder.aiOnnx.mul([
                final_mask,
                self.builder.aiOnnx.constant(np.array(1000.0, self.dtype))
            ])
            final_mask = self.builder.reshape_const(
                self.builder.aiOnnx, [final_mask],
                [self.batch_size, 1, 1, self.seq_len])

            # TODO: This shouldn't be needed. No Variables on this path.
            final_mask = self.builder.customOp(
                opName="Detach",
                opVersion=1,
                domain="ai.graphcore",
                inputs=[final_mask],
                attributes={"pass_through_creation": 1})[0]
            self.mask = final_mask
        return final_mask

    def dotproduct_attention(self, qkv, masks):
        split_qkv = self.builder.aiOnnx.split([qkv],
                                              num_outputs=3,
                                              axis=1,
                                              split=[self.hidden_size] * 3,
                                              debugPrefix="QKV_Split")

        def extract_heads(tensor, transpose=False):
            comb_shape = [
                self.batch_size, self.seq_len, self.num_heads, self.qkv_length
            ]
            tensor = self.builder.reshape_const(self.builder.aiOnnx, [tensor],
                                                comb_shape)
            perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
            return self.builder.aiOnnx.transpose([tensor], perm=perm)

        # q = [batch_size * seq_len, hidden_size]
        # kt = [hidden_size, batch_size * seq_len]
        # v = [batch_size * seq_len, hidden_size]
        q, kt, v = [extract_heads(t, i == 1) for i, t in enumerate(split_qkv)]

        # Attention calculation
        with self.builder.nameScope('Z'):
            scores = self.builder.aiOnnx.matmul([q, kt], "AttentionDotProduct")
            # TODO: Use serialise matmul
            if not self.use_default_mem_proportion:
                self.builder.setAvailableMemoryProportion(
                    scores, self.available_memory_proportion)

            scale = self.builder.aiOnnx.constant(
                np.array(1 / np.sqrt(self.qkv_length), self.dtype), "Scale")
            scores = self.builder.aiOnnx.mul([scores, scale])

            if masks:
                mask = self.attention_mask(masks)
                scores = self.builder.aiOnnx.add([scores, mask], "ApplyMask")

            scores = self.builder.aiOnnx.softmax([scores], axis=-1)
            scores = self.attn_dropout(scores)

            # x[batch_size, attention_heads, sequence_length, sequence_length] * v[batch_size, attention_heads, sequence_length, qkv_length]
            z = self.builder.aiOnnx.matmul([scores, v])
            if not self.use_default_mem_proportion:
                self.builder.setAvailableMemoryProportion(
                    z, self.available_memory_proportion)

            # [batch_size, attention_heads, sequence_length, qkv_length] -> [batch_size, sequence_length, attention_heads, qkv_length]
            z = self.builder.aiOnnx.transpose([z], perm=[0, 2, 1, 3])
            # [batch_size, sequence_length, attention_heads, qkv_length] -> [batch_size*sequence_length, attention_heads * qkv_length]
            z = self.builder.reshape_const(
                self.builder.aiOnnx, [z],
                [self.seq_len * self.batch_size, self.hidden_size])
        return z

    def forward(self, input_x: str, masks: str):
        # Transform input -> query, keys and value
        qkv_weight, projection_weight = [
            param.popart_tensor for param in self.params
        ]
        qkv = self.builder.aiOnnx.matmul([input_x, qkv_weight],
                                         'DenseTransform')
        if self.serialize_matmul:
            self.builder.setSerializeMatMul({qkv}, 'output_channels', 3, True)
        if not self.use_default_mem_proportion:
            self.builder.setAvailableMemoryProportion(
                qkv, self.available_memory_proportion)

        # Self-attention
        x = self.dotproduct_attention(qkv, masks)

        # Projection
        x = self.builder.aiOnnx.matmul([x, projection_weight], 'Projection')
        if not self.use_default_mem_proportion:
            self.builder.setAvailableMemoryProportion(
                x, self.available_memory_proportion)

        if not self.residual:
            return x

        # Residual
        x = self.dropout(x)
        x = self.builder.aiOnnx.add([input_x, x], 'Residual')
        x = self.norm(x)
        return x


class FeedForward(Block):
    def __init__(self,
                 name,
                 input_size,
                 ff_size,
                 dropout,
                 dropout_prob,
                 epsilon,
                 residual=True,
                 intermediate_act_func='gelu',
                 alpha=None,
                 increment_scope=True,
                 serialize_matmul=False,
                 use_default_memory_proportion=True,
                 available_memory_proportion=None,
                 **kwargs):
        scope_provider = kwargs['scope_provider']
        self.apply_dropout = dropout
        if increment_scope:
            scope = scope_provider.get_scope(name, 'next')
        else:
            scope = scope_provider.get_scope(name, 'prev')
        super(FeedForward, self).__init__(params=[], scope=scope, **kwargs)
        self.residual = residual

        if serialize_matmul:
            split = Split(dim='output_channels',
                          num_splits=ff_size // input_size)
        else:
            split = None
        self.dense1 = Dense(scope_provider.get_scope("1", 'prev'),
                            input_size,
                            ff_size,
                            split=split,
                            activation=intermediate_act_func,
                            alpha=alpha,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs)
        if serialize_matmul:
            split = Split(dim='reducing_dim', num_splits=ff_size // input_size)
        else:
            split = None
        self.dense2 = Dense(scope_provider.get_scope("2", "prev"),
                            ff_size,
                            input_size,
                            split=split,
                            activation=None,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs)
        if residual:
            if dropout:
                self.dropout = Dropout(scope_provider.get_scope("Dropout", "prev"),
                                       dropout_prob, **kwargs)
            self.norm = Norm(scope_provider.get_scope("Norm", "prev"), input_size,
                             epsilon, **kwargs)
        self.total_execution_phases = self.total_phases()

    def forward(self, input_x):
        x = self.dense1(input_x)
        x = self.dense2(x)
        if not self.residual:
            return x
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.builder.aiOnnx.add([input_x, x])
        x = self.norm(x)
        return x


class MaskLM(Block):
    def __init__(self,
                 name,
                 vocab_size,
                 hidden_size,
                 sequence_length,
                 batch_size,
                 num_mask_tokens,
                 projection_weight,
                 slice_input=True,
                 **kwargs):
        scope_provider = kwargs['scope_provider']
        super(MaskLM, self).__init__(params=[],
                                     scope=scope_provider.get_scope(name=f'MLM/{name}', execution_phase='next'),
                                     **kwargs)
        self.sequence_len = sequence_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_length = vocab_size
        self.num_mask_tokens = num_mask_tokens
        self.slice_input = slice_input
        self.dense = Dense(scope_provider.get_scope("Projection", self.scope.execution_phase),
                           hidden_size,
                           vocab_size,
                           split=None,
                           activation='relu',
                           params=[projection_weight, None],
                           **kwargs)
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in):
        if self.slice_input:
            x = self.builder.reshape_const(
                self.builder.aiOnnx, [x_in],
                [self.batch_size, self.sequence_len, self.hidden_size])

            x = self.builder.aiOnnxOpset9.slice([x],
                                                axes=[1],
                                                starts=[0],
                                                ends=[self.num_mask_tokens])
        else:
            x = x_in

        x = self.dense(x)
        x = self.builder.reshape_const(
            self.builder.aiOnnx, [x],
            [self.batch_size, self.num_mask_tokens, self.vocab_length])
        return x


class NextSentencePred(Block):
    def __init__(self, name, batch_size, sequence_length, hidden_size,
                 **kwargs):
        scope_provider = kwargs['scope_provider']
        scope = scope_provider.get_scope(name, execution_phase='next')
        params = []
        super().__init__(scope, params, **kwargs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.pooler = Dense(scope_provider.get_scope(
            "Pool", execution_phase=self.scope.execution_phase),
            input_dim=hidden_size,
            output_dim=hidden_size,
            split=None,
            activation='tanh',
            **kwargs)
        self.classifier = Dense(scope=scope_provider.get_scope(
            "Classifier", execution_phase=self.scope.execution_phase),
            input_dim=hidden_size,
            output_dim=2,
            split=None,
            **kwargs)
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in):
        x = self.builder.reshape_const(
            self.builder.aiOnnx, [x_in],
            [self.batch_size, self.sequence_length, self.hidden_size])

        x = self.builder.aiOnnxOpset9.slice([x],
                                            axes=[1],
                                            starts=[0],
                                            ends=[1])

        # This reshape is doing the job of a squeeze, but allows for in-place
        # operation.
        x = self.builder.reshape_const(self.builder.aiOnnx, [x],
                                       [self.batch_size, self.hidden_size])
        x = self.pooler(x)
        return self.classifier(x)


class SquadProjection(Block):
    def __init__(self, name, batch_size, sequence_length, hidden_size,
                 **kwargs):
        scope_provider = kwargs['scope_provider']
        scope = scope_provider.get_scope(name, execution_phase='next')
        params = []
        super().__init__(scope, params, **kwargs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.classifier = Dense(
            scope=scope_provider.get_scope(name, execution_phase=self.scope.execution_phase),
            input_dim=hidden_size,
            output_dim=2,
            split=None,
            **kwargs)
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in):
        x = self.classifier(x_in)

        start_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[0], ends=[1], debugPrefix='slice_ans_start')
        end_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[1], ends=[2], debugPrefix='slice_ans_end')

        start_logits = self.builder.reshape_const(
            self.builder.aiOnnx, [start_logits],
            [self.batch_size, self.sequence_length],
            debugPrefix="answer_start")
        end_logits = self.builder.reshape_const(
            self.builder.aiOnnx, [end_logits],
            [self.batch_size, self.sequence_length],
            debugPrefix="answer_end")

        return start_logits, end_logits
