# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialised BERT Encoder layers."""
import numpy as np

from phased_execution.bert_layers import Attention, FeedForward, MaskLM
from phased_execution.layers import Add, Dense, Dropout, Embedding, Norm
from phased_execution.nn import Block

__all__ = [
    "EmbeddingSerialised",
    "BertEmbedding",
    "AttentionSplitIO",
    "AttentionSplitHidden",
    "FeedForwardSplitHidden",
    "FeedForwardSplitIO",
    "MaskLMSerialised",
]


def generate_simplified_periodic_pos_data(dtype, shape, scale=4):
    def value(x, y):
        return .02 / .707 * np.cos(2 * scale * np.pi * x * y / shape[1])

    X, Y = np.mgrid[:shape[0], :shape[1]]
    return np.vectorize(value)(X, Y).astype(dtype)


def detach(builder, input_x, pass_through_creation=1):
    return builder.customOp(
        opName="Detach",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[input_x],
        attributes={"pass_through_creation": pass_through_creation})[0]


def constant_tensor(builder, value, dtype=None, debug_name=""):
    value = np.array(value)
    if dtype is not None:
        value = value.astype(dtype)
    return builder.aiOnnx.constant(value, debug_name)


class EmbeddingSerialised(Block):
    def __init__(self,
                 scope,
                 input_dim: int,
                 output_dim: int,
                 num_splits: int,
                 dtype: str = "float16",
                 custom: bool = False,
                 detach: bool = False,
                 weight_transposed: bool = False,
                 skip_scopes=True,
                 **kwargs):
        """Turns non-negative integers (indexes/tokens) into dense vectors
        of fixed size.

        Args:
            input_dim (int): Size of the vocabulary, i.e. maximum integer index + 1.
            output_dim (int): Dimension of the dense embedding.
            dtype (str, optional): Data type of output embeddings. Defaults to 'float16'.

        Returns:
            str:  Output tensor of shape (x0, x1, ... xN-1, output_dim) where (x0, x1, .... xN-1) is shape of input tensor.
        """
        super().__init__(params=[], scope=scope, **kwargs)
        self.dtype = dtype
        self.popart_dtype = 'FLOAT' if dtype == np.float32 else 'FLOAT16'

        if input_dim % num_splits:
            raise ValueError('`input_dim` must be a multiple of `num_splits`.')
        if hasattr(scope, 'execution_phase'):
            raise ValueError(
                'Split embedding cannot have a single phased execution scope.')

        layers = []
        self.split_input_dim = input_dim // num_splits
        # If masks are pre-fetched in execution phase 0 & 1
        # embedding execution phase starts at phase = 2.
        scope_provider = kwargs['scope_provider']
        if skip_scopes:
            _ = scope_provider.get_scope(name=f'skip',
                                         execution_phase='next',
                                         skip_scope=True)
        for i in range(num_splits):
            scope = scope_provider.get_scope(
                name=f'split{i}', execution_phase='next')
            layers.append(
                Embedding(scope,
                          self.split_input_dim,
                          output_dim,
                          custom=custom,
                          detach=detach,
                          weight_transposed=weight_transposed,
                          dtype=dtype,
                          **kwargs))

        self.accum_scope = scope_provider.get_scope('Accum', 'prev')
        self.layers = layers
        self.custom = custom
        self.detach = detach
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim}
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in: str):
        def mask_input_indices(i):
            x_split = self.builder.aiOnnx.sub([
                x_in,
                constant_tensor(self.builder, i * self.split_input_dim,
                                np.uint32)
                ])
            mask = self.builder.aiOnnx.less([
                x_split,
                constant_tensor(self.builder, self.split_input_dim,
                                np.uint32)
            ])
            mask = detach(self.builder, mask)
            masked_indices = self.builder.aiOnnx.mul([x_split,
                                                     self.builder.aiOnnx.cast([mask], "UINT32")])
            return masked_indices, mask

        if self.scope_provider.phased_execution_type == "DUAL":
            x_even_sum = None
            x_odd_sum = None
        else:
            x_sum = None

        for i, layer in enumerate(self.layers):
            with self.scope_provider(self.builder, layer.scope):
                masked_indices, mask = mask_input_indices(i)

            x_out = layer(masked_indices)

            with self.scope_provider(self.builder, layer.scope):
                fp_mask = self.builder.aiOnnx.cast([mask], self.popart_dtype)
                fp_mask = self.builder.aiOnnx.unsqueeze([fp_mask], [1])
                x_out = self.builder.aiOnnx.mul([x_out, fp_mask])
                if self.scope_provider == "DUAL":
                    if i % 2:
                        if i == 1:
                            x_odd_sum = x_out
                        else:
                            x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                    else:
                        if i == 0:
                            x_even_sum = x_out
                        else:
                            x_even_sum = self.builder.aiOnnx.add(
                                [x_out, x_even_sum])
                else:
                    if i == 0:
                        x_sum = x_out
                    else:
                        x_sum = self.builder.aiOnnx.add([x_out, x_sum])

        if self.scope_provider == "DUAL":
            # final accumulation in the accum scope
            with self.scope_provider(self.builder, self.accum_scope):
                return self.builder.aiOnnx.add([x_odd_sum, x_even_sum])
        else:
            return x_sum



class BertEmbedding(Block):
    def __init__(self, vocab_size, hidden_size, sequence_length,
                 max_positional_length, num_vocab_splits, epsilon,
                 apply_dropout, dropout_prob, mode, dtype, detach,
                 weight_transposed, custom=True, **kwargs):
        scope_provider = kwargs['scope_provider']
        additional_scopes = [kwargs['builder'].outlineAttributes({'outline_scope': 'Embeddings'})]
        scope = scope_provider.get_scope('Embeddings', additional_scopes=additional_scopes)
        super().__init__(scope, **kwargs)
        if num_vocab_splits > 1:
            self.token_embedding = EmbeddingSerialised(
                scope_provider.get_scope('Token'),
                input_dim=vocab_size,
                output_dim=hidden_size,
                num_splits=num_vocab_splits,
                custom=custom,
                dtype=dtype,
                detach=detach,
                weight_transposed=weight_transposed,
                **kwargs)
        else:
            self.token_embedding = Embedding(
                scope_provider.get_scope('Token', execution_phase='next'),
                input_dim=vocab_size,
                output_dim=hidden_size,
                custom=custom,
                dtype=dtype,
                detach=detach,
                weight_transposed=weight_transposed,
                **kwargs)
        num_segments = 2
        self.segment_embedding = Embedding(
            scope_provider.get_scope(
                'Segment', execution_phase='next'), num_segments,
            hidden_size, dtype, **kwargs)

        self.position_embedding = Embedding(
            scope_provider.get_scope('Position', execution_phase='prev'),
            max_positional_length,
            hidden_size,
            dtype,
            **kwargs)

        self.add = Add(scope_provider.get_scope('Sum', execution_phase='prev'), **kwargs)
        self.norm = Norm(scope_provider.get_scope('Norm', execution_phase='prev'),
                         hidden_size, epsilon, dtype, **kwargs)
        self.apply_dropout = apply_dropout
        if apply_dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(
                    'Dropout', execution_phase='prev'), dropout_prob,
                **kwargs)
        self.total_execution_phases = self.total_phases()

    def forward(self, indices, positions, segments):
        # Size of act = [micro_batch_size * seq_len, hidden_size]
        x = self.add([
            self.token_embedding(indices),
            self.position_embedding(positions),
            self.segment_embedding(segments),
        ])
        x = self.norm(x)
        if self.apply_dropout:
            return self.dropout(x)
        else:
            return x


class AttentionSplitIO(Block):
    def __init__(self, name: str, num_splits, hidden_size, num_heads,
                 serialize_matmul, available_memory_proportion, epsilon, dropout,
                 dropout_prob, attn_dropout, attn_dropout_prob, micro_batch_size, sequence_length, dtype, task,
                 num_mask_tokens, use_default_mem_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        if hidden_size % num_splits:
            raise ValueError('Hidden size must be a multiple of num_splits.')
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        attention_splits = []
        self.split_size = hidden_size // num_splits
        self.name = name
        for i in range(num_splits):
            attention_splits.append(
                Attention(f"{name}Split{i}",
                          hidden_size // num_splits,
                          hidden_size,
                          num_heads,
                          serialize_matmul,
                          available_memory_proportion,
                          epsilon,
                          dropout,
                          dropout_prob,
                          attn_dropout,
                          attn_dropout_prob,
                          micro_batch_size,
                          sequence_length,
                          dtype,
                          task,
                          num_mask_tokens,
                          use_default_mem_proportion=use_default_mem_proportion,
                          residual=False))
        self.layers = attention_splits
        self.accum_scope = scope_provider.get_scope(
            f'{self.name}/AttnAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'{name}/AttnNorm', self.accum_scope.execution_phase),
            hidden_size, epsilon, dtype, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(f'{name}/AttnDropout',
                                         self.accum_scope.execution_phase), dropout_prob,
                **kwargs)
        else:
            self.dropout = lambda x: x

    def forward(self, x_in: str, masks: str):
        split_attention_out = []

        for i, attention_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_split = self.builder.aiOnnxOpset9.slice(
                    [x_in],
                    axes=[1],
                    starts=[i * self.split_size],
                    ends=[(i + 1) * self.split_size])
                if i > 1:
                    attention_split.mask = attention_split[i % 2].mask
                split_attention_out.append(attention_split(x_split, masks))

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.builder.aiOnnx.concat(split_attention_out, axis=1)
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class AttentionSplitHidden(Block):
    def __init__(self, name: str, num_splits, hidden_size, num_heads,
                 serialize_matmul, available_memory_proportion, epsilon, dropout,
                 dropout_prob, attn_dropout, attn_dropout_prob, micro_batch_size, sequence_length, dtype, task,
                 num_mask_tokens, use_default_mem_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        # AttentionSplitHidden splits the num_heads, keeping size_per_head same.
        # Since hidden_size = num_heads * size_per_head , num_heads and hiddden_size
        # should be multiple of num_splits.

        if hidden_size % num_splits:
            raise ValueError('Hidden size must be a multiple of num_splits.')

        if num_heads % num_splits:
            raise ValueError('Num heads must be a multiple of num_splits.')

        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        attention_splits = []
        self.split_size = hidden_size // num_splits
        self.name = name
        for i in range(num_splits):
            attention_splits.append(
                Attention(f"Split{i}",
                          hidden_size,
                          self.split_size,
                          num_heads // num_splits,
                          serialize_matmul,
                          available_memory_proportion,
                          epsilon,
                          dropout,
                          dropout_prob,
                          attn_dropout,
                          attn_dropout_prob,
                          micro_batch_size,
                          sequence_length,
                          dtype,
                          task,
                          num_mask_tokens,
                          residual=False,
                          use_default_mem_proportion=use_default_mem_proportion,
                          **kwargs))
        self.layers = attention_splits
        self.accum_scope = scope_provider.get_scope(f'AttnAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'AttnNorm', self.accum_scope.execution_phase),
            hidden_size, epsilon, dtype, **kwargs)
        if dropout:
            self.dropout = Dropout(scope_provider.get_scope(f'AttnDropout',
                                                            self.accum_scope.execution_phase),
                                   dropout_prob,
                                   dtype=dtype,
                                   **kwargs)
        else:
            self.dropout = lambda x: x

    def forward(self, x_in: str, masks: str):
        x_odd_sum = None
        x_even_sum = None

        for i, attention_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                if i > 1:
                    if self.scope_provider.phased_execution_type == 'DUAL':
                        attention_split.mask = self.layers[i % 2].mask
                    else:
                        attention_split.mask = self.layers[0].mask
                x_out = attention_split(x_in, masks)
                if i % 2:
                    if i == 1:
                        x_odd_sum = x_out
                    else:
                        x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                else:
                    if i == 0:
                        x_even_sum = x_out
                    else:
                        x_even_sum = self.builder.aiOnnx.add(
                            [x_out, x_even_sum])

        # final accumulation in the last layer's scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            x = self.builder.aiOnnx.add([x_odd_sum, x_even_sum])

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class FeedForwardSplitHidden(Block):
    def __init__(self, name, num_splits, input_size, ff_size, dropout,
                 dropout_prob, epsilon, use_default_memory_proportion,
                 available_memory_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        ffwd_splits = []
        self.split_size = ff_size // num_splits
        self.name = name
        for i in range(num_splits):
            ffwd_splits.append(
                FeedForward(f'Split{i}',
                            input_size,
                            self.split_size,
                            dropout,
                            dropout_prob,
                            epsilon,
                            residual=False,
                            increment_scope=True,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs))
        self.layers = ffwd_splits
        self.accum_scope = scope_provider.get_scope(f'FFAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'FFNorm', self.accum_scope.execution_phase), input_size,
            epsilon, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(
                    f'FFDropout', self.accum_scope.execution_phase),
                dropout_prob, **kwargs)
        else:
            self.dropout = lambda x: x
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in: str):
        x_odd_sum = None
        x_even_sum = None

        for i, ffwd_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_out = ffwd_split(x_in)
                if i % 2:
                    if i == 1:
                        x_odd_sum = x_out
                    else:
                        x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                else:
                    if i == 0:
                        x_even_sum = x_out
                    else:
                        x_even_sum = self.builder.aiOnnx.add(
                            [x_out, x_even_sum])

        # final accumulation in the last layer's scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            x = self.builder.aiOnnx.add([x_odd_sum, x_even_sum])

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class FeedForwardSplitIO(Block):
    def __init__(self, name, num_splits, input_size, ff_size, dropout,
                 dropout_prob, epsilon, use_default_memory_proportion,
                 available_memory_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        ffwd_splits = []
        self.split_size = input_size // num_splits
        self.name = name
        for i in range(num_splits):
            ffwd_splits.append(
                FeedForward(f'{name}/Split{i}',
                            self.split_size,
                            ff_size,
                            dropout,
                            dropout_prob,
                            epsilon,
                            residual=False,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs))
        self.layers = ffwd_splits
        self.accum_scope = scope_provider.get_scope(f'{name}/FFAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'{name}/FFNorm', self.accum_scope.execution_phase),
            input_size, epsilon, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(f'{name}/FFDropout',
                                         self.accum_scope.execution_phase), dropout_prob,
                **kwargs)
        else:
            self.dropout = lambda x: x
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in: str):
        split_ffwd_out = []

        for i, ffwd_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_split = self.builder.aiOnnxOpset9.slice(
                    [x_in],
                    axes=[1],
                    starts=[i * self.split_size],
                    ends=[(i + 1) * self.split_size])
                split_ffwd_out.append(ffwd_split(x_split))

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.builder.aiOnnx.concat(split_ffwd_out, axis=1)
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class MaskLMSerialised(Block):
    def __init__(self, num_splits, vocab_size, hidden_size, sequence_length,
                 micro_batch_size, num_mask_tokens, projection_weights, activation, no_cls_layer,
                 epsilon, projection_bias, **kwargs):
        scope_provider = kwargs['scope_provider']
        additional_scopes = [kwargs['builder'].outlineAttributes({'outline_scope': 'MLMSerialised'})]
        scope = scope_provider.get_scope('MLMSerialised', additional_scopes=additional_scopes)
        super().__init__(params=[],
                         scope=scope,
                         **kwargs)
        self.slice_scope = scope_provider.get_scope('Slice', 'next')
        self.micro_batch_size = micro_batch_size
        self.vocab_length = vocab_size
        self.hidden_size = hidden_size
        self.sequence_len = sequence_length
        self.num_mask_tokens = num_mask_tokens
        self.no_cls_layer = no_cls_layer
        self.projection_bias = projection_bias
        if not no_cls_layer:
            scope = scope_provider.get_scope("LMPrediction", self.slice_scope.execution_phase)
            self.pred_head_transform = Dense(scope,
                                             hidden_size,
                                             hidden_size,
                                             activation=activation,
                                             **kwargs)
            scope = scope_provider.get_scope('LMPrediction/Norm', self.slice_scope.execution_phase)
            self.norm = Norm(scope, hidden_size, epsilon, **kwargs)
        layers = []
        for i in range(num_splits):
            layers.append(
                MaskLM(f'Split{i}',
                       vocab_size // num_splits,
                       hidden_size,
                       sequence_length,
                       micro_batch_size,
                       num_mask_tokens,
                       projection_weights[i],
                       activation=None,
                       slice_input=False,
                       no_cls_layer=True,
                       projection_bias=projection_bias,
                       **kwargs))
        self.concat_scope = scope_provider.get_scope('Concat', 'next')
        self.layers = layers
        self.total_execution_phases = self.total_phases()

    def forward(self, x_in):
        with self.scope_provider(self.builder, self.slice_scope):
            x = self.builder.reshape_const(
                self.builder.aiOnnx, [x_in],
                [self.micro_batch_size, self.sequence_len, self.hidden_size])

            x = self.builder.aiOnnxOpset9.slice([x],
                                                axes=[1],
                                                starts=[0],
                                                ends=[self.num_mask_tokens])
            x = self.builder.reshape_const(self.builder.aiOnnx, [x],
                                           [self.micro_batch_size * self.num_mask_tokens, self.hidden_size])
            if not self.no_cls_layer:
                x = self.pred_head_transform(x)
                x = self.norm(x)
        projection_splits = []
        for layer in self.layers:
            projection_splits.append(layer(x))

        # Stack outputs in the concat scope
        with self.scope_provider(self.builder, self.concat_scope):
            x = self.builder.aiOnnx.concat(projection_splits, axis=2)

        return x
