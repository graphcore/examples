# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
from functools import reduce
from operator import mul

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
import tensorflow.compat.v1 as tf
"""Replaces parts of bert/modeling.py & pretraining.py, for multi-stage embeddings."""


class MultiStageEmbedding:
    """An embedding that can be split over multiple pipeline stages (/IPUs)."""

    def __init__(self, embedding_size, vocab_size, initializer_range, n_stages, matmul_serialize_factor, dtype):
        if vocab_size % n_stages != 0:
            raise ValueError(
                f"Vocab size ({vocab_size}) should be a multiple of n_stages ({n_stages})"
            )
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.n_stages = n_stages
        self.matmul_serialize_factor = matmul_serialize_factor
        self.dtype = dtype
        self.table_list = []

    @property
    def _stage_vocab_size(self):
        return self.vocab_size // self.n_stages

    def _get_table(self):
        # See bert/modeling.py
        return tf.get_variable(
            name="word_embeddings",
            shape=[self._stage_vocab_size, self.embedding_size],
            initializer=tf.truncated_normal_initializer(
                stddev=self.initializer_range),
            dtype=self.dtype,
        )

    @contextlib.contextmanager
    def _stage_scope(self, stage):
        assert 0 <= stage < self.n_stages
        # Don't create a named scope for n_stages == 1, allowing for name-compatibility
        # with original BERT checkpoints
        if self.n_stages == 1:
            yield
        else:
            with tf.variable_scope(f"s{stage}"):
                yield

    # Multi-stage embedding/projection API

    def lookup(self, stage, input_ids, partial_embeddings):
        """Performs part of an embedding lookup in the multi-stage embedding table.
        stage -- int -- this function should be called with stage {0 .. n_stages}
        input_ids -- tensor(N; int) -- must be a flat (1D) word ID tensor
        partial_embeddings -- tensor(N x E; dtype) or None -- the result of a previous lookup()
        returns -- tensor(N x E; dtype) -- accumulated embeddings (note that this only contains
                   correct embeddings for all input_ids after all stages have been looked up)
        """
        if len(input_ids.shape) != 1:
            ids_shape = input_ids.shape
            num_ids = reduce(mul, ids_shape, 1)
            input_ids = array_ops.reshape(input_ids, [num_ids])
        assert (
            len(input_ids.shape) == 1
        ), "input IDs should be flat for MultiStageEmbedding.lookup"

        with self._stage_scope(stage):
            table = self._get_table()
            self.table_list.append(table)
            V = self._stage_vocab_size
            offset = stage * V

            # To perform the lookup, we shift input_ids down, clip & mask
            mask = (offset <= input_ids) & (input_ids < offset + V)
            block_input_ids = tf.clip_by_value(input_ids - offset, 0, V - 1)
            block_embeddings = tf.cast(
                tf.expand_dims(mask, 1), table.dtype
            ) * self.gather(table, block_input_ids)

            # Optional partial_embeddings to accumulate (across stages)
            if partial_embeddings is not None:
                return partial_embeddings + block_embeddings
            return block_embeddings

    def projection(self, stage, inputs):
        """Performs part of an embedding projection from the (optionally shared) embedding table.
        stage -- int -- this function should be called with stage {0 .. n_stages}
        inputs -- tensor(* x E; dtype) -- activations to project up to the vocabulary size
        returns -- tensor(* x (V/n_stages); dtype) -- partial projections (for this stage
        """

        with self._stage_scope(stage):
            return self.matmul(inputs, self.table_list[stage], transpose_b=True)

    @staticmethod
    def merge_projection(projections):
        """Merge projections across stages, from projection().
        projections -- [n_stages x tensor(* x (V/n_stages); dtype)] -- list of projection() outputs
        returns -- tensor(* x V; dtype) -- complete projection
        """
        if len(projections) == 1:
            return projections[0]
        return tf.concat(projections, -1)

    # Overridable core op implementations (e.g. for IPU-specifics)

    @staticmethod
    def gather(table, input_ids):
        """Override-able gather op to use for lookup()."""
        return tf.gather(table, input_ids)

    @staticmethod
    def matmul(inputs, weights, transpose_b):
        """Override-able matmul op to use for projection()."""
        return tf.matmul(inputs, weights, transpose_b=transpose_b)


def staged_matmul_wrapper(embedding,
                          split_count, max_predictions_per_seq):
    stage_func_list = list()

    def get_stage_func(embedding, index, split_count):
        generated_function = None
        if index == 0:
            def matmul_stage(masked_tokens_tensor):

                with tf.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    # with tf.variable_scope("cls/predictions"):
                    output = embedding.projection(index, masked_tokens_tensor)
                    return {"prev_output": output}
            generated_function = matmul_stage
        elif index == split_count - 1:
            def matmul_stage(prev_output, masked_tokens_tensor):
                common_embedding_table = None
                with variable_scope.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    output = embedding.projection(index, masked_tokens_tensor)
                    output = embedding.merge_projection([prev_output, output])
                    return {"mlm_logits": output}
            generated_function = matmul_stage
        else:
            def matmul_stage(
                prev_output,
                masked_tokens_tensor
            ):
                with variable_scope.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    output = embedding.projection(index, masked_tokens_tensor)
                    output = embedding.merge_projection([prev_output, output])
                    return {"prev_output": output}
            generated_function = matmul_stage
        return generated_function

    for i in range(split_count):
        stage_func_list.append(get_stage_func(embedding=embedding,
                                              index=i,
                                              split_count=split_count))
    return stage_func_list


def staged_embedding_lookup_wrapper(embedding, split_count, micro_batch_size, seq_length):
    stage_func_list = list()

    def get_stage_func(embedding, index):
        generated_function = None
        if index == 0:
            def embedding_stage(input_ids):
                with tf.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    output = embedding.lookup(index, input_ids, None)
                    return {"prev_output": output}
            generated_function = embedding_stage
        elif index == split_count - 1:
            def embedding_stage(prev_output,
                                input_ids,
                                masked_lm_weights):
                with tf.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    output = embedding.lookup(index, input_ids, prev_output)
                    masked_lm_weights = tf.cast(
                        masked_lm_weights, dtype=embedding.dtype)
                    if index == split_count - 1:
                        output = tf.reshape(
                            output, [micro_batch_size, seq_length, -1])
                        return {
                            "word_embeddings": output,
                            "masked_lm_weights": masked_lm_weights}
            generated_function = embedding_stage
        else:
            def embedding_stage(
                    prev_output,
                    input_ids,
            ):
                with tf.variable_scope("bert/embeddings", reuse=tf.AUTO_REUSE):
                    output = embedding.lookup(index, input_ids, prev_output)
                    return {"prev_output": output}
            generated_function = embedding_stage
        return generated_function

    for i in range(split_count):
        stage_func_list.append(get_stage_func(embedding=embedding, index=i))
    return stage_func_list


def get_split_embedding_stages(embedding, split_count, bert_config, micro_batch_size, seq_length):
    with tf.variable_scope("bert", reuse=tf.AUTO_REUSE, use_resource=True):
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE, use_resource=True):
            embedding_stages_func = staged_embedding_lookup_wrapper(
                embedding=embedding, split_count=split_count, micro_batch_size=micro_batch_size, seq_length=seq_length)
            return embedding_stages_func


def get_split_matmul_stages(embedding, split_count, bert_config):
    matmul_stages_func = staged_matmul_wrapper(
        embedding=embedding, split_count=split_count, max_predictions_per_seq=bert_config.max_predictions_per_seq)
    return matmul_stages_func
