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

import os
import sys
from typing import NamedTuple

import numpy as np
import pytest
import tensorflow as tf
import torch
from modeling import BertConfig as TFBertConfig
from modeling import BertModel as TFBertModel
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler, utils
from tests.torch_bert import BertConfig as TorchBertConfig
from tests.torch_bert import BertEmbeddings as TorchBertEmbeddings
from tests.utils import (check_tensors, check_tf_torch_model,
                         copy_torch_weights_to_tf,
                         run_fwd_model)

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
torch.manual_seed(seed)

TF_TO_TORCH = {
    "word_embeddings:0": "word_embeddings.weight",
    "position_embeddings:0": "position_embeddings.weight",
    "token_type_embeddings:0": "token_type_embeddings.weight",
    "GroupNorm/gamma:0": "LayerNorm.weight",
    "GroupNorm/beta:0": "LayerNorm.bias"
}


class TestConfig(NamedTuple):
    batch_size: int = 1
    sequence_length: int = 128
    vocab_size: int = 30400
    hidden_size: int = 128
    num_attention_heads: int = 2
    hidden_act: str = "gelu"
    max_position_embeddings: int = 512
    max_predictions_per_seq: int = 20
    hidden_dropout_prob: float = 0.0
    layer_norm_eps: float = 0.001
    type_vocab_size: int = 2
    data_type: str = "float"
    initializer_range: float = 1.0
    matmul_serialize_factor: int = 5

    @property
    def dtype(self):
        if self.data_type == "float":
            return tf.float32
        elif self.data_type == "float16":
            return tf.float16
        else:
            raise TypeError(f"{self.data_type} is not supported.")


test_config = TestConfig()


@pytest.mark.parametrize("config", [(test_config)])
@pytest.mark.parametrize("phase", ["fwd"])
def test_embedding(config, phase):
    # define input
    indices = np.random.randint(0, test_config.vocab_size,
                                (test_config.batch_size, test_config.sequence_length)).astype(
        np.int32)
    positions = np.reshape(np.arange(test_config.sequence_length), (test_config.batch_size, test_config.sequence_length)).astype(
        np.int32)
    segments = np.random.randint(0, 2,
                                 (test_config.batch_size, test_config.sequence_length)).astype(
        np.int32)
    inputs = [d for d in [indices, positions, segments]]

    # build model
    # PyTorch model
    torch_config = TorchBertConfig(
        vocab_size_or_config_json_file=test_config.vocab_size,
        hidden_size=test_config.hidden_size,
        hidden_act=test_config.hidden_act,
        num_attention_heads=test_config.num_attention_heads,
        hidden_dropout_prob=test_config.hidden_dropout_prob,
        max_position_embeddings=test_config.max_position_embeddings,
        type_vocab_size=test_config.type_vocab_size,
        update_embedding_dict=True,
        layer_norm_eps=test_config.layer_norm_eps
    )
    torch_model = TorchBertEmbeddings(torch_config)
    torch_model.eval()

    # TF model
    tf_config = TFBertConfig(
        vocab_size=test_config.vocab_size,
        hidden_size=test_config.hidden_size,
        hidden_act=test_config.hidden_act,
        num_attention_heads=test_config.num_attention_heads,
        max_position_embeddings=test_config.max_position_embeddings,
        max_predictions_per_seq=test_config.max_predictions_per_seq,
        hidden_dropout_prob=test_config.hidden_dropout_prob,
        type_vocab_size=test_config.type_vocab_size,
        initializer_range=test_config.initializer_range,
        dtype=test_config.dtype,
        matmul_serialize_factor=test_config.matmul_serialize_factor,
        static_mask=False
    )

    # farward check
    if phase == "fwd":
        torch_outputs = run_fwd_model(inputs, torch_model)

        with tf.Graph().as_default():
            tf_model = TFBertModel(tf_config, is_training=True)

            with ops.device('cpu'):
                input_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
                position_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
                segment_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
            cfg = utils.create_ipu_config()
            cfg = utils.auto_select_ipus(cfg, 1)
            utils.configure_ipu_system(cfg)
            utils.move_variable_initialization_to_cpu()
            with ops.device("/device:IPU:0"):
                opt = ipu_compiler.compile(tf_model.embeddings_layer, inputs=[
                    input_ids, position_ids, segment_ids])

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # copy pytorch weight to tf
                var_and_init = copy_torch_weights_to_tf(
                    torch_model, tf_model, TF_TO_TORCH, {}, sess)
                sess.run(var_and_init)
                # run tf feed feed farward
                tf_outputs = sess.run(
                    opt, {input_ids: indices, position_ids: positions, segment_ids: segments})
                # compare tf output with pytorch output
                check_tensors(tf_outputs, torch_outputs, margin=1.5e-8)

    # backward check
    elif phase == "bwd":
        l1_lambda = 0.1
        base_lr = 0.01
        optim = torch.optim.SGD(torch_model.parameters(),
                                base_lr,
                                weight_decay=0.0,
                                momentum=0.0)

        torch_output = torch_model(
            *[torch.from_numpy(t).long() for t in inputs])
        # pytorch backward
        torch_loss = l1_lambda * torch.norm(torch_output, 1)
        torch_loss.backward()   # calculate gradients
        optim.step()    # update gradients
        torch_outputs = [torch_output.detach().numpy()]

        # TF
        with tf.Graph().as_default():
            tf_model = TFBertModel(tf_config, is_training=True)
            with ops.device('cpu'):
                input_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
                position_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
                segment_ids = tf.placeholder(
                    shape=[test_config.batch_size, test_config.sequence_length], dtype=tf.int32)
            cfg = utils.create_ipu_config()
            cfg = utils.auto_select_ipus(cfg, 1)
            utils.configure_ipu_system(cfg)
            utils.move_variable_initialization_to_cpu()

            def embedding_graph(input_ids, position_ids, segment_ids):
                embedding_output = tf_model.embeddings_layer(
                    input_ids, position_ids, segment_ids)
                l1_loss = l1_lambda * tf.norm(embedding_output, 1)
                optimizer = tf.train.GradientDescentOptimizer(base_lr)
                train_step = optimizer.minimize(l1_loss)
                return embedding_output, l1_loss, train_step

            with ops.device("/device:IPU:0"):
                opt = ipu_compiler.compile(embedding_graph, inputs=[
                    input_ids, position_ids, segment_ids])

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                var_and_init = copy_torch_weights_to_tf(
                    torch_model, tf_model, TF_TO_TORCH, {}, sess)
                sess.run(var_and_init)
                tvars = sess.run({v.name: v for v in tf.trainable_variables()})
                print(tvars)
                tf_outputs, tf_loss = sess.run(
                    opt, {input_ids: indices, position_ids: positions, segment_ids: segments})
                # sess.run(opt, {input_ids: indices, position_ids: positions, segment_ids: segments})
                # Compare the farward output
                check_tf_torch_model(
                    sess, torch_model, TF_TO_TORCH, margin=5e-7)
            check_tensors(torch_outputs, tf_outputs, margin=5e-7)
    else:
        raise ValueError(
            f"`phase` only can be set to [`fwd`, `bwd`] which mean farward or backward respectively.")
