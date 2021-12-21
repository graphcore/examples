# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu


def setup_random_seed():
    seed = 1989
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ipu.utils.reset_ipu_seed(seed)


def test_embedding_layer():
    from fastspeech2 import FastSpeechEmbeddings
    from tests.test_utils import check_tensor
    from tests.tf2_fastspeech2 import TFFastSpeechEmbeddings, FastSpeech2Config

    setup_random_seed()
    config = FastSpeech2Config()
    seqlen = 128
    input_id = np.random.permutation(range(seqlen))
    input_id = np.expand_dims(input_id, axis=0)
    speaker_id = 0

    base_lr = 0.01
    optimizer1 = tf.keras.optimizers.SGD(base_lr)
    optimizer2 = tf.keras.optimizers.SGD(base_lr)

    emb1 = TFFastSpeechEmbeddings(config)
    with tf.GradientTape() as tape:
        output1 = emb1([input_id, speaker_id])
        loss1 = tf.reduce_mean(tf.math.abs(output1))

    emb2 = FastSpeechEmbeddings(config)
    _ = emb2(input_id)
    emb2.set_weights(emb1.get_weights())

    grad1 = tape.gradient(loss1, emb1.trainable_weights)
    optimizer1.apply_gradients(zip(grad1, emb1.trainable_weights))

    with tf.GradientTape() as tape2:
        output2 = emb2(input_id)
        loss2 = tf.reduce_mean(tf.math.abs(output2))

    grad2 = tape2.gradient(loss2, emb2.trainable_weights)
    optimizer2.apply_gradients(zip(grad2, emb1.trainable_weights))

    # Check the weights
    for w1, w2 in zip(emb1.weights, emb2.weights):
        check_tensor(w1.numpy(), w2.numpy())
    # Check the outputs
    check_tensor(output1.numpy(), output2.numpy())
    # Check the gradients
    for g1, g2 in zip(grad1, grad2):
        check_tensor(g1.values.numpy(), g2.values.numpy())
