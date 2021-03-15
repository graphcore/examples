# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# Based on the Google colab shakespeare on TPU example, licenced under
# the Apache License, Version 2.0. Original URL:
# https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb

import numpy as np
import tensorflow as tf

from tensorflow.python import ipu
from tensorflow.python.ipu.keras.layers import Embedding, LSTM
from tensorflow.python.keras.layers import Dense, TimeDistributed
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

SHAKESPEARE_TXT = 'shakespeare.txt'

SEQ_LEN = 64
BATCH_SIZE = 24
REPEAT_COUNT = 1024 // BATCH_SIZE
EMBEDDING_DIM = 512


def transform(txt):
    return np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)


def get_dataset():
    with tf.io.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
        txt = f.read()

    source = tf.constant(transform(txt), dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices(source).batch(SEQ_LEN+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    BUFFER_SIZE = 10000
    ds = ds.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return ds.repeat()


def get_model():
    return ipu.keras.Sequential(
        [Embedding(256, 128),
         LSTM(EMBEDDING_DIM, return_sequences=True),
         LSTM(EMBEDDING_DIM, return_sequences=True),
         TimeDistributed(Dense(256, activation='softmax'))],
        gradient_accumulation_count=REPEAT_COUNT)


def main():
    # Configure IPUs.
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    # Set up IPU strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        model = get_model()

        model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
        model.fit(get_dataset(), steps_per_epoch=100, epochs=10)


if __name__ == '__main__':
    main()
