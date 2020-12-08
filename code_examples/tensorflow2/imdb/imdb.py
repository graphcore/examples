# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# All contributions by François Chollet:
# Copyright (c) 2015 - 2019, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015 - 2019, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017 - 2019, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2019, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information
#
# See https://github.com/keras-team/keras/blob/1a3ee8441933fc007be6b2beb47af67998d50737/examples/imdb_lstm.py
import tensorflow as tf

from tensorflow.python import ipu

from tensorflow.python.ipu.keras.layers import Embedding, LSTM
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizer_v2.adam import Adam

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")

max_features = 20000
minibatch_size = 32
gradient_accumulation_count = 16


# Define the dataset.
def get_dataset():
    (x_train, y_train), (_, _) = imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=80)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.repeat()
    ds = ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))
    ds = ds.batch(minibatch_size, drop_remainder=True)
    return ds


# Define the model.
def get_model():
    input_layer = Input(shape=(80), dtype=tf.int32, batch_size=minibatch_size)

    with ipu.keras.PipelineStage(0):
        x = Embedding(max_features, 128)(input_layer)

    with ipu.keras.PipelineStage(1):
        x = LSTM(128, dropout=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

    return ipu.keras.PipelineModel(input_layer,
                                   x,
                                   gradient_accumulation_count=gradient_accumulation_count)


def main():
    # Configure IPUs.
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    # Set up IPU strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        model = get_model()

        # The effective batch size is minibatch_size x gradient_accumulation_count,
        # so choose LR appropriately.
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005))
        model.fit(get_dataset(), steps_per_epoch=768, epochs=2)


if __name__ == '__main__':
    main()
