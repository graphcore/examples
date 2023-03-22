# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf

from tensorflow.python import ipu

from ipu_tensorflow_addons.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam

if tf.__version__[0] != "2":
    raise ImportError("TensorFlow 2 is required for this example")


max_features = 20000
minibatch_size = 32
gradient_accumulation_steps_per_replica = 16


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
    return tf.keras.Sequential(
        [
            Embedding(max_features, 128),
            LSTM(128, dropout=0.2),
            Dense(1, activation="sigmoid"),
        ]
    )


def main():
    # Configure IPUs.
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    # Set up IPU strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        model = get_model()
        model.set_pipelining_options(gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica)
        model.set_pipeline_stage_assignment([0, 1, 1])

        # The effective batch size is minibatch_size x gradient_accumulation_steps_per_replica x num_replicas,
        # so choose LR appropriately.
        model.compile(steps_per_execution=384, loss="binary_crossentropy", optimizer=Adam(0.005))
        model.fit(get_dataset(), steps_per_epoch=768, epochs=2)


if __name__ == "__main__":
    main()
