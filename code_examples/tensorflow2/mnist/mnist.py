# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf

from tensorflow import keras
from tensorflow.python import ipu

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")


# The input data and labels.
def create_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
    train_ds = train_ds.map(
        lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    return train_ds.repeat()


# Create the model using the IPU-specific Sequential class instead of the
# standard tf.keras.Sequential class
def create_model():
    model = ipu.keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    return model


def main():
    # Configure the IPU system
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    # Create an IPU distribution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        # Create an instance of the model.
        model = create_model()

        # Get the training dataset.
        ds = create_dataset()

        # Train the model.
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer = tf.keras.optimizers.SGD())
        model.fit(ds, steps_per_epoch=2000, epochs=4)

if __name__ == '__main__':
    main()
