# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from data import CIFAR10_Data
from model import get_model
from utils import parse_params


if __name__ == '__main__':
    opts = parse_params(enable_multi_ipu=False)
    # Configure the IPU system
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    # Create an IPU distribution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()
    print("Loading the data...")
    data = CIFAR10_Data()
    ds_train, ds_test = data.get_train_dataset(opts.batch_size), data.get_test_dataset(opts.batch_size)

    with strategy.scope():
        # Create an instance of the model.
        print("Initialize the model")
        model = get_model(logits_only=False)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        print("Training...")
        training_steps = len(data.y_train) // opts.batch_size
        model.fit(ds_train, steps_per_epoch=training_steps, epochs=opts.epochs)
        print("Check the result")
        loss, accuracy = model.evaluate(ds_test)
        print("Validation accuracy: {}%".format(100.0 * accuracy))
        print("Validation loss: {}".format(loss))
