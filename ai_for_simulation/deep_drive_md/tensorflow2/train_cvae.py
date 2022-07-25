# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from cvae.CVAE import run_cvae

import tensorflow as tf
from tensorflow.python import ipu


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=200, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs for training")
parser.add_argument("--img_size", type=int, default=22, help="size [N] of input image, where data sample is NxNx3")
parser.add_argument("--dataset_size", type=int, default=16384, help="number of data samples in generated data set")
parser.add_argument("--validation", dest="val", action="store_true", help="train with validation data")
parser.add_argument("--no-validation", dest="val", action="store_false", help="train without validation data")
parser.set_defaults(val=False)

args = parser.parse_args()

if __name__ == '__main__':

    img_size = args.img_size
    epochs = args.num_epochs
    num_samples = args.dataset_size
    bs = args.batch_size
    dataset = tf.random.uniform(
        shape=[num_samples, img_size, img_size, 3], minval=0, maxval=255, dtype=tf.dtypes.float32) / 255.0

    # Configure the IPU system
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()
    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        cvae = run_cvae(epochs=epochs, batch_size=bs, cm_data_input=dataset, validation=args.val)
