# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import sys
import tensorflow as tf
from tensorflow.python import keras
import unittest


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from outfeed_optimizer import OutfeedOptimizer


@unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
class TestOutfeedOptimizer(unittest.TestCase):


    def test_no_model_supplied(self):
        with self.assertRaisesRegex(ValueError,
                                    "The model must be specified if a Keras"):
            OutfeedOptimizer(tf.keras.optimizers.SGD(), None)


    def test_keras_v1_optimizer_supplied(self):
        with self.assertRaisesRegex(ValueError,
                                    "Only subclasses of Keras optimizer_v2.Optimizer and TensorFlow"):
            OutfeedOptimizer(keras.optimizers.SGD(), None)


    def test_tf_optimizer_supplied(self):
        """ Test that a TensorFlow native optimizer can be supplied without
            raising an error. """
        OutfeedOptimizer(tf.compat.v1.train.GradientDescentOptimizer(0.001), None)
        pass
