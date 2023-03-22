# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import pytest
import sys
import tensorflow as tf


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from outfeed_optimizer import OutfeedOptimizer


def test_no_model_supplied():
    with pytest.raises(ValueError, match="The model must be specified if a Keras"):
        OutfeedOptimizer(tf.keras.optimizers.SGD(), None)


def test_tf_optimizer_supplied():
    """Test that a TensorFlow native optimizer can be supplied without
    raising an error."""
    OutfeedOptimizer(tf.compat.v1.train.GradientDescentOptimizer(0.001), None)
