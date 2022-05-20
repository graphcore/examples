# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import tensorflow.compat.v1 as tf


class ClosureInitializer(tf.keras.initializers.Initializer):
    """
    Returns initial tensor from call to closed function, ignoring
    shape and dtype args passed into __call__ in favour of any
    bound into the closure.
    """
    def __init__(self, closure):
        self.func = closure

    def __call__(self, shape, dtype, **kwargs):
        return self.func()
