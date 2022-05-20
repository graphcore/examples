# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from ipu_tensorflow_addons.keras.layers import Dropout as IpuDropout
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization


class IpuLayerNormCustom(IpuLayerNormalization):
    """A modification to the IPU Layer Normalization layer that
       skips the transpose operations."""

    def call(self, inputs, training=None):
        input_shape = inputs.shape
        permuted = tf.reshape(inputs, (-1, input_shape[-1]))
        # Call group norm
        outputs = super(IpuLayerNormalization, self).call(permuted, training)
        return tf.reshape(outputs, input_shape)


class IpuDropoutCustom(IpuDropout):
    """A modification to the IPU Dropout layer that skips the layer
       call if the rate is zero. This means that in this case
       the associated operations aren't added to the graph."""

    def call(self, inputs, training=None):
        if self.rate:
            return super().call(inputs, training=training)
        return inputs
