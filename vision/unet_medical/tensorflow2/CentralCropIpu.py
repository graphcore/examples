# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf
from tensorflow.python import ipu


class CentralCropIpu(tf.keras.layers.Layer):
    """This is a custom operation for central crop without inplacing"""

    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size

    def call(self, inputs, *args, **kwargs):
        inputs_shape = tf.shape(inputs)
        img_hd = inputs_shape[1]
        img_diff = img_hd - self.output_size
        check = tf.debugging.assert_non_negative(
            img_diff, message="The crop output size {} should not be greater than input size".format(self.output_size)
        )

        with tf.control_dependencies([check]):
            return self.crop_custom_op(inputs)

    def crop_custom_op(self, inputs):
        output_shape = (inputs.shape[0], self.output_size, self.output_size, inputs.shape[3])
        outputs = {
            "output_types": [inputs.dtype],
            "output_shapes": [tf.TensorShape(output_shape)],
        }

        base_path = os.path.realpath(os.path.dirname(__file__))
        lib_path = os.path.join(base_path, "libcustom_op.so")
        central_fraction = str(self.output_size / inputs.shape[1])
        cropped = ipu.custom_ops.precompiled_user_op(
            [inputs], lib_path, attributes=central_fraction, gradient_attributes=central_fraction, outs=outputs
        )
        return cropped[0]
