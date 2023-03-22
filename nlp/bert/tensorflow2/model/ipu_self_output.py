# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import tensorflow as tf
from transformers.modeling_tf_utils import get_initializer
from transformers.models.bert.modeling_tf_bert import BertConfig, TFBertSelfOutput


class IpuTFBertSelfOutput(TFBertSelfOutput):
    def __init__(self, config: BertConfig, use_projection_bias=False, **kwargs):
        self.config = config
        self.use_projection_bias = use_projection_bias
        super().__init__(config, **kwargs)
        # use this flag here to keep the order of weights the same
        if not use_projection_bias:
            self.dense = tf.keras.layers.Dense(
                units=self.config.hidden_size,
                kernel_initializer=get_initializer(self.config.initializer_range),
                use_bias=self.use_projection_bias,
                name="dense",
            )

    def build(self, input_shape: tf.TensorShape):
        self.dense.build(input_shape)
        self.LayerNorm.build(input_shape)
        super().build(input_shape)
