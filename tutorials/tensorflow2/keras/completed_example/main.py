# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow.keras as keras
from tensorflow.python import ipu

from model import model_fn, pipeline_model_fn
from utils import load_data, parse_params


# Store class and shape information.
num_classes = 10
input_shape = (28, 28, 1)

x_train, y_train, x_test, y_test = load_data()
args = parse_params()

if not args.use_ipu:
    # Model.__init__ takes two required arguments, inputs and outputs.
    model = keras.Model(*model_fn(input_shape, num_classes))

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    print('Training')
    model.fit(x_train, y_train, epochs=3, batch_size=64)
    print('Evaluation')
    model.evaluate(x_test, y_test)
else:
    # Standard IPU TensorFlow setup.
    ipu_config = ipu.utils.create_ipu_config()
    ipu_config = ipu.utils.auto_select_ipus(ipu_config, 2)
    ipu.utils.configure_ipu_system(ipu_config)

    # Create an execution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        # As with keras.Model.__init__, ipu.keras.Model.__init__ takes two
        # required arguments, inputs and outputs.
        if not args.pipelining:
            # Note that the model function from the CPU example can be reused.
            ipu_model = ipu.keras.Model(*model_fn(input_shape, num_classes))
        else:
            ipu_model = ipu.keras.PipelineModel(
                *pipeline_model_fn(input_shape, num_classes),
                gradient_accumulation_count=args.gradient_accumulation_count)

        # Compile our model as with the CPU example.
        ipu_model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])
        ipu_model.summary()

        print("Training")
        ipu_model.fit(x_train, y_train, epochs=3, batch_size=64)

        print("Evaluation")
        result = ipu_model.evaluate(x_test, y_test, batch_size=64)
        print(f"loss: {result[0]:.4f} - accuracy: {result[1]:.4f}")
