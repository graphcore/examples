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

args = parse_params()
(train_dataset, test_dataset), (train_data_len, test_data_len) = load_data(args.batch_size)

if not args.use_ipu:

    model = keras.Model(*model_fn(input_shape, num_classes))

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.

    model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"], steps_per_execution=1)
    model.summary()

    print("Training")
    train_dataset = train_dataset.batch(args.batch_size)
    steps_per_epoch = train_data_len // args.batch_size
    model.fit(
        train_dataset,
        epochs=3,
        batch_size=args.batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    print("Evaluation")
    test_dataset = test_dataset.batch(args.batch_size)
    model.evaluate(test_dataset)

else:

    num_ipus = 2

    # Standard IPU TensorFlow setup.
    ipu_config = ipu.config.IPUConfig()
    ipu_config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()

    # Create an execution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():

        if not args.pipelining:
            # Note that the model function from the CPU example can be reused.
            ipu_model = keras.Model(*model_fn(input_shape, num_classes))
            steps_per_execution = train_data_len // args.batch_size // num_ipus
            steps_per_epoch = train_data_len // args.batch_size
            evaluation_steps = test_data_len // args.batch_size
        else:
            ipu_model = keras.Model(*pipeline_model_fn(input_shape, num_classes))
            ipu_model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=args.gradient_accumulation_steps_per_replica
            )
            steps_per_execution = train_data_len // args.batch_size
            # Make sure an integer number of gradient updates occur during
            # one execution of the IPU program
            steps_per_execution -= steps_per_execution % args.gradient_accumulation_steps_per_replica
            steps_per_epoch = steps_per_execution
            evaluation_steps = test_data_len // args.batch_size
            # Make sure that there are no partial executions of the pipeline
            # (note that num_ipus==pipeline_length)
            evaluation_steps -= evaluation_steps % num_ipus

        # Compile our model as with the CPU example.
        ipu_model.compile(
            "sgd",
            "categorical_crossentropy",
            metrics=["accuracy"],
            steps_per_execution=steps_per_execution,
        )
        ipu_model.summary()

        train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)

        print("Training")
        ipu_model.fit(
            train_dataset,
            epochs=3,
            batch_size=args.batch_size,
            steps_per_epoch=steps_per_epoch,
        )

        print("Evaluation")
        test_dataset = test_dataset.batch(args.batch_size, drop_remainder=True)

        result = ipu_model.evaluate(test_dataset, batch_size=args.batch_size, steps=evaluation_steps)
        print(f"loss: {result[0]:.4f} - accuracy: {result[1]:.4f}")

print("Program ran successfully")
