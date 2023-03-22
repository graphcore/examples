# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import unittest
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).absolute().parent.parent))
from precision import Precision
from custom_exceptions import UnsupportedFormat


class PrecisionAPITest(unittest.TestCase):
    # these tests will fail if the API breaks

    def test_keras_setfloat16(self):
        tf.keras.mixed_precision.set_global_policy("float16")
        assert tf.keras.mixed_precision.global_policy().name == "float16"

    def test_keras_setfloat32(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        assert tf.keras.mixed_precision.global_policy().name == "float32"

    def test_create_float16_layers(self):
        tf.keras.mixed_precision.set_global_policy("float16")
        input_tensor = tf.keras.Input(shape=(None, 1))
        dense_layer = tf.keras.layers.Dense(1.0, activation=None)
        output_tensor = dense_layer(input_tensor)
        assert dense_layer.weights[0].dtype == tf.float16  # test kernel type
        assert output_tensor.dtype == tf.float16


class InvalidPrecisionTest(unittest.TestCase):
    def test_invalid_precision(self):
        self.assertRaises(NameError, Precision, "foo")


class UnsupportedPrecisionTest(unittest.TestCase):
    def test_unsupported_precision(self):
        self.assertRaises(UnsupportedFormat, Precision, "32.16")


class Precision32_32Test(unittest.TestCase):
    def setUp(self):
        self.precision_instance = Precision("32.32")

    def test_full_precision_instantiation(self):
        assert self.precision_instance.compute_precision == tf.float32
        assert self.precision_instance.weight_update_precision == tf.float32

    def test_set_precision(self):
        self.precision_instance.apply()

        input_layer = tf.keras.layers.Input(shape=(1))
        layer = tf.keras.layers.Dense(1)
        assert layer.dtype == tf.float32
        output = layer(input_layer)
        assert output.dtype == tf.float32

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        with tf.GradientTape() as tape:
            input_data = np.array([[1]])
            output_data = model(input_data)

            grads = tape.gradient(output_data, model.trainable_variables)
            assert all([g.dtype == tf.float32 for g in grads])


class Precision16_16Test(unittest.TestCase):
    def setUp(self):
        self.precision_instance = Precision("16.16")

    def test_half_precision_instanciation(self):
        assert self.precision_instance.compute_precision == tf.float16
        assert self.precision_instance.weight_update_precision == tf.float16

    def test_set_precision(self):
        self.precision_instance.apply()

        input_layer = tf.keras.layers.Input(shape=(None, 1))
        layer = tf.keras.layers.Dense(1)
        assert layer.dtype == tf.float16
        output = layer(input_layer)
        assert output.dtype == tf.float16

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        with tf.GradientTape() as tape:
            input_data = np.array([[1]])
            output_data = model(input_data)

            grad = tape.gradient(output_data, model.trainable_variables)
            assert all([g.dtype == tf.float16 for g in grad])


class Precision_16_32Test(unittest.TestCase):
    def setUp(self):
        self.precision_instance = Precision("16.32")

    def test_half_precision_instanciation(self):
        assert self.precision_instance.compute_precision == tf.float16
        assert self.precision_instance.weight_update_precision == tf.float32

    def test_set_precision(self):
        self.precision_instance.apply()

        input_layer = tf.keras.layers.Input(shape=(None, 1))
        layer = tf.keras.layers.Dense(1)
        assert layer.compute_dtype == tf.float16
        output = layer(input_layer)
        assert output.dtype == tf.float16

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        with tf.GradientTape() as tape:
            input_data = np.array([[1]])
            output_data = model(input_data)

            grad = tape.gradient(output_data, model.trainable_variables)
            # note gradients dtype are tf.float32
            assert all([g.dtype == tf.float32 for g in grad])
        # note that model weights are also tf.float32
        assert all([var.dtype == tf.float32 for var in model.trainable_variables])


class PrecisionUnderflowTest(unittest.TestCase):
    # this test divides a small number by 10 (multiplies by 0.1)
    # if the computation is done in float16 there will be an underflow

    def setUp(self):
        self.input_value = np.array([[1e-7]])
        self.weight = np.array(0.1)
        self.expected_value = self.input_value * self.weight

    def get_model(self):
        input_layer = tf.keras.layers.Input(shape=(1))
        initializer = tf.keras.initializers.Constant(self.weight)
        layer = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=initializer)
        dense_output = layer(input_layer)
        full_precision_output = tf.cast(dense_output, dtype=tf.float32)
        model = tf.keras.Model(input_layer, full_precision_output)

        return model

    def test_half_precision(self):
        half_precision = Precision("16.16")
        half_precision.apply()

        model = self.get_model()

        computed_value = model(self.input_value).numpy()

        # checking for underflow
        self.assertNotAlmostEqual(self.expected_value[0][0], computed_value[0][0], places=10)

    def test_full_precision(self):
        full_precision = Precision("32.32")
        full_precision.apply()

        model = self.get_model()

        computed_value = model(self.input_value).numpy()

        # checking it did NOT underflow
        self.assertAlmostEqual(self.expected_value[0][0], computed_value[0][0], places=10)

    def test_mixed_precision(self):
        mixed_precision = Precision("16.32")
        mixed_precision.apply()

        model = self.get_model()

        computed_value = model(self.input_value).numpy()

        # checking for underflow
        self.assertNotAlmostEqual(self.expected_value[0][0], computed_value[0][0], places=10)


class PrecisionWeightUpdateTest(unittest.TestCase):
    # this test checks the precision of a weight update
    # this is done by forcing a sum between a large number (weight)
    # with a small number (gradient). If the weight update is done
    # in float32 the weight will change, if it is done in float16
    # the weight keeps its initial value

    def setUp(self):
        self.weight_initial_value = 1000
        self.loss_factor_value = 1e-1

    def perform_weight_update(self):
        input_layer = tf.keras.layers.Input(shape=(1))

        weight_initializer = tf.keras.initializers.Constant(self.weight_initial_value)
        layer = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=weight_initializer)

        dense_output = layer(input_layer)
        model = tf.keras.Model(input_layer, dense_output)

        loss_factor_initializer = tf.keras.initializers.Constant(self.loss_factor_value)
        loss = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=loss_factor_initializer, trainable=False)

        sgd = tf.keras.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            input_data = np.array([[1]])
            loss_data = loss(model(input_data))

        grads = tape.gradient(loss_data, model.trainable_variables)
        sgd.apply_gradients(zip(grads, model.trainable_variables))

        print(grads)
        print(model.trainable_variables)

        return model.trainable_variables[0][0][0].numpy()

    def test_mixed_precision_update(self):
        Precision("16.32").apply()

        weight = self.perform_weight_update()

        # weight changed, weight update done in 32 bits
        self.assertNotEqual(self.weight_initial_value, weight)

    def test_half_precision_update(self):
        Precision("16.16").apply()

        weight = self.perform_weight_update()

        # weight didn't change, weight update done in 16 bits
        self.assertEqual(self.weight_initial_value, weight)
