# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python import ipu, framework
import unittest
import numpy as np


class TestFPExceptions(unittest.TestCase):

    def ipu_prog(self, x1, x2, fn, cast_to_float16, fp_exceptions):

        if cast_to_float16:
            tf.keras.mixed_precision.set_global_policy('float16')

        ds = tf.data.Dataset.from_tensor_slices((x1, x2))
        if cast_to_float16:
            ds = ds.map(lambda x1, x2: (tf.cast(x1, tf.float16), tf.cast(x2, tf.float16)))
        ds = ds.batch(1, drop_remainder=True)

        iterator = iter(ds)

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.floating_point_behaviour.inv = fp_exceptions
        cfg.floating_point_behaviour.div0 = fp_exceptions
        cfg.floating_point_behaviour.oflo = fp_exceptions
        cfg.floating_point_behaviour.nanoo = fp_exceptions
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            x1_input_layer = tf.keras.Input(shape=(1,))
            x2_input_layer = tf.keras.Input(shape=(1,))
            lambda_layer = tf.keras.layers.Lambda(fn)
            output = lambda_layer([x1_input_layer, x2_input_layer])
            model = tf.keras.Model([x1_input_layer, x2_input_layer], output)

        @tf.function
        def ipu_fn():
            x1_data, x2_data = next(iterator)
            return model([x1_data, x2_data])

        return strategy.run(ipu_fn).numpy()

    def test_simple_fp_exceptions_disabled(self):

        result = self.ipu_prog(x1=[1], x2=[1], fn=lambda x: x[0]/x[1], cast_to_float16=True, fp_exceptions=False)
        self.assertEqual(result, 1)

    def test_simple_fp_exceptions_enabled(self):

        result = self.ipu_prog(x1=[1], x2=[1], fn=lambda x: x[0]/x[1], cast_to_float16=True, fp_exceptions=True)
        self.assertEqual(result, 1)

    def test_div_2(self):
        result = self.ipu_prog(x1=[8], x2=[2], fn=lambda x: x[0]/x[1], cast_to_float16=True, fp_exceptions=False)

        self.assertEqual(result, 4)

    def test_div_0_fp_exceptions_disabled(self):

        result = self.ipu_prog(x1=[8], x2=[0], fn=lambda x: x[0]/x[1], cast_to_float16=True, fp_exceptions=False)
        self.assertTrue(np.isnan(result))

    def test_div_0_fp_exceptions_enabled(self):

        with self.assertRaises(framework.errors_impl.InternalError):
            self.ipu_prog(x1=[8], x2=[0], fn=lambda x: x[0]/x[1], cast_to_float16=True, fp_exceptions=True)

    def test_integer_div_0_fp_exceptions_disabled(self):

        result = self.ipu_prog(x1=[8], x2=[0], fn=lambda x: x[0]/x[1], cast_to_float16=False, fp_exceptions=False)
        self.assertEqual(result, np.inf)

    def test_integer_div_0_fp_exceptions_enabled(self):

        with self.assertRaises(framework.errors_impl.InternalError):
            self.ipu_prog(x1=[8], x2=[0], fn=lambda x: x[0]/x[1], cast_to_float16=False, fp_exceptions=True)

    def test_overflow_mult(self):

        with self.assertRaises(framework.errors_impl.InternalError):
            self.ipu_prog(x1=[65504], x2=[10], fn=lambda x: x[0]*x[1], cast_to_float16=True, fp_exceptions=True)

    def test_overflow_integer_sum_fp_exceptions_enabled(self):

        with self.assertRaises(framework.errors_impl.InternalError):
            self.ipu_prog(x1=[int(2**31-1)], x2=[1],
                          fn=lambda x: tf.cast(x[0]+x[1], tf.int32),
                          cast_to_float16=False, fp_exceptions=True)

    def test_overflow_integer_sum_fp_exceptions_disabled(self):

        result = self.ipu_prog(x1=[int(2**31-1)], x2=[1],
                               fn=lambda x: tf.cast(x[0]+x[1], tf.int32),
                               cast_to_float16=False, fp_exceptions=False)
        self.assertEqual(result, -2147483648)
