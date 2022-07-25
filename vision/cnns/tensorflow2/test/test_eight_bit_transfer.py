# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).absolute().parent.parent))
from eight_bit_transfer import EightBitTransfer
from custom_exceptions import InvalidPrecisionException


class CheckEightBitCastingBehaviour(unittest.TestCase):

    def test_eight_bit_transfer_compress_func(self):
        test_input = tf.random.uniform(
            (10,), minval=0, maxval=256, dtype=tf.float32, seed=42)
        eight_bit_transfer_cast_result = EightBitTransfer(tf.float32).compress(test_input)
        manual_result = tf.math.floor(test_input)
        difference = test_input.numpy() - tf.cast(eight_bit_transfer_cast_result,
                                                  dtype=tf.float32).numpy()
        assert((difference >= 0).all() and (difference < 1).all())
        assert((tf.cast(eight_bit_transfer_cast_result,
               dtype=tf.float32).numpy() == manual_result.numpy()).all())
        self.assertEqual(eight_bit_transfer_cast_result.dtype, tf.uint8)

    def test_eight_bit_transfer_decompress_func_fp16(self):
        compute_precision = tf.float16
        test_input = tf.cast(tf.range(-1, 257, 1), dtype=tf.uint8)
        eight_bit_transfer_obj = EightBitTransfer(compute_precision)
        eight_bit_transfer_cast_result = eight_bit_transfer_obj.decompress(
            test_input)
        manual_cast_result = tf.constant(
            tf.cast(test_input, dtype=eight_bit_transfer_obj.tf_compute_precision))
        assert((eight_bit_transfer_cast_result.numpy() == manual_cast_result.numpy()).all())
        self.assertEqual(eight_bit_transfer_cast_result.dtype,
                         eight_bit_transfer_obj.tf_compute_precision)

    def test_eight_bit_transfer_decompress_func_fp32(self):
        compute_precision = tf.float32
        test_input = tf.cast(tf.range(-1, 257, 1), dtype=tf.uint8)
        eight_bit_transfer_obj = EightBitTransfer(compute_precision)
        eight_bit_transfer_cast_result = eight_bit_transfer_obj.decompress(
            test_input)
        manual_cast_result = tf.constant(
            tf.cast(test_input, dtype=eight_bit_transfer_obj.tf_compute_precision))
        assert((eight_bit_transfer_cast_result.numpy() == manual_cast_result.numpy()).all())
        self.assertEqual(eight_bit_transfer_cast_result.dtype,
                         eight_bit_transfer_obj.tf_compute_precision)

    def test_compress_func_w_overflow(self):
        compute_precision = tf.float16
        test_input = tf.Variable(tf.range(-1, 257, 1))
        expected_result = tf.Variable(tf.range(-1, 257, 1))
        expected_result[0].assign(255)
        expected_result[-1].assign(0)
        eight_bit_transfer_obj = EightBitTransfer(compute_precision)
        eight_bit_transfer_cast_result = eight_bit_transfer_obj.compress(
            test_input)
        assert((eight_bit_transfer_cast_result.numpy() == expected_result.numpy()).all())

    def test_valid_compute_precision(self):
        with self.assertRaises(InvalidPrecisionException):
            _ = EightBitTransfer(64)
