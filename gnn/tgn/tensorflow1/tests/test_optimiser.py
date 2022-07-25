# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

import optimiser


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_adam(dtype: tf.dtypes.DType) -> None:
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("impl"):
            impl_x = tf.get_variable("x", (),
                                     dtype=dtype,
                                     initializer=tf.zeros_initializer())
            impl_update = optimiser.Adam(0.1).minimize_with_global_step(
                (impl_x - 3)**2)

        with tf.variable_scope("ref"):
            ref_x = tf.get_variable("x", (),
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            ref_update = tf.train.AdamOptimizer(0.1).minimize((ref_x - 3)**2)

        session.run(tf.global_variables_initializer())
        for _ in range(10):
            impl, ref, _, _ = session.run(
                [impl_x, ref_x, impl_update, ref_update])
            np.testing.assert_allclose(impl, ref, rtol=1e-3)
        assert ref < 6
