# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

"""Networks for VAE implementation in VCD paper"""
import numpy as np
import tensorflow.compat.v1 as tf


def encoder(X_input, Z_dim, dtype, n_hidden=200):
    """As in paper"""
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE, use_resource=True):

        # Calculate sqrt(n_hidden) - for initialisers
        sqrt_n_hid_inv = 1. / np.sqrt(float(n_hidden))

        # Separate networks for approx posterior mean and log std
        with tf.variable_scope('mean', use_resource=True, reuse=tf.AUTO_REUSE):
            relu0_mean = tf.layers.dense(
                X_input,
                units=n_hidden,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                name='relu0_mu')

            relu1_mean = tf.layers.dense(
                relu0_mean,
                units=n_hidden,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                name='relu1_mu')
            Z_cond_X_mean = tf.layers.dense(
                relu1_mean,
                units=Z_dim,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(dtype=dtype),
                bias_initializer=tf.random_normal_initializer(dtype=dtype),
                name='posterior_mean')

        with tf.variable_scope('std', use_resource=True, reuse=tf.AUTO_REUSE):
            relu0_std = tf.layers.dense(
                X_input,
                units=n_hidden,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                name='relu0_std')

            relu1_std = tf.layers.dense(
                relu0_std,
                units=n_hidden,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
                name='relu1_std')

            Z_cond_X_log_std = tf.layers.dense(
                relu1_std,
                units=Z_dim,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(dtype=dtype),
                bias_initializer=tf.random_normal_initializer(dtype=dtype),
                name='posterior_log_std')

        # More numerically-stable exponential function
        def _pos_softplus(x):
            return x + tf.log(1. + tf.exp(1e-4 - x))

        def _neg_softplus(x):
            return 1e-4 + tf.log(1. + tf.exp(x - 1e-4))

        Z_cond_X_std = tf.where(Z_cond_X_log_std >= 0,
                                _pos_softplus(Z_cond_X_log_std),
                                _neg_softplus(Z_cond_X_log_std))
        return Z_cond_X_mean, Z_cond_X_std


def decoder(Z_cond_X_samples, output_dims, dtype, n_hidden=200):
    """As in paper"""
    # Calculate sqrt(n_hidden) - for initialisers
    sqrt_n_hid_inv = 1. / np.sqrt(float(n_hidden))

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE, use_resource=True):
        relu0_dec = tf.layers.dense(
            Z_cond_X_samples,
            units=n_hidden,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
            bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
            name='relu0_dec')

        relu1_dec = tf.layers.dense(
            relu0_dec,
            units=n_hidden,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
            bias_initializer=tf.random_normal_initializer(stddev=sqrt_n_hid_inv, dtype=dtype),
            name='relu1_dec')

        lin_out = tf.layers.dense(
            relu1_dec,
            units=np.prod(output_dims),
            activation=None,
            kernel_initializer=tf.random_normal_initializer(dtype=dtype),
            bias_initializer=tf.random_normal_initializer(dtype=dtype),
            name='dec_lin_out')

        return {'logits': lin_out}
