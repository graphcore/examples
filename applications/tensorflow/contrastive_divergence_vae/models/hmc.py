# Copyright 2019 Graphcore Ltd.
# coding=utf-8
from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def update_step_size_ruiz(step_size, mean_accept_prob, adapt_rate=0.01, target_acc_prob=0.9):
    """Update MCMC step size as in the matlab implementation. Update is based on the fractional
    difference between the rate of accepted proposals and the target acceptance rate"""
    acc_ratio = (mean_accept_prob - target_acc_prob) / target_acc_prob
    return step_size + adapt_rate * acc_ratio * step_size


def update_step_size_constant(step_size, mean_accept_prob, adapt_rate=0.01, target_acc_prob=0.75):
    """Don't update step size"""
    return step_size


class HamiltonianMonteCarlo(object):
    def __init__(self, config, batch_size, testing=False, keep_samples=False, Z_dim=10, dtype=tf.float32):
        # Number of steps
        self.n_burn_in_steps = config.get('n_burn_in_steps', 300 if testing else 8)
        self.n_post_burn_steps = config.get('n_hmc_steps', 300 if testing else 0)
        self.n_leapfrog_steps = config.get('n_leapfrog_steps', 5)

        # Step size adaption
        self.hmc_step_size_init = config.get('hmc_step_size_init', 0.5 / Z_dim)
        target_acc_prob = config.get('acceptance_target', 0.9)
        adapt_fn_name = config.get('step_size_adaption_fn', 'ruiz')
        adaption_rate = config.get('step_size_adaption_rate', 0.01 * batch_size / 100)
        adapt_fns = {'ruiz': update_step_size_ruiz,
                     'constant': update_step_size_constant}

        # Set function to update step size
        self.update_step_size = partial(adapt_fns[adapt_fn_name],
                                        target_acc_prob=target_acc_prob,
                                        adapt_rate=adaption_rate)

        self.dtype = dtype

        self.testing = testing
        self.keep_samples = keep_samples  # Just output final states or all states after burn-in?

        if self.testing:
            # Whether to initialise test step size with current training value
            self.init_test_mcmc_train_step_size = config.get('use_train_step_size_init', False)

    def leapfrog_step(self, lf_step_id, state, mom, ss):
        state += ss * mom
        proposed_U = -self.target_log_prob(state)
        grad_U = tf.gradients(proposed_U, [state])[0]
        mom -= ss * grad_U
        return [lf_step_id + 1, state, mom, ss]

    def leapfrog_cond(self, lf_step_id, state, mom, ss):
        return tf.less(lf_step_id, self.n_leapfrog_steps)

    def hmc_step(self, hmc_step_id, state, samples, ss):
        """Single HMC step, based on Neal, 2011.
        See p. 14 https://arxiv.org/pdf/1206.1901.pdf"""

        # Half step for momentum
        momentum_init = tf.random.normal(tf.shape(state))
        energy_init = -self.target_log_prob(state)
        grad_energy = tf.gradients(energy_init, [state])[0]
        momentum = momentum_init - grad_energy * ss * 0.5

        # LF updates
        _, z_sample_lf, momentum_lf, _ = \
            tf.while_loop(cond=self.leapfrog_cond,
                          body=self.leapfrog_step,
                          loop_vars=[tf.constant(0),
                                     state,
                                     momentum,
                                     ss],
                          back_prop=False,
                          maximum_iterations=self.n_leapfrog_steps - 1,
                          name='leapfrog_loop')

        # Another half step for momentum - final leapfrog step
        z_sample_lf += ss * momentum_lf
        prop_energy_lf = -self.target_log_prob(z_sample_lf)
        grad_U = tf.gradients(prop_energy_lf, [z_sample_lf])[0]
        momentum_lf -= ss * grad_U * 0.5

        # Negate momentum at end of trajectory to make the proposal symmetric (as per Ruiz code)
        momentum_lf = -momentum_lf

        # Kinetic energies before/after
        T_init = tf.reduce_sum(momentum_init ** 2, -1) * 0.5  # before
        T_prop = tf.reduce_sum(momentum_lf ** 2, -1) * 0.5  # after

        # Do accept/reject based on ratio of energies
        accept = tf.random_uniform(tf.shape(T_init), ) < tf.exp(energy_init - prop_energy_lf + T_init - T_prop)
        new_state = tf.where(accept, z_sample_lf, state)

        # Update the step size if in burn in period
        mean_accept = tf.reduce_mean(tf.cast(accept, self.dtype))
        step_size_update = self.update_step_size(ss, mean_accept)
        step_size_maybe_update = tf.where(hmc_step_id < self.n_burn_in_steps, step_size_update, ss)
        if self.keep_samples:
            samples = samples.write(hmc_step_id, new_state)

        return [hmc_step_id + 1, new_state, samples, step_size_maybe_update]

    def hmc_cond(self, hmc_step_id, state, samples, ss):
        return tf.less(hmc_step_id, self.n_burn_in_steps + self.n_post_burn_steps)

    def run_hmc(self, z_sample_init, unnorm_log_prob_target):
        # # Standard normal prior
        # p_z = tfd.MultivariateNormalDiag(tf.zeros_like(z_sample_init), tf.ones_like(z_sample_init))
        #
        with tf.variable_scope('hmc', reuse=tf.AUTO_REUSE, use_resource=True):
            hmc_step_size_tr = tf.get_variable(name='hmc_step_size_tr',
                                               shape=(),
                                               initializer=tf.constant_initializer(self.hmc_step_size_init),
                                               dtype=self.dtype,
                                               use_resource=True,
                                               trainable=False)
        if self.testing:
            if self.init_test_mcmc_train_step_size:
                hmc_step_size = hmc_step_size_tr
            else:
                hmc_step_size = self.hmc_step_size_init
        else:
            hmc_step_size = hmc_step_size_tr

        self.target_log_prob = unnorm_log_prob_target
        if self.keep_samples:
            z_samples = tf.TensorArray(dtype=self.dtype,
                                       size=self.n_burn_in_steps + self.n_post_burn_steps,
                                       name='hmc_keep_samples_arr')
        else:
            z_samples = tf.zeros((), dtype=self.dtype)

        _, z_t, z_samples_arr, new_step_size = \
            tf.while_loop(cond=self.hmc_cond,
                          body=self.hmc_step,
                          loop_vars=[tf.constant(0),
                                     z_sample_init,
                                     z_samples,
                                     hmc_step_size],
                          back_prop=False,
                          maximum_iterations=self.n_burn_in_steps + self.n_post_burn_steps,
                          name='hmc_while_loop')

        hmc_step_size_op = tf.no_op() if self.testing else tf.assign(hmc_step_size, new_step_size)

        if self.keep_samples:
            z_samples = tf.reshape(z_samples_arr.stack(),
                                   (self.n_burn_in_steps + self.n_post_burn_steps,
                                    tf.shape(z_sample_init)[0],
                                    tf.shape(z_sample_init)[-1]))[-self.n_post_burn_steps:]

            return z_samples, hmc_step_size_op
        else:
            return z_t, hmc_step_size_op
