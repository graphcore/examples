# Copyright 2019 Graphcore Ltd.
# coding=utf-8
"""
Derived from
https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo
"""
import tensorflow as tf
from tensorflow.contrib.compiler import xla
import tensorflow_probability as tfp
import time

try:
    from tensorflow.python import ipu
    device = '/device:IPU:0'
    scope = ipu.scopes.ipu_scope
    options = tf.python.ipu.utils.create_ipu_config()
    tf.python.ipu.utils.configure_ipu_system(options)
except ImportError:
    device = '/device:GPU:0'
    scope = tf.device

N_REPEATS = 100
N_LEAPFROG = 5
N_STEPS_PER_REPEAT = int(10e3)
TARGET_TIME_TEN_THOUSAND_STEPS = 0.22


# Target distribution is proportional to: `exp(-x (1 + x))`.
def unnormalized_log_prob(x):
    return -x - x**2.


# Initialize the HMC transition kernel.
hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_prob,
        num_leapfrog_steps=N_LEAPFROG,
        step_size=1.)


# Run single HMC step repeatedly
def run_single_steps():
    def _step(i, state):
        new_state, _ = hmc.one_step(state, hmc.bootstrap_results(state))
        return [i + 1, new_state]

    _, s = tf.while_loop(cond=lambda i, _: i < N_STEPS_PER_REPEAT,
                         body=_step,
                         loop_vars=[tf.constant(0), 1.])

    return s


# To test effect of bootstrap_results in run_single_steps(), run bootstrap_results in isolation
def test_bootstrap_results():
    def _step(i, state):
        new_state = hmc.bootstrap_results(state).proposed_state
        return [i + 1, new_state]

    _, s = tf.while_loop(cond=lambda i, _: i < N_STEPS_PER_REPEAT,
                         body=_step,
                         loop_vars=[tf.constant(0), 1.])

    return s


if __name__ == '__main__':
    with scope(device):
        ss = xla.compile(run_single_steps, ())
        # br = xla.compile(test_bootstrap_results, ())

    conf = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=conf)
    sess.run(tf.global_variables_initializer())

    # Run once to compile
    sess.run(ss)
    # sess.run(br)

    t_total = 0.
    t_total_br = 0.

    print('Running HMC.')
    for itr in range(N_REPEATS):
        # HMC
        t_bef = time.time()
        state_out = sess.run(ss)
        t_total += time.time() - t_bef

    # for itr in range(N_REPEATS):
    #     # Bootstrap results
    #     t_bef = time.time()
    #     _ = sess.run(br)
    #     t_total_br = time.time() - t_bef

    print(f'Avg time per step {t_total / float(N_REPEATS * N_STEPS_PER_REPEAT)}')
