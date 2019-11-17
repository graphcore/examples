# Copyright 2019 Graphcore Ltd.
# Copyright (c) 2019, Graphcore Ltd, All rights reserved.

from tensorflow.python.ipu.scopes import ipu_scope, ipu_shard
from tensorflow.python.ipu import utils, loops, ipu_infeed_queue, ipu_compiler
import tensorflow_probability as tfp
import tensorflow as tf
import time as time
import numpy as np

# Model and sampling parameters
# Note: increasing model size, number of steps, or dataset size may cause out of memory errors
first_layer_size = 16
num_burnin_steps = 100
num_results = 400
num_leapfrog_steps = 200
input_file = "returns_and_features_for_mcmc.txt"
num_skip_columns = 2
output_file = "output_samples.txt"

# Print the about message
print("\nMCMC sampling example with TensorFlow Probability\n"
      " Single precision\n"
      " First layer size {}\n"
      " Number of burn-in steps {}\n"
      " Number of results {}\n"
      " Number of leapfrog steps {}"
      .format(first_layer_size,
              num_burnin_steps,
              num_results,
              num_leapfrog_steps
              ))

# Load data
raw_data = np.genfromtxt(input_file, skip_header=1,
                         delimiter="\t", dtype='float32')

# Pre-process data
num_features = raw_data.shape[1] - num_skip_columns - 1
observed_return_ = raw_data[:, num_skip_columns]
observed_features_ = raw_data[:, num_skip_columns+1:]

# Model is an MLP with num_features input dims and layer sizes: first_layer_size, 1, 1
num_model_parameters = num_features * first_layer_size + \
    first_layer_size + first_layer_size + 3

# Print dataset parameters
print(" Number of data items {}\n"
      " Number of features per data item {}\n"
      " Number of model parameters {}\n"
      .format(raw_data.shape[0],
              num_features,
              num_model_parameters
              ))


# Import TensorFlow modules
tfd = tfp.distributions

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize TensorFlow graph and session
tf.reset_default_graph()
config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)


# Build the neural network
def bdnn(x, p):
    nf = num_features
    nt = first_layer_size

    # Unpack model parameters
    w1 = tf.reshape(p[nt+1:nt+nf*nt+1], [nf, nt])
    w2 = tf.reshape(p[1:nt+1], [nt, 1])
    w3 = p[0]
    b1 = p[nt+nf*nt+3:]
    b2 = tf.expand_dims(p[nt+nf*nt+2], 0)
    b3 = p[nt+nf*nt+1]

    # Build layers
    x = tf.tanh(tf.nn.xw_plus_b(x, w1, b1))
    x = tf.nn.xw_plus_b(x, w2, b2)
    x = x * w3 + b3
    return tf.squeeze(x)


# Model posterior log probability
def model_log_prob(ret, feat, p):
    # Parameters of distributions
    prior_scale = 200
    studentT_scale = 100

    # Features normalization
    def normalize_features(f):
        return 0.001 * f

    # Prior probability distributions on model parameters
    rv_p = tfd.Independent(tfd.Normal(loc=0. * tf.ones(shape=[num_model_parameters], dtype=tf.float32),
                                      scale=prior_scale * tf.ones(shape=[num_model_parameters], dtype=tf.float32)),
                           reinterpreted_batch_ndims=1)

    # Likelihood
    alpha_bp_estimate = bdnn(normalize_features(feat), p)
    rv_observed = tfd.StudentT(
        df=2.2, loc=alpha_bp_estimate, scale=studentT_scale)

    # Sum of logs
    return (rv_p.log_prob(p) +
            tf.reduce_sum(rv_observed.log_prob(ret)))


# Place the graph on IPU
with ipu_scope('/device:IPU:0'):

    # Data items
    observed_return = tf.cast(observed_return_, 'float32')
    observed_features = tf.cast(observed_features_, 'float32')

    # Initial chain state
    initial_chain_state = [
        0.0 * tf.ones(shape=[num_model_parameters], dtype=tf.float32)
    ]

    # Bijectors
    unconstraining_bijectors = [
        tfp.bijectors.Identity()
    ]

    # Initialize the step_size
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        step_size = tf.get_variable(
            name='step_size',
            initializer=tf.constant(.01, dtype=tf.float32),
            trainable=False,
            use_resource=True
        )

    # Put the graph into a function so it can be compiled for running on IPU
    def hmc_graph():
        # Target log probability function
        def target_log_prob_fn(*args):
            return model_log_prob(observed_return, observed_features, *args)

        # Hamiltonian Monte Carlo kernel
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                    target_rate=0.2,
                    num_adaptation_steps=num_burnin_steps,
                    decrement_multiplier=0.1),
                state_gradients_are_stopped=False),
            bijector=unconstraining_bijectors)

        # Graph to sample from the chain
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_chain_state,
            kernel=hmc_kernel)

    # Compile the graph
    [p], kernel_results = ipu_compiler.compile(hmc_graph, [])


# Configure IPU
config = utils.create_ipu_config()
config = utils.auto_select_ipus(config, [1])
utils.configure_ipu_system(config)

# Initialize variables
utils.move_variable_initialization_to_cpu()
init_g = tf.global_variables_initializer()
sess.run(init_g)

# Warm up
print("\nWarming up...")
sess.run([p, kernel_results])
print("Done\n")

# Sample
print("Sampling...")
start_time = time.time()
[samples_, kernel_results_] = sess.run([p, kernel_results])
end_time = time.time()
print("Done\n")

# Write samples to file
np.savetxt(output_file, samples_, delimiter='\t')

# Print result
print("Written samples to {}".format(output_file))
print("Acceptance rate {0:.2f}".format(
    kernel_results_.inner_results.is_accepted.mean()))
print("Final step size {0:.4f}".format(
    kernel_results_.inner_results.extra.step_size_assign[0]))
print("Completed in {0:.2f} seconds\n".format(end_time - start_time))
