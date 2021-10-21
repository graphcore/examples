# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 David J. Warne, Centre for Data Science, QUT, AUS
"""
ABC algorithm for COVID-19 modelling, replicated across multiple IPUs.

See README for model background.
"""

import numpy as np
import os
import time as time
from tensorflow.python import ipu
import tensorflow as tf
import tensorflow_probability as tfp

import covid_data
from argparser import get_argparser

tfd = tfp.distributions

# Parse the CLI args
ap = get_argparser()
args = ap.parse_args()

assert (not args.enqueue_chunk_size or
        args.n_samples_per_batch % args.enqueue_chunk_size == 0), \
    "--enqueue-chunk-size must divide into --n-samples-per-batch exactly"
if args.samples_filepath:
    assert os.path.exists(os.path.dirname(os.path.abspath(args.samples_filepath))), \
        "Path to save samples (--samples-fn) does not exist."


# Mapping to tf constants to avoid graph recompilation.
args.tolerance = tf.constant(args.tolerance, dtype=tf.float32)
args.n_samples_target = tf.constant(args.n_samples_target, dtype=tf.int32)
args.max_n_runs = tf.constant(args.max_n_runs, dtype=tf.int32)
# The parameters args.enqueue_chunk_size and n_samples_per_batch are not mapped
# to constants since they change the data structure and respective
# layout of convolutions on the IPU.

# Modelling constants
COUNTRY_DATA_TRAIN, POPULATION = covid_data.get_data(args.country)
# Casting population to tf.constant avoids recompilation but increases
# processing time by around 15%
# POPULATION = tf.constant(POPULATION, dtype=tf.float32)

# State transition matrix     S   I  A  R  D  Ru
MIXING_MATRIX = tf.constant([[-1, 1, 0, 0, 0, 0],   # S + I -> 2I
                             [0, -1, 1, 0, 0, 0],   # I -> A
                             [0, 0, -1, 1, 0, 0],   # A -> R
                             [0, 0, -1, 0, 1, 0],   # A -> D
                             [0, -1, 0, 0, 0, 1]],  # I -> Ru
                            dtype=tf.float32)
# Define uniform prior limits
# theta ~ U(l,u) theta = [alpha_0,alpha,beta,gamma,delta,eta,n,kappa,-log w]
UNIFORM_PRIOR_LOWER_LIMIT = tf.constant(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
UNIFORM_PRIOR_UPPER_LIMIT = tf.constant(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 100.0, 6.0])

# Run args
MAX_REPORT_SIZE = int(5e9)

if args.n_days is None:
    country_data_train = COUNTRY_DATA_TRAIN
else:
    country_data_train = COUNTRY_DATA_TRAIN[:, :args.n_days]


def configure_ipu():
    """Configure and reserve IPUs."""

    cfg = ipu.utils.IPUConfig()
    cfg.auto_select_ipus = args.replication_factor
    cfg.configure_ipu_system()


configure_ipu()

# Create an IPU distribution strategy.
strategy = ipu.ipu_strategy.IPUStrategy()

# Create outfeed for streaming data to host
outfeed_data = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
    'outfeed_data', replication_factor=args.replication_factor)


def conditional_enqueue_op(params, n_accs, dists, gain):
    """Enqueue only if relevant samples are included."""
    def _enq_fn(to_enq):
        return tf.no_op() if args.no_outfeed_ops \
            else outfeed_data.enqueue(to_enq)

    if args.outfeed_num_samples:
        maybe_enqueue_op = tf.cond(
            tf.math.greater(gain, 0),
            lambda: _enq_fn([params, dists, n_accs]),
            lambda: tf.no_op()
        )
    else:
        maybe_enqueue_op = tf.cond(
            tf.math.greater(gain, 0),
            lambda: _enq_fn([params, dists]),
            lambda: tf.no_op()
        )
    return maybe_enqueue_op


def chunked_outfeed_enqueue(chunk_id, total_gain, p_vec, d_vec, acc_mask):
    """Enqueue only relevant chunks.

    Iterate over chunks of param vector samples,
    only enqueue the host to outfeed if it has an
    accepted sample in it
    """
    # sync between replicas
    g = ipu.cross_replica_ops.assume_equal_across_replicas(
        ipu.cross_replica_ops.cross_replica_sum(
            acc_mask[chunk_id], name="accumulated_sum"))
    maybe_enqueue = \
        conditional_enqueue_op(params=tf.gather(p_vec, chunk_id, axis=1),
                               dists=tf.gather(d_vec, chunk_id),
                               n_accs=acc_mask[chunk_id],
                               gain=g)

    with tf.control_dependencies([maybe_enqueue]):
        g = tf.identity(g)
    return chunk_id + 1, total_gain + g, p_vec, d_vec, acc_mask


@tf.function(experimental_compile=True)
def build_graph(accumulated_number_of_samples, run_number, local_tolerance):
    """Run full simulation over all days."""
    # init of the simulation
    n_days = tf.cast(country_data_train.shape[1], tf.int32)
    P = tf.ones(args.n_samples_per_batch) * POPULATION
    A_0 = tf.ones(args.n_samples_per_batch) * country_data_train[0, 0]
    R_0 = tf.ones(args.n_samples_per_batch) * country_data_train[1, 0]
    D_0 = tf.ones(args.n_samples_per_batch) * country_data_train[2, 0]
    # param_vector elements are
    # [alpha_0, alpha, beta, gamma, delta, eta, n, kappa, -log w]
    param_vector = tf.transpose(tfd.Uniform(
        UNIFORM_PRIOR_LOWER_LIMIT,
        UNIFORM_PRIOR_UPPER_LIMIT,
        ).sample(args.n_samples_per_batch))

    summary = tf.zeros([n_days, 3, args.n_samples_per_batch])

    S_store = P - param_vector[7] * A_0 - (A_0 + R_0 + D_0)
    I_store = param_vector[7] * A_0
    A_store = A_0
    R_store = R_0
    D_store = D_0
    Ru_store = tf.zeros(args.n_samples_per_batch)

    summary = tf.tensor_scatter_nd_add(
        tensor=summary,
        indices=[[0, 0], [0, 1], [0, 2]],
        updates=tf.stack([A_store, R_store, D_store]))

    init_idx = tf.zeros([], dtype=tf.int32) + 1
    init_vars = \
        [init_idx, summary, S_store, I_store,
         A_store, R_store, D_store, Ru_store]

    def body(i, s, S, I, A, R, D, Ru):
        """Single update for one day."""
        U = A / (tf.pow(10.0, param_vector[8]))
        alpha_t = param_vector[0] + (
            param_vector[1] / (tf.constant(1.0) + tf.pow(U, param_vector[6])))
        h_1 = (S * I / P) * alpha_t
        h_2 = I * param_vector[3]
        h_3 = A * param_vector[2]
        h_4 = A * param_vector[4]
        h_5 = I * param_vector[2] * param_vector[5]
        h = tf.stack([h_1, h_2, h_3, h_4, h_5])
        normal_sample = tfd.Normal(loc=h, scale=tf.sqrt(h)).sample()
        Y_store = tf.clip_by_value(tf.math.floor(normal_sample), 0.0, P)

        m = tf.matmul(tf.transpose(MIXING_MATRIX), Y_store)

        # Note: Simple vectorisation suppresses parameter update in loop.
        S = tf.clip_by_value(S + m[0, :], 0.0, P)
        I = tf.clip_by_value(I + m[1, :], 0.0, P)
        A = tf.clip_by_value(A + m[2, :], 0.0, P)
        R = tf.clip_by_value(R + m[3, :], 0.0, P)
        D = tf.clip_by_value(D + m[4, :], 0.0, P)
        Ru = tf.clip_by_value(Ru + m[5, :], 0.0, P)

        s = tf.tensor_scatter_nd_add(tensor=s,
                                     indices=[[i, 0], [i, 1], [i, 2]],
                                     updates=tf.stack([A, R, D]))

        return i+1, s, S, I, A, R, D, Ru

    # populate summary with data from different days
    k, summary, *_ = tf.while_loop(
        cond=lambda k, *_: k < n_days,
        body=body,
        loop_vars=init_vars
        )

    # calculate Euclid distances between real and simulated data
    t_summary = tf.transpose(summary, perm=[2, 1, 0])
    distances = tf.norm(tf.broadcast_to(country_data_train, tf.constant(
        [args.n_samples_per_batch,
         country_data_train.shape[0], country_data_train.shape[1]],
        dtype=tf.int32)) - t_summary, axis=2)
    reduced_distances = tf.reduce_sum(distances, axis=1)
    # calculate which simulations were successful
    acceptance_vector = tf.cast(
        reduced_distances <= local_tolerance, dtype=tf.int32)

    if args.enqueue_chunk_size:
        # split simulations into chunks, iterate over each chunk, counting
        # num. accepted and enqueueing chunk to outfeed if any accepted
        n_chunk = tf.constant(args.n_samples_per_batch // int(args.enqueue_chunk_size))
        acc_chunk_shp = [n_chunk, int(args.enqueue_chunk_size)]
        acc_chunk = \
            tf.reduce_sum(tf.reshape(acceptance_vector, acc_chunk_shp), axis=1)
        param_chunk_shp = [param_vector.shape[0]] + acc_chunk_shp
        init_vars = [tf.constant(0),
                     tf.constant(0),
                     tf.reshape(param_vector, param_chunk_shp),
                     tf.reshape(reduced_distances, acc_chunk_shp),
                     acc_chunk]
        _, gain, _, _, _ = tf.while_loop(cond=lambda n, *_: tf.less(n, n_chunk),
                                         body=chunked_outfeed_enqueue,
                                         loop_vars=init_vars)
    else:
        num_accepted_samples = tf.reduce_sum(
            acceptance_vector, name="num_accepted_samples")

        # sync between replicas
        gain = ipu.cross_replica_ops.cross_replica_sum(
            num_accepted_samples, name="accumulated_sum")

        # transfer stats for simulations with at least once success
        maybe_enq = conditional_enqueue_op(params=param_vector,
                                           dists=reduced_distances,
                                           n_accs=num_accepted_samples,
                                           gain=gain)

    total_number_of_samples = accumulated_number_of_samples + gain
    return total_number_of_samples, run_number + 1, local_tolerance


@tf.function(experimental_compile=True)
def loop_collect_samples(local_samples_target, local_max_num_runs, local_tolerance):
    """Repeat batch simulations until target condition is reached."""
    a = tf.zeros([], dtype=tf.int32)  # Number of accepted samples
    n = tf.zeros([], dtype=tf.int32)  # Number of runs
    a, n, *_ = tf.while_loop(
        lambda a, n, *_:
        tf.logical_and(
            tf.less(a, local_samples_target),
            tf.less(n, local_max_num_runs)),
        build_graph, [a, n, local_tolerance])

    return a, n


def dequeue_and_postproc(time_it=False):
    """Dequeue the outfeed data stream and filter out the relevant data."""
    if time_it and not args.sparse_output:
        start_time = time.time()

    deq_out = outfeed_data.dequeue()
    deq_end_time = time.time()

    if deq_out[0].shape[0] > 0:  # Only process if something dequeued
        if args.outfeed_num_samples:
            (param_vector, reduced_distances, num_accepted_samples) = \
                deq_out
            print(f"Samples per IPU = {np.sum(num_accepted_samples, axis=0)}")
        else:
            (param_vector, reduced_distances) = deq_out
        if time_it and not args.sparse_output:
            print(f'Dequeue-only time: {deq_end_time - start_time}')

        # Filtering relevant samples
        if args.replication_factor > 1:
            s = tf.shape(param_vector)
            pv = param_vector
            param_vector = tf.reshape(
                pv, tf.concat([[s[0] * s[1]], s[2:]], axis=0))
            t = reduced_distances.shape
            rd = reduced_distances
            reduced_distances = tf.reshape(
                rd, tf.concat([[t[0] * t[1]], [t[2]]], axis=0))

        acceptance_vector = tf.cast(
            reduced_distances <= args.tolerance, dtype=tf.bool)

        t_param_vector = tf.transpose(param_vector, perm=[1, 0, 2])
        eval_param_vector = tf.boolean_mask(
            t_param_vector, acceptance_vector, axis=1)
        if time_it and not args.sparse_output:
            proc_end_time = time.time()
            print(f'Process dequeued samples time: {proc_end_time - deq_end_time}')
        return param_vector, reduced_distances, eval_param_vector
    else:
        return None, None, None


def main():
    """Warmup, timing, and stats output handling."""
    with strategy.scope():
        # Warm-up
        if not args.sparse_output:
            print("Warming up...")
        strategy.run(
            loop_collect_samples,
            [args.n_samples_target,
             tf.constant(1, dtype=tf.int32),
             args.tolerance])
        if not args.no_outfeed_ops:
            outfeed_data.dequeue()

        # Time the compute
        if not args.sparse_output:
            print("Running...")
        start_time = time.time()
        num_accepted_samples, num_runs = strategy.run(
            loop_collect_samples,
            [args.n_samples_target,
             args.max_n_runs,
             args.tolerance])
        end_time = time.time()
        samples_collected = np.int(num_accepted_samples)
        num_runs = np.int(num_runs)
        run_duration = end_time - start_time

        # Dequeue the data
        if args.no_outfeed_ops:
            start_time = end_time = time.time()
        else:
            start_time = time.time()
            param_vector, reduced_distances, eval_param_vector = \
                dequeue_and_postproc(time_it=True)
            end_time = time.time()
        deq_proc_duration = end_time - start_time
        duration = run_duration + deq_proc_duration
        if args.sparse_output:
            print(f"{duration:.3f} \t {1e3*duration/num_runs:.3f} \t "
                  f"{run_duration:.3f} \t {1e3*run_duration/num_runs:.3f}")
        else:
            print(f"Running ABC inference for {args.country}\n"
                  f"\tBatch size: {args.n_samples_per_batch}\n"
                  f"\tTolerance: {args.tolerance}"
                  f"\tTarget number of samples: {args.n_samples_target}"
                  f"\tEnqueue chunk size: {args.enqueue_chunk_size}")
            print("=========================================")
            print("IPU runs completed in {0:.3f} seconds\n".format(
                run_duration))
            print(f"Samples collected: {samples_collected:.0f}")
            print(f"Number of runs: {num_runs:.0f} "
                  f"with {args.replication_factor} replica(s)")
            print("Time per run: {0:.3f} milliseconds\n".format(
                1e3*run_duration/num_runs))

            print("Debug: Time for dequeue and processing: "
                  "{0:.3f} second\n".format(deq_proc_duration))
            print("Debug: Total Time (inc dequeue): {0:.3f} second\n".format(
                duration))
            print("Debug: Time per run (inc dequeue): "
                  "{0:.3f} milliseconds\n".format(1e3*duration/num_runs))
            if not args.no_outfeed_ops:
                print(f"param_vector.shape = {param_vector.shape}")
                print(f"reduced_distances.shape = {reduced_distances.shape}")
                print(f"eval_param_vector.shape = {eval_param_vector.shape}")

        if samples_collected < args.n_samples_target:
            raise NotImplementedError(
                "Too few iterations. Increase max_num_runs parameter.")

        if args.samples_filepath:
            # Save the accepted samples if filepath given
            np.savetxt(args.samples_filepath,
                       eval_param_vector.numpy(),
                       delimiter=",")


if __name__ == '__main__':
    main()
