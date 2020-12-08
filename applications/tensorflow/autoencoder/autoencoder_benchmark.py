# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# Original paper:
# Training Deep AutoEncoders for Collaborative Filtering
# By Oleksii Kuchaiev and Boris Ginsburg
# https://arxiv.org/pdf/1708.01715.pdf

import argparse
import multiprocessing
import os
import sys
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from autoencoder_data import AutoencoderData
from autoencoder_model import AutoencoderModel
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import loops, ipu_infeed_queue
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.scopes import ipu_scope

GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'saver',
                 'writer'])


class IPUGraphParams:
    def __init__(self,
                 device_index,
                 opts):

        self.base_lr = 2 ** opts.base_learning_rates[device_index]
        self.decay_lr = opts.learning_rate_decay
        self.lrs = [self.base_lr * opts.batch_size * decay
                    for decay in opts.learning_rate_decay]
        self.epochs = opts.epochs
        self.iterations_per_epoch = training_data.size / (opts.batch_size * opts.batches_per_step)
        self.iterations = int(opts.epochs * self.iterations_per_epoch)
        self.lr_drops = [int(i * self.iterations) for i in opts.learning_rate_schedule]
        self.current_lr = self.lrs.pop(0)
        self.next_drop = self.lr_drops.pop(0)
        self.batch_accs = []
        self.throughputs = []


def graph_builder(
        opts,
        observed_ratings,
        learning_rate=0.001):

    # Build the encoder-decoder graph
    predictions = AutoencoderModel(opts)(observed_ratings)

    # Loss: masked mean squared error
    mask = tf.math.sign(observed_ratings)
    masked_MSEloss = tf.losses.mean_squared_error(
        observed_ratings, predictions, mask)
    rmse_metric = tf.math.sqrt(masked_MSEloss)
    loss = opts.loss_scaling * masked_MSEloss

    # Dense re-feeding when training
    if opts.dense_refeeding:
        predictions_after_refeeding = AutoencoderModel(opts)(predictions)
        MSEloss = tf.losses.mean_squared_error(
            predictions_after_refeeding, predictions)
        loss = opts.loss_scaling * (masked_MSEloss + MSEloss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Optimizer
        if opts.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=0.9)
        # Op to calculate every variable gradient
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        # Loss scaling
        grads = [(grad / opts.loss_scaling, var) for grad, var in grads]

        # Apply weight_decay directly to gradients
        if opts.weight_decay != 0:
            grads = [(grad + (opts.weight_decay * var), var) for grad, var in grads]

        # clip gradients
        if opts.gradient_clipping:
            grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

    return loss / opts.loss_scaling, rmse_metric, apply_grads


def training_graph(opts, training_data, device_index=0, learning_rate=0.001):
    train_graph = tf.Graph()

    with train_graph.as_default():

        dataset, _, placeholders = training_data.get_dataset(
            opts, is_training=True)
        infeed = ipu_infeed_queue.IPUInfeedQueue(
            dataset, "training_dataset_infeed{0}".format(device_index), 0)

        with ipu_scope('/device:IPU:0'):

            def comp_fn():
                def body(total_loss_, sum_rmse_metric, *args):
                    data_tensors = args
                    observed_ratings = data_tensors[0]
                    loss, rmse_metric, apply_grads_ = graph_builder(opts,
                                                                    observed_ratings=observed_ratings,
                                                                    learning_rate=placeholders["learning_rate"])
                    with tf.control_dependencies([apply_grads_]):
                        return total_loss_ + loss, sum_rmse_metric + rmse_metric

                return loops.repeat(opts.batches_per_step,
                                    body,
                                    [tf.constant(0, tf.float32),
                                     tf.constant(0, tf.float32)],
                                    infeed)

            total_loss, sum_rmse_metric = ipu_compiler.compile(comp_fn, [])

        rmse = sum_rmse_metric / opts.batches_per_step
        loss = total_loss / opts.batches_per_step

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("RMSE/train", rmse)

        train_summary = tf.summary.merge_all()
        train_saver = tf.train.Saver()

        ipu_utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter(
        opts.logs_path + '/train{0}'.format(device_index),
        graph=train_graph,
        flush_secs=30)

    ipu_options = ipu_utils.create_ipu_config(profiling=False)
    ipu_options = ipu_utils.set_floating_point_behaviour_options(ipu_options,
                                                                 inv=opts.fp_exceptions,
                                                                 div0=opts.fp_exceptions,
                                                                 oflo=opts.fp_exceptions,
                                                                 esr=opts.prng,
                                                                 nanoo=True)
    ipu_options = ipu_utils.auto_select_ipus(ipu_options, 1)
    ipu_utils.configure_ipu_system(ipu_options)

    train_sess = tf.Session(graph=train_graph)

    return GraphOps(train_graph,
                    train_sess,
                    train_init,
                    [loss, train_summary, rmse],
                    placeholders,
                    infeed,
                    train_saver,
                    train_writer)


# ------------------- Compile graph before running training
def compile_graph(opts, graph):
    # Run dummy op with the training session to initialize any TF_POPLAR_FLAGS
    graph.session.run(graph.placeholders['learning_rate'] + 1,
                      feed_dict={graph.placeholders['learning_rate']: 0})

    # Initialize and run with step 0 to force graph to compile
    graph.session.run(graph.init)
    graph.saver.save(graph.session, opts.init_path)
    graph.session.run(graph.iterator.initializer)
    training_run(graph, learning_rate=0)


# ----------------- GENERAL TRAINING ----------------
def training_run(train, learning_rate):
    # Run Training
    loss, summary, accuracy = train.session.run(
        train.ops,
        feed_dict={
            train.placeholders["learning_rate"]: learning_rate
        })
    return loss, summary, accuracy


# ------------ CONSTRUCT GRAPHS AND SET UP ----------
# ---------------- MULTITHREADING QUEUE -------------

def train_process_init(opts, training_data):

    ipuGraphParams = []
    for device_index in range(opts.num_ipus):
        graphParams = IPUGraphParams(device_index, opts)
        ipuGraphParams.append(graphParams)

    # ------------- Set up multiprocessing ------------
    queue = multiprocessing.Queue()

    throughputs = []
    training_processes = []

    for device_index in range(opts.num_ipus):
        tp = multiprocessing.Process(target=train_process,
                                     args=(ipuGraphParams[device_index],
                                           opts,
                                           training_data,
                                           device_index,
                                           queue))
        training_processes.append(tp)
        tp.start()

    # Wait for 30 seconds to give processes a chance to spawn and attempt to attach to IPU
    time.sleep(30)

    # Check for any failed processes
    for index, tp in enumerate(training_processes):
        # Subprocess failed.  Probably failed to attach to IPU
        if tp.exitcode == 1:
            training_processes.pop(index)
            tp.terminate()

    # Stop benchmark if num_ipus available != num_ipus requested
    if len(training_processes) != opts.num_ipus:
        print("\nERROR: Tried to attach to {0} IPUs but only {1} were available.  Stopping benchmark.\n".format(opts.num_ipus, len(training_processes)))
        for tp in training_processes:
            tp.terminate()
        return

    for tp in training_processes:
        throughput = queue.get()
        throughputs.append(throughput)

    for tp in training_processes:
        tp.join()

    aggregate_avg_throughput = np.sum([throughputs[i]["avg_users_per_sec"] for i in range(opts.num_ipus)])
    aggregate_max_throughput = np.sum([throughputs[i]["max_users_per_sec"] for i in range(opts.num_ipus)])

    for tp in throughputs:
        print("\nDevice {0}: Learning Rate = {1}, Loss = {2}, Accuracy = {3},"
              " Average users/sec = {4}, Max users/sec = {5}".format(tp["device"],
                                                                     tp["original_lr"],
                                                                     tp["loss"],
                                                                     tp["train_acc"],
                                                                     tp["avg_users_per_sec"],
                                                                     tp["max_users_per_sec"]))

    print("\nAggregate average throughput over {0} IPU devices = {1} users/sec".format(opts.num_ipus, int(aggregate_avg_throughput)))
    print("\nAggregate maximum throughput over {0} IPU devices = {1} users/sec".format(opts.num_ipus, int(aggregate_max_throughput)))


def train_process(graphParams, opts, training_data, device_index, q=None):

    # ------------- TRAINING LOOP ----------------
    print("Device {0}: SETTING UP THE GRAPH".format(device_index))

    original_lr = graphParams.base_lr * opts.batch_size

    train = training_graph(opts, training_data, device_index, graphParams.base_lr)
    compile_graph(opts, train)

    print('Device {0}: TRAINING LOOP'.format(device_index))
    print_format = (
        "device: {device:6d}, step: {step:6d}, epoch: {epoch:6.2f}, lr: {lr:6.2g}, original_lr: {original_lr:6.2g}, loss: {loss:6.3f}, RMSE: {train_acc:6.3f}"
        ", latest users/sec: {latest_users_per_sec:6d}, avg users/sec: {avg_users_per_sec:6d}, max users/sec: {max_users_per_sec:6d}, time: {it_time:8.6f}")

    # Calculate mean throughput as total users / total time
    tot_time = 0.
    tot_users_iterated = 0.

    for e in range(graphParams.iterations):

        if e > graphParams.next_drop:
            graphParams.current_lr = graphParams.lrs.pop(0)
            if len(graphParams.lr_drops) > 0:
                graphParams.next_drop = graphParams.lr_drops.pop(0)
            else:
                graphParams.next_drop = np.inf
            print("Device {0}: Learning_rate change to {1}".format(device_index, graphParams.current_lr))

        # Run Training
        batch_start = time.time()
        loss, summary, batch_acc = training_run(train, graphParams.current_lr)
        batch_time = time.time() - batch_start
        tot_time += batch_time

        # Calculate Stats
        graphParams.batch_accs.append(batch_acc)
        tot_users_iterated += opts.batches_per_step * opts.batch_size
        graphParams.throughputs.append(opts.batches_per_step * opts.batch_size / batch_time)

        # Print metrics such as loss, throughput, accuracy etc
        if not (e + 1) % opts.steps_per_log or e == 0 or e == graphParams.iterations - 1:

            train_acc = np.mean(graphParams.batch_accs)
            max_throughput = max(graphParams.throughputs)
            avg_throughput = tot_users_iterated / tot_time

            epoch = float(opts.batches_per_step * opts.batch_size *
                          (e + 1)) / training_data.size

            stats = {
                'device': device_index,
                'step': e + 1,
                'epoch': epoch,
                'lr': graphParams.current_lr,
                'original_lr': original_lr,
                'loss': loss,
                'train_acc': train_acc,
                'it_time': batch_time,
                'latest_users_per_sec': int(opts.batches_per_step * opts.batch_size / batch_time),
                'avg_users_per_sec': int(avg_throughput),
                'max_users_per_sec': int(max_throughput)}

            print(print_format.format(**stats))
            # Save accuracy
            train.writer.add_summary(summary, e)

    # --------------- CLEANUP ----------------
    train.session.close()

    # Put the metrics somewhere accessible from the main thread
    q.put(stats)


def preprocess_options(opts):
    # Allow learning rate sweeps in the form 'x..y'
    if ".." in opts.base_learning_rates:
        rs = opts.base_learning_rates.split("..")
        lb, ub = float(rs[0]), float(rs[1])
        lrs = []
        for i in range(opts.num_ipus):
            if opts.num_ipus > 1:
                p = (ub - lb) * i / (opts.num_ipus - 1)
            else:
                p = 0
            lrs.append(str(lb + p))
        opts.base_learning_rates = ','.join(lrs)

    opts.base_learning_rates = list(
        map(float, opts.base_learning_rates.split(',')))
    opts.base_learning_rates = [-1. * blr for blr in opts.base_learning_rates]
    opts.learning_rate_decay = list(
        map(float, opts.learning_rate_decay.split(',')))
    opts.learning_rate_schedule = list(
        map(float, opts.learning_rate_schedule.split(',')))

    if len(opts.base_learning_rates) != opts.num_ipus:
        print("Error: Must have as many learning rate values as IPU devices!")
        sys.exit(1)

    dataset_suffix = os.path.basename(opts.training_data_file)
    batch_size_dict = {'3m_train.txt': [64, 128],
                       '6m_train.txt': [64, 256],
                       '1y_train.txt': [32, 256],
                       'full_train.txt': [8, 512]}

    if not opts.batch_size:
        if dataset_suffix in batch_size_dict.keys():
            opts.batch_size = batch_size_dict[dataset_suffix][0]
        else:
            raise Exception("\n\nError: Unrecognised training dataset file name: \"{}\"\n"
                            "Either set batch and graph sizes manually, or ensure "
                            "training dataset file name is one of the following:"
                            "\n\t{}, {}, {}, {}\n".format(dataset_suffix, *batch_size_dict))
    if not opts.size:
        if dataset_suffix in batch_size_dict.keys():
            opts.size = batch_size_dict[dataset_suffix][1]
        else:
            raise Exception("\n\nError: Unrecognised training dataset file name: \"{}\"\n"
                            "Either set batch and graph sizes manually, or ensure "
                            "training dataset file name is one of the following:"
                            "\n\t{}, {}, {}, {}\n".format(dataset_suffix, *batch_size_dict))

    # Logs and checkpoint paths
    name = "bs{}-rn{}-{}".format(opts.batch_size,
                                 opts.size, time.strftime('%Y%m%d_%H%M%S'))
    opts.logs_path = os.path.join(opts.logdir, 'logs-{}'.format(name))
    opts.checkpoint_path = os.path.join(
        opts.logdir, 'weights-{}/ckpt'.format(name))
    opts.init_path = os.path.join(opts.logdir,
                                  'init-weights-rn{}/ckpt'.format(opts.size))
    # Create log dir
    os.makedirs(opts.logs_path, exist_ok=True)

    return opts


def get_options():
    parser = argparse.ArgumentParser(
        description='Autoencoder for recommendations training in TensorFlow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------- DATASET ------------------
    group = parser.add_argument_group('Dataset')
    group.add_argument(
        '--training-data-file',
        type=str,
        required=True,
        help="Training data file.")
    group.add_argument(
        '--pipeline-num-parallel',
        type=int,
        default=32,
        help="Number of users to preprocess in parallel")

    # -------------- MODEL ------------------
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--size',
        type=int,
        choices=[128, 256, 512],
        default=None,
        help='First layer size: automatically set depending '
             'on training dataset file name if not set manually. '
             '(128 for 3-month dataset, 256 for 6-month and 1-year, 512 for full).')
    group.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training graph: automatically set '
             'depending on training dataset file name if not set manually. '
             '(64 for 3-month, 6-month and 1-year datasets, 8 for full).')
    group.add_argument(
        '--no-selu',
        action="store_false",
        dest="apply_selu",
        default=True,
        help='Do not apply SELU activation.')
    group.add_argument(
        '--no-dropout',
        action="store_false",
        dest="apply_dropout",
        default=True,
        help='Do not apply dropout.')
    group.add_argument(
        '--no-dense-refeeding',
        action="store_false",
        dest="dense_refeeding",
        default=True,
        help='Remove dense refeeding.')
    group.add_argument(
        '--precision',
        type=str,
        default="16.16",
        help="Setting of Ops and Master datatypes ie 16.16, 16.32, 32.32")
    group.add_argument(
        '--no-prng',
        action="store_false",
        dest='prng',
        default=True,
        help="Disable Stochastic Rounding (Enabled by default)")
    group.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help="Value for weight decay bias, setting to 0 removes weight decay.")
    group.add_argument(
        '--loss-scaling',
        type=float,
        default=1e3,
        help="Value for loss scaling, setting to 1 removes loss scaling.")
    group.add_argument(
        '--gradient-clipping',
        action='store_true',
        help="Clip gradients between -1 and 1.")

    # -------------- TRAINING ------------------
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--optimizer',
        type=str,
        choices=['Momentum', 'SGD'],
        default='Momentum',
        help="Optimizer: Momentum (default) or SGD")
    group.add_argument(
        '--base-learning-rates',
        type=str,
        default='12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20',
        help="Negative base learning rate exponents. Comma Separated ('12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20') or linearly spaced range ('16..20')")
    group.add_argument(
        '--learning-rate-decay',
        type=str,
        default="1,0.1,0.01",
        help="Learning rate decay schedule. Comma Separated ('1,0.1,0.01')")
    group.add_argument(
        '--learning-rate-schedule',
        type=str,
        default="0.5,0.75",
        help="Learning rate drop points (proportional). Comma Separated ('0.5,0.75')")
    group.add_argument(
        '--num-ipus',
        type=int,
        default=16,
        help="Number of IPU devices")
    group.add_argument(
        '--epochs',
        type=int,
        default=160,
        help="Number of training epochs")
    group.add_argument(
        '--steps-per-log',
        type=int,
        default=40,
        help="Log statistics every N steps.")
    group.add_argument(
        '--fp-exceptions',
        action="store_true",
        help="Turn on floating point exceptions")
    group.add_argument(
        '--batches-per-step',
        type=int,
        default=250,
        help="How many minibatches to perform on the device before returning to the host.")
    group.add_argument(
        '--logdir',
        type=str,
        default="./logdir",
        help="Log and weights save directory")

    opts = parser.parse_args()

    return preprocess_options(opts)


if __name__ == '__main__':

    opts = get_options()

    # Large number of deprecation warnings that cannot be resolved yet.
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Display Options.
    log_str = ("Autoencoder-{size} Training\n"
               " Running on {num_ipus} IPUs\n"
               " Dataset {training_data_file}\n"
               " Precision {precision}\n"
               " Logging to {logdir}\n"
               " Stochastic Rounding {prng}\n"
               "Training Graph\n"
               " Optimizer {optimizer}\n"
               " Batch Size {batch_size}\n"
               " Epochs {epochs}\n"
               " Base Learning Rates 2^{base_learning_rates}\n")
    if opts.loss_scaling:
        log_str += " Loss Scaling {loss_scaling}\n"
    if opts.weight_decay:
        log_str += " Weight Decay {weight_decay}\n"

    print(log_str.format(**vars(opts)))

    # load data
    print("Loading training data")
    training_data = AutoencoderData(data_file_name=opts.training_data_file)

    print("Users: {}".format(training_data.size))
    print("Items: {}".format(training_data.input_size))

    opts.input_size = training_data.input_size

    train_process_init(opts, training_data)
