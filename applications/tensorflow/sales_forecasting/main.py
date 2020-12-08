# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse
from collections import deque, namedtuple
import json
import math
import multiprocessing
import numpy as np
import os
import tensorflow.compat.v1 as tf
import time


from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import cross_replica_optimizer
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import summary_ops
from tensorflow.python.ipu.scopes import ipu_scope, ipu_shard
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

from data import MLPData
from model import MLPModel
import util


graph_builder_output = namedtuple(
    'graph_builder_output', ['loss', 'rmse_metric', 'grads_op'])


GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'saver',
                 'writer',
                 'report',
                 'mode'])


def graph_builder(
        opts,
        observed=None,
        ground_truth=None,
        learning_rate=0.001,
        mode=util.Modes.TRAIN):

    # Build the neural network
    predictions = MLPModel(opts, mode=mode)(observed)

    # Loss
    loss = opts.loss_scaling * tf.cast(tf.losses.absolute_difference(ground_truth, predictions, reduction=tf.losses.Reduction.MEAN), dtype=getattr(tf, opts.dtypes[0]))

    # Error metric
    rmse_metric = util.exp_rmspe(ground_truth, predictions)

    if mode == util.Modes.TRAIN:
        # Training
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # Wrap in a CrossReplica if we're replicating across multiple IPUs
        if opts.replication_factor > 1:
            optimizer = cross_replica_optimizer.CrossReplicaOptimizer(optimizer)
        # Batch norm variable update dependency
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Op to calculate every variable gradient
            grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        # Loss scaling
        grads = [(grad / opts.loss_scaling, var) for grad, var in grads]

        # Apply weight_decay directly to gradients
        if opts.weight_decay != 0:
            grads = [(grad + (opts.weight_decay * var), var)
                     if 'l2tag' in var.name and 'kernel' in var.name
                     else (grad, var) for grad, var in grads]

        # clip gradients
        if opts.gradient_clipping:
            grads = [(tf.clip_by_value(grad, -1., 1.), var)
                     for grad, var in grads]

        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
        return loss / opts.loss_scaling, rmse_metric, apply_grads
    elif mode == util.Modes.VALID:
        return loss / opts.loss_scaling, rmse_metric, None


def generic_graph(opts, data, trainFlag):
    graph = tf.Graph()
    training = trainFlag == util.Modes.TRAIN
    mode_name = 'training' if training else 'validation'
    batches_per_step = opts.batches_per_step if training else opts.validation_batches_per_step
    # When replicating, we divide the data stream into N streams, so we only need to do 1/N batches in each stream.
    # For this reason, batches_per_step must be a minimum of N.
    batches_per_step = int(batches_per_step / opts.replication_factor)

    with graph.as_default():
        dataset, placeholders = data.get_dataset(opts, mode=trainFlag)
        kwargs = {} if opts.replication_factor == 1 else {'replication_factor': opts.replication_factor}
        infeed = ipu_infeed_queue.IPUInfeedQueue(dataset, f"{mode_name}_dataset_infeed", **kwargs)

        with ipu_scope(f'/device:IPU:0'):
            def comp_fn():
                def body(total_loss, total_rmse, batch):
                    loss, rmse, grad_op = graph_builder(opts,
                                                        observed=batch[:, :-1],
                                                        ground_truth=tf.expand_dims(batch[:, -1], axis=1),
                                                        learning_rate=placeholders['learning_rate'] if training else None,
                                                        mode=trainFlag)
                    if not training:
                        return total_loss + loss, total_rmse + rmse
                    with tf.control_dependencies([grad_op]):
                        return total_loss + loss, total_rmse + rmse
                return loops.repeat(batches_per_step,
                                    body,
                                    [tf.constant(0, getattr(np, opts.dtypes[0]))]*2,
                                    infeed)
            outputs = ipu_compiler.compile(comp_fn, [])

        # Average them over batches per step
        avg_loss, avg_rmse = [x / batches_per_step for x in outputs]

        # Add relevant things to the tf.summary for both
        if training:
            tf.summary.scalar("loss", avg_loss)
            tf.summary.scalar("learning_rate", placeholders["learning_rate"])
        tf.summary.scalar(f"RMSPE/{mode_name}", avg_rmse)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        ipu_utils.move_variable_initialization_to_cpu()
        init = tf.global_variables_initializer()

        report = None
        if opts.compiler_report:
            if training:
                summary_ops.ipu_compile_summary('compile_summary', avg_loss)
            with tf.device('cpu'):
                print('Initializing training report...')
                report = gen_ipu_ops.ipu_event_trace()

    writer = tf.summary.FileWriter(
        opts.logs_path + f'/{mode_name}',
        graph=graph,
        flush_secs=30)

    # Attach to IPUs and configure system
    # Subprocesses must set up IPU systems in their own scopes, then use their devices as IPU:0
    if (not training and opts.multiprocessing) or training:
        config = ipu_utils.create_ipu_config(profiling=training,
                                             use_poplar_text_report=True,
                                             max_cross_replica_sum_buffer_size=10000000,
                                             max_inter_ipu_copies_buffer_size=10000000)
        if opts.select_ipus == 'AUTO':
            config = ipu_utils.auto_select_ipus(config, [opts.replication_factor])
        else:
            config = ipu_utils.select_ipus(config, [opts.select_ipus[not training]])
        config = ipu_utils.set_compilation_options(config, {"prng.enable": str(opts.prng).lower()})
        ipu_utils.configure_ipu_system(config)

    graph_outputs = ([avg_loss] if training else [avg_rmse]) + [summary]
    sess = tf.Session(graph=graph)
    return GraphOps(graph,
                    sess,
                    init,
                    graph_outputs,
                    placeholders if training else None,
                    infeed,
                    saver,
                    writer,
                    report,
                    trainFlag)

# ----------------- GENERAL TRAINING ----------------


def run(graph_op, i=0, learning_rate=None):  # Must pass a learning_rate in for training graphs. and e for valid graphs
    feed = {graph_op.placeholders["learning_rate"]: learning_rate} if graph_op.mode == util.Modes.TRAIN else {}
    start = time.time()
    outputs = graph_op.session.run(
        graph_op.ops,
        feed_dict=feed
    )
    time_taken = time.time() - start
    graph_op.writer.add_summary(outputs[-1], i)
    return outputs[:-1] + [time_taken]


def generate_report(graph):
    print(f'Generating training report... {graph.report}')
    report = graph.session.run(graph.report)
    compilation_report = ipu_utils.extract_compile_reports(report)
    execution_report = ipu_utils.extract_execute_reports(report)

    with open("report.txt", "w") as f:
        f.write(ipu_utils.extract_all_strings_from_event_trace(report))
    with open("compilation_report.json", "w") as f:
        json.dump(compilation_report, f)
    with open("execution_report.json", "w") as f:
        json.dump(execution_report, f)
    print('Reports saved to .')


def compile_graph(opts, graph):
    # Run a dummy op with the training session to initialize any TF_POPLAR_FLAGS
    if graph.mode == util.Modes.TRAIN:
        graph.session.run(graph.placeholders['learning_rate'] + 1, feed_dict={graph.placeholders['learning_rate']: 0})

    # Load weights from checkpoint
    if opts.use_init:
        try:
            print("Loaded weights from checkpoint")
            graph.saver.restore(graph.session, opts.init_path)
            graph.session.run(graph.init)
        except tf.errors.NotFoundError:
            print("No checkpoint found - creating one")
            graph.session.run(graph.init)
            if graph.mode == util.Modes.TRAIN:
                graph.saver.save(graph.session, opts.init_path)
        finally:
            graph.session.run(graph.iterator.initializer)
    else:
        graph.session.run(graph.init)
        graph.session.run(graph.iterator.initializer)
    run(graph, i=-1, learning_rate=0)


def train_process(opts, training_data, valid_data):
    # Metric calculation and init
    opts.iterations_per_epoch = training_data._size / (opts.batch_size * opts.batches_per_step)
    opts.steps_per_valid_log = math.ceil(opts.iterations_per_epoch / opts.valid_per_epoch)
    opts.iterations = math.ceil(opts.epochs * opts.iterations_per_epoch)
    assert opts.mov_mean_window < opts.iterations, "Choose a moving mean window smaller than the number of iterations. To do all iterations, set to 0"
    lr_scheduler = opts.lr_schedule_type(opts, verbose=True)
    train_logger = util.Logger(opts, mode=util.Modes.TRAIN)
    if not opts.no_validation:
        opts.validation_batches_per_step = valid_data._size // opts.validation_batch_size
        shared_history = multiprocessing.Array('d', opts.iterations)
        val_logger = util.Logger(opts, mode=util.Modes.VALID, history_array=shared_history if opts.multiprocessing else [])

    if opts.multiprocessing:
        process = util.ParallelProcess(target=validation_process, args=(opts, valid_data, lr_scheduler, val_logger))

    # Build and compile training graph
    print("Building training graph")
    train = generic_graph(opts, training_data, util.Modes.TRAIN)
    compile_graph(opts, train)

    # Build and compile validation graph if not in a separate process
    if not opts.no_validation and not opts.multiprocessing:
        valid = validation_process(opts, valid_data)

    # Training loop
    print("Begin training loop")
    for i in range(opts.iterations):
        if not opts.multiprocessing and not opts.no_validation:
            # When interleaving, run a dummy op to load the session onto the IPU before timing throughput
            train.session.run(train.ops, feed_dict={train.placeholders['learning_rate']: 0})
        # Run the graph once
        loss, batch_time = run(train, learning_rate=lr_scheduler.lr, i=i+1)

        # Aggregate and print stats
        train_logger.update(i, batch_time=batch_time, loss=loss)

        # If we're only compiling report, do so and stop at epoch 0
        if i == 0 and opts.compiler_report:
            generate_report(train)
            return

        # Validation on first, last and scheduled steps
        if not opts.no_validation and (i in [0, opts.iterations-1] or not (i+1) % opts.steps_per_valid_log):
            filepath = train.saver.save(train.session, opts.checkpoint_path)
            if opts.multiprocessing:
                process.queue.put((i + 1, filepath))
                time.sleep(0)
            else:
                valid.saver.restore(valid.session, filepath)
                if not opts.multiprocessing and not opts.no_validation:
                    # When interleaving, run a dummy op to load the session onto the IPU before timing throughput
                    valid.session.run(valid.ops)
                val_rmspe, val_batch_time = run(valid, i=i+1)
                val_logger.update(i, batch_time=val_batch_time, batch_acc=val_rmspe)

        # Schedule the learning rate based on val accuracy, but if that's not available, then training loss
        # If we're multiprocessing, then schedule inside the subprocess
        if not opts.multiprocessing:
            lr_scheduler.schedule(loss if opts.no_validation else val_rmspe, i)

    # Clean up
    train.session.close()
    if not opts.no_validation:
        if opts.multiprocessing:
            process.cleanup()
        else:
            valid.session.close()

    # Print best RMSPE
    if not opts.no_validation:
        rmspe_list = [x for x in val_logger.history[:] if x > 0]
        if rmspe_list:
            print(f'Best RMSPE: {min(rmspe_list):6.4f}')
        else:
            print("There have been no valid RMSPE results.")


def validation_process(opts, valid_data, lr_scheduler=None, val_logger=None, queue=None):
    print('Building validation graph')
    valid = generic_graph(opts, valid_data, util.Modes.VALID)
    compile_graph(opts, valid)

    if not opts.multiprocessing:
        return valid

    while True:
        step, filepath = queue.get()
        if step == -1:
            break

        valid.saver.restore(valid.session, filepath)
        val_rmspe, val_batch_time = run(valid, i=step)
        val_logger.update(step, batch_time=val_batch_time, batch_acc=val_rmspe)

        # Schedule the learning rate based on val accuracy
        lr_scheduler.schedule(val_rmspe, step)
    valid.session.close()


def preprocess_options(opts):
    # Convert comma/dot separated strings into lists
    opts.learning_rate_decay = [float(x) for x in opts.learning_rate_decay.split(',')]
    opts.learning_rate_schedule = [float(x) for x in opts.learning_rate_schedule.split(',')]
    if opts.select_ipus != 'AUTO':
        opts.select_ipus = [int(x) for x in opts.select_ipus.split(',')]
    opts.dtypes = ['float' + x for x in opts.precision.split('.')]

    # Set the LR scheduler
    opts.lr_schedule_type = [util.ManualScheduler, util.DynamicScheduler][opts.lr_schedule_type == 'dynamic']

    # Create checkpoint and log paths
    cur_time = time.strftime('%Y%m%d_%H%M%S')
    name = f"bs{opts.batch_size}-{cur_time}"
    opts.logs_path = os.path.join(opts.log_dir, f'logs-{name}')
    opts.checkpoint_path = os.path.join(opts.log_dir, f'weights-{name}/ckpt')
    opts.init_path = os.path.join(opts.log_dir, f'init-weights-rn{cur_time}/ckpt')

    # Ensure either: 1) data is passed or 2) using synthetic
    assert opts.datafolder or opts.use_synthetic_data, "You must either supply a data folder or use synthetic data with the flag --use-synthetic-data"

    # Datafolder is preprocessed to have 'train.csv' and 'val.csv'
    opts.training_data = os.path.join(opts.datafolder, 'train.csv') if not opts.use_synthetic_data else None
    opts.validation_data = os.path.join(opts.datafolder, 'val.csv') if not opts.use_synthetic_data else None

    # Ensure there are enough batches_per_step to replicate across
    assert opts.batches_per_step >= opts.replication_factor, "There must be enough batches (N) to split when replicating across N graphs."

    # Set learning rate based on base
    opts.learning_rate = (2**opts.base_learning_rate) * opts.batch_size
    return opts


def get_options():
    parser = argparse.ArgumentParser(
        description='MLP With Embeddings training in Tensorflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---------------- LR ---------------------
    group = parser.add_argument_group('LR')
    group.add_argument('--lr-plateau-patience',         type=int,   default=10,
                       help="Number of loss stagnation epochs before LR decays")
    group.add_argument('--lr-schedule-plateau-factor',  type=float, default=0.1,
                       help="Rate at which LR decays if loss stagnates")
    group.add_argument('--lr-schedule-type',            type=str,   default='dynamic',
                       help="LR schedule: 'manual' or 'dynamic'")
    group.add_argument('--base-learning-rate',          type=float, default=-6,  # Default LR: 0.015625 / bs
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group = parser.add_argument_group(' LR (Manual Scheduler)')
    group.add_argument('--learning-rate-decay',         type=str,   default="1, 1",
                       help="Learning rate decay schedule. Comma Separated ('1,0.1,0.01')")
    group.add_argument('--learning-rate-schedule',      type=str,   default="0.35",
                       help="Learning rate drop points (proportional). Comma Separated ('0.5,0.75')")
    group = parser.add_argument_group(' LR (Dynamic Scheduler)')
    group.add_argument('--no-lr-warmup',                default=True, action='store_false', dest='lr_warmup',
                       help="Turn off warming up the learning rate at start of training")
    group.add_argument('--lr-warmup-steps',             type=int,   default=5,
                       help="Number of steps to warm up learning rate")

    # -------------- DATASET -----------------
    group = parser.add_argument_group('Dataset')
    group.add_argument('-d', '--datafolder',        type=str,       default='.',
                       help="Path to compressed rossmann store sales data folder")
    group.add_argument('--use-synthetic-data',      default=False,  action='store_true',
                       help="Use synthetic data. Synthetic data is random data generated directly on the IPU as needed by the program, removing any host <-> IPU data transfers.")
    group.add_argument('--log-dir',                 type=str,       default="./log-dir",
                       help="Log and weights save directory")

    # --------------- MODEL ------------------
    group = parser.add_argument_group('Model')
    group.add_argument('--batch-size',              type=int,   default=100,
                       help="Set batch-size for training graph")
    group.add_argument('--validation-batch-size',   type=int,   default=100,
                       help="Batch-size for validation graph")
    group.add_argument('--precision',               type=str,   default="32.32",
                       help="Setting of Ops and Master datatypes ie 16.16, 16.32, 32.32")
    group.add_argument('--no-prng',                 action="store_false",   dest='prng', default=True,
                       help="Disable Stochastic Rounding")
    group.add_argument('--loss-scaling',            type=float, default=1,
                       help="Value for loss scaling, setting to 1 removes loss scaling.")
    group.add_argument('--weight-decay',            type=float, default=0.00005,
                       help="Value for weight decay bias, setting to 0 removes weight decay.")
    group.add_argument('--gradient-clipping',       action='store_true',
                       help="Clip gradients between -1 and 1.")
    group.add_argument('--replication-factor',      type=int,   default=1,
                       help="Number of IPUs to replicate the graph across for data parallelism.")

    # -------------- TRAINING ------------------
    group = parser.add_argument_group('Training')
    group.add_argument('--epochs',              type=int,   default=100,
                       help="Number of training epochs")
    group.add_argument('--select-ipus',         type=str,   default="AUTO",
                       help="Select IPUs either: AUTO or tuple of ids ('TRAIN,VALID')")
    group.add_argument('--valid-per-epoch',     type=float, default=1,
                       help="Validation steps per epoch.")
    # The rossmann dataset has 806871 elements.  In data.py, drop_remainder drops 806871 % batch_size = 71 elements (by default), leaving 806800.
    # With a default "batch size" of 100 and a default "batches per step" of 8068 the model trains on 1 epoch of data per step.
    group.add_argument('--batches-per-step',    type=int,   default=8068,
                       help="How many minibatches to perform on the device before returning to the host."
                            "When replication is set to N, the data stream is split into N streams, each doing 1/N batches."
                            "If replication-factor is set to N, this must be a minimum of N.")
    group.add_argument('--steps-per-log',       type=int,   default=1,
                       help="Log statistics every N steps.")
    validmult = parser.add_mutually_exclusive_group()  # Can only multiprocess if we're validating
    validmult.add_argument('--no-validation',   action="store_true",
                           help="Dont do any validation runs.")
    validmult.add_argument('--multiprocessing', action="store_true",
                           help="Run the validation and training graphs in separate processes.")
    group.add_argument('--use-init',            action="store_true",
                       help="Use the same weight initialization across runs.")
    group.add_argument('--compiler-report',     action="store_true",
                       help="Include a compiler report in the log")
    group.add_argument('--mov-mean-window', type=int, default=10,
                       help="Number of iterations to take the throughput moving mean over. Set to 0 for all iterations.")
    opts = parser.parse_args()
    return preprocess_options(opts)


if __name__ == '__main__':
    # Get the parsed command line options
    opts = get_options()

    # Turn off depreciation warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Display options
    log_str = ("MLP With Embeddings Training.\n"
               " Precision {precision}\n"
               " Logging to {log_dir}\n"
               " Stochastic Rounding {prng}\n"
               "Training Graph.\n"
               " Dataset {training_data}\n"
               " Batch Size {batch_size}.\n"
               " Epochs {epochs}\n"
               " Base Learning Rate 2^{base_learning_rate}\n"
               "  Learning Rate {learning_rate}\n"
               " Loss Scaling {loss_scaling}\n"
               " Weight Decay {weight_decay}\n")
    if not opts.no_validation:
        log_str += ("Validation Graph.\n"
                    " Dataset {validation_data}\n"
                    " Batch Size {validation_batch_size}\n")
    log_str += "Checkpoint Path {checkpoint_path}\n"

    print(log_str.format(**vars(opts)))

    # If the data is not already preprocessed, preprocess it
    if not opts.use_synthetic_data and not util.is_preprocessed(opts.datafolder):
        util.preprocess_data(opts.datafolder)

    print("Loading training data")
    opts.training_data = MLPData(opts, data_path=opts.training_data)
    print(f"Rows: {opts.training_data._size}")

    print("Loading evaluation data")
    opts.validation_data = MLPData(opts, data_path=opts.validation_data)
    print(f"Rows: {opts.validation_data._size}")

    # If using synthetic data, set the environment variable required
    if opts.use_synthetic_data:
        if 'TF_POPLAR_FLAGS' in os.environ:
            os.environ['TF_POPLAR_FLAGS'] += ' --use_synthetic_data --synthetic_data_initializer=random'
        else:
            os.environ['TF_POPLAR_FLAGS'] = '--use_synthetic_data --synthetic_data_initializer=random'

    # Execute
    train_process(opts, opts.training_data, opts.validation_data)
