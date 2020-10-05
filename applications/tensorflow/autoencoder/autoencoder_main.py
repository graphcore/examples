# Copyright 2020 Graphcore Ltd.
# Original paper:
# Training Deep AutoEncoders for Collaborative Filtering
# By Oleksii Kuchaiev and Boris Ginsburg
# https://arxiv.org/pdf/1708.01715.pdf

import argparse
import os
import time
from collections import namedtuple, deque

import numpy as np
import tensorflow.compat.v1 as tf
import util
from autoencoder_data import AutoencoderData
from autoencoder_model import AutoencoderModel
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import loops, ipu_infeed_queue
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.scopes import ipu_scope


def graph_builder(
        opts,
        observed_ratings,
        ground_truth=None,
        learning_rate=0.001,
        type='TRAIN'):

    # Build the encoder-decoder graph
    predictions = AutoencoderModel(opts)(observed_ratings)

    if type == 'TRAIN':
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
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            # Op to calculate every variable gradient
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))

            # Loss scaling
            grads = [(grad / opts.loss_scaling, var) for grad, var in grads]

            # Apply weight_decay directly to gradients
            if opts.weight_decay != 0:
                grads = [(grad + (opts.weight_decay * var), var)
                         for grad, var in grads]

            # clip gradients
            if opts.gradient_clipping:
                grads = [(tf.clip_by_value(grad, -1., 1.), var)
                         for grad, var in grads]
            # Op to update all variables according to their gradient
            apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

        return loss / opts.loss_scaling, rmse_metric, apply_grads
    elif type == 'VALID':
        # Loss
        mask = tf.math.sign(ground_truth)
        masked_MSEloss = tf.losses.mean_squared_error(
            ground_truth, predictions, mask)
        rmse_metric = tf.math.sqrt(masked_MSEloss)
        return rmse_metric
    else:
        return tf.constant(0), tf.constant(0), predictions


GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'saver',
                 'writer'])


def training_graph(opts, training_data):
    train_graph = tf.Graph()

    with train_graph.as_default():

        dataset, train_iterator, placeholders = training_data.get_dataset(
            opts, is_training=True)
        infeed = ipu_infeed_queue.IPUInfeedQueue(
            dataset, "training_dataset_infeed", 0)

        with ipu_scope('/device:IPU:0'):

            def comp_fn():
                def body(total_loss_, sum_rmse_metric, *args, **kwargs):
                    data_tensors = args
                    observed_ratings = data_tensors[0]
                    loss, rmse_metric, apply_grads_ = graph_builder(opts,
                                                                    observed_ratings=observed_ratings,
                                                                    learning_rate=placeholders["learning_rate"],
                                                                    type='TRAIN')
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
        tf.summary.scalar("learning_rate", placeholders["learning_rate"])
        tf.summary.scalar("RMSE/train", rmse)

        if opts.compiler_report:
            ipu_ops.ipu_compile_summary('compile_summary', loss)

        train_summary = tf.summary.merge_all()
        train_saver = tf.train.Saver()

        ipu_utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter(
        opts.logs_path + '/train',
        graph=train_graph,
        flush_secs=30)

    ipu_options = util.get_config(opts, profiling=opts.compiler_report)
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


def validation_graph(opts, valid_data):
    # Do not apply dropout during validation
    opts.apply_dropout = False

    valid_graph = tf.Graph()
    tf_device_ordinal = 0 if opts.multiprocessing else 1
    with valid_graph.as_default():
        dataset, _, _ = valid_data.get_dataset(opts, is_training=False)
        infeed = ipu_infeed_queue.IPUInfeedQueue(
            dataset, "validation_dataset_infeed", tf_device_ordinal)

        with ipu_scope('/device:IPU:{}'.format(tf_device_ordinal)):
            def comp_fn():
                def body(sum_rmse_metric, *args, **kwargs):
                    data_tensors = args
                    observed_ratings, ground_truth = tf.split(
                        data_tensors[0], num_or_size_splits=2, axis=1)
                    rmse_metric = graph_builder(opts,
                                                observed_ratings=observed_ratings,
                                                ground_truth=ground_truth,
                                                type='VALID')
                    return sum_rmse_metric + rmse_metric

                return loops.repeat(opts.validation_batches_per_step,
                                    body,
                                    [tf.constant(0, tf.float32)],
                                    infeed)

            (sum_rmse_metric,) = ipu_compiler.compile(comp_fn, [])

        # Accuracy Ops
        rmse = sum_rmse_metric / opts.validation_batches_per_step

        valid_summary = tf.summary.scalar("RMSE/validation", rmse)
        valid_saver = tf.train.Saver()

        ipu_utils.move_variable_initialization_to_cpu()
        valid_init = tf.global_variables_initializer()

    valid_writer = tf.summary.FileWriter(
        opts.logs_path + '/valid',
        graph=valid_graph,
        flush_secs=30)

    ipu_options = util.get_config(opts, False)
    if opts.multiprocessing:
        ipu_utils.configure_ipu_system(ipu_options)
    valid_sess = tf.Session(graph=valid_graph)

    return GraphOps(valid_graph,
                    valid_sess,
                    valid_init,
                    [rmse, valid_summary],
                    None,
                    infeed,
                    valid_saver,
                    valid_writer)


# ----------------- GENERAL TRAINING ----------------

def training_run(train, learning_rate):
    # Run Training
    loss, summary, accuracy = train.session.run(
        train.ops,
        feed_dict={
            train.placeholders["learning_rate"]: learning_rate
        })
    return loss, summary, accuracy


def validation_run(valid, e=0):
    # Run Validation graph. The loop.repeat is setup to execute the full test
    # batch in a single sess.run call.
    accuracy, summary = valid.session.run(valid.ops)
    valid.writer.add_summary(summary, e)
    return accuracy


def build_init(opts, training_data):
    train = training_graph(opts, training_data)
    train.session.run(train.init)
    train.saver.save(train.session, opts.init_path)


def train_process(opts, training_data, valid_data):

    if opts.multiprocessing:
        import multiprocessing
        queue = multiprocessing.Queue()

        v_process = multiprocessing.Process(
            target=validation_process, args=(
                opts, valid_data, queue))
        v_process.start()

    # --------------- OPTIONS ---------------------

    base_lr = 2 ** opts.base_learning_rate
    decay_lr = opts.learning_rate_decay
    lrs = [base_lr * opts.batch_size * decay for decay in decay_lr]
    epochs = opts.epochs
    iterations_per_epoch = training_data.size / \
        (opts.batch_size * opts.batches_per_step)
    steps_per_valid = int(iterations_per_epoch / opts.valid_per_epoch)
    iterations = int(epochs * iterations_per_epoch)
    lr_drops = [int(i * iterations) for i in opts.learning_rate_schedule]
    current_lr = lrs.pop(0)
    next_drop = lr_drops.pop(0)
    batch_accs = deque(maxlen=opts.steps_per_log)
    batch_times = deque(maxlen=opts.steps_per_log)

    # -------------- BUILD GRAPH ------------------

    train = training_graph(opts, training_data)

    # ------------- INITIALIZE SESSION -----------

    print('INITIALIZE SESSION')
    train.session.run(train.init)
    train.saver.save(train.session, opts.init_path)

    train.session.run(train.iterator.initializer)

    # --------------- BUILD VALIDATION -----------

    print('BUILD VALIDATION')
    if not opts.no_validation and not opts.multiprocessing:
        valid = validation_process(opts, valid_data)

    # ------------- TRAINING LOOP ----------------

    print('TRAINING LOOP')
    print_format = (
        "step: {step:6d}, epoch: {epoch:6.2f}, lr: {lr:6.2g}, loss: {loss:6.3f}, RMSE: {train_acc:6.3f}"
        ", users/sec: {img_per_sec:6.2f}, time: {it_time:8.6f}")

    for e in range(iterations):

        if e > next_drop:
            current_lr = lrs.pop(0)
            if len(lr_drops) > 0:
                next_drop = lr_drops.pop(0)
            else:
                next_drop = np.inf
            print("Learning_rate change to {}".format(current_lr))

        # Run Training
        start = time.time()
        loss, summary, batch_acc = training_run(train, current_lr)
        batch_time = time.time() - start

        # Calculate Stats
        batch_accs.append([batch_acc])
        batch_times.append([batch_time])

        train_acc = np.mean(batch_accs)
        avg_batch_time = np.mean(batch_times)

        epoch = float(opts.batches_per_step * opts.batch_size *
                      (e + 1)) / training_data.size

        # Print loss
        if not (e + 1) % opts.steps_per_log or e == 0:
            stats = {
                'step': e + 1,
                'epoch': epoch,
                'lr': current_lr,
                'loss': loss,
                'train_acc': train_acc,
                'it_time': avg_batch_time,
                'img_per_sec': opts.batches_per_step * opts.batch_size / avg_batch_time}

            print(print_format.format(**stats))
            # Save accuracy
            train.writer.add_summary(summary, e)

        # Eval
        if not opts.no_validation and (
                not (e + 1) % steps_per_valid or e == 0 or e + 1 == iterations):
            filepath = train.saver.save(train.session, opts.checkpoint_path)
            if opts.multiprocessing:
                queue.put((e + 1, epoch, filepath))
                time.sleep(0)
            else:
                valid.saver.restore(valid.session, filepath)
                start = time.time()
                print("Running validation...")
                accuracy = validation_run(valid, e + 1)
                valid_time = time.time() - start
                print(
                    "Validation RMSE (step {}, epoch {:6.2f}, users/sec {:6.2f}): {:6.4f}".format(
                        e +
                        1,
                        epoch,
                        opts.validation_batch_size *
                        opts.validation_batches_per_step /
                        valid_time,
                        accuracy))

    # --------------- CLEANUP ----------------
    train.session.close()
    if not opts.no_validation:
        if opts.multiprocessing:
            queue.put((-1, 0, ""))
            queue.close()
            queue.join_thread()
            v_process.join()
        else:
            valid.session.close()


def validation_process(opts, valid_data, q=None):
    # --------------- OPTIONS ---------------------

    opts.validation_batches_per_step = valid_data.size // opts.validation_batch_size

    # -------------- BUILD GRAPH ------------------

    valid = validation_graph(opts, valid_data)

    # ------------- INITIALIZE SESSION -----------

    valid.session.run(valid.iterator.initializer)
    try:
        valid.saver.restore(valid.session, opts.init_path)
    except tf.errors.NotFoundError:
        valid.session.run(valid.init)

    # --------------- OPTIONS ---------------------

    validation_iterations = valid_data.size // opts.validation_batch_size

    # -------------- COMPILE RUN ------------------

    valid.session.run(valid.ops)

    if not opts.multiprocessing:
        return valid

    while True:
        step, epoch, filepath = q.get()
        if step == -1:
            break

        # Eval
        valid.saver.restore(valid.session, filepath)
        start = time.time()
        accuracy = validation_run(valid, step)
        valid_time = time.time() - start
        print(
            "Validation RMSE (step {}, epoch {:6.2f}, users/sec {:6.2f}): {:6.4f}".format(
                step,
                epoch,
                opts.validation_batch_size *
                opts.validation_batches_per_step /
                valid_time,
                accuracy))

    valid.session.close()


def testing_process(opts, valid_data):

    path = tf.train.get_checkpoint_state(opts.testing_on_checkpoint)
    print('Checkpoint path:')
    print(path)

    # --------------- BUILD TESTING -----------

    print('BUILD TESTING')
    valid = validation_process(opts, valid_data)

    # ------------- RUN TESTING ----------------

    valid.saver.restore(valid.session, path.model_checkpoint_path)

    start = time.time()
    print("Running testing...")
    accuracy = validation_run(valid, 1)
    valid_time = time.time() - start
    print("Testing RMSE: {:6.4f}".format(accuracy))

    # --------------- CLEANUP ----------------

    valid.session.close()


def preprocess_options(opts):
    opts.learning_rate_decay = list(
        map(float, opts.learning_rate_decay.split(',')))
    opts.learning_rate_schedule = list(
        map(float, opts.learning_rate_schedule.split(',')))

    if opts.select_ipus == 'AUTO':
        opts.select_ipus = (-1, -1)
    else:
        opts.select_ipus = list(map(int, opts.select_ipus.split(',')))

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
    if not opts.validation_batch_size:
        opts.validation_batch_size = opts.batch_size

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
        '--validation-data-file',
        type=str,
        required=True,
        help="Validation data file.")
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
        '--validation-batch-size',
        type=int,
        default=None,
        help="Batch-size for validation graph. Set to (training) batch-size if not specified")
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
        help="Disable Stochastic Rounding")
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
        '--base-learning-rate',
        type=int,
        default=-16,
        help="Base learning rate exponent (2**N). blr = lr /  bs")
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
        '--epochs',
        type=int,
        default=160,
        help="Number of training epochs")
    group.add_argument(
        '--select-ipus',
        type=str,
        default="AUTO",
        help="Select IPUs either: AUTO or tuple of ids ('TRAIN,VALID')")
    group.add_argument(
        '--valid-per-epoch',
        type=float,
        default=1,
        help="Validation steps per epoch.")
    group.add_argument(
        '--steps-per-log',
        type=int,
        default=1,
        help="Log statistics every N steps.")
    group.add_argument(
        '--no-validation',
        action="store_true",
        help="Dont do any validation runs.")
    group.add_argument(
        '--multiprocessing',
        action="store_true",
        help="Run the validation and training graphs in separate processes.")
    group.add_argument(
        '--build-init',
        action="store_true",
        help="Save a weight initialization to reuse each run")
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
    group.add_argument(
        '--testing-on-checkpoint',
        type=str,
        help="Only run testing on checkpoint directory specified (example: ./log-dir/weights-bs64-rn128-20200331_160859/). Testing data path is taken from validation data file.")
    group.add_argument(
        '--compiler-report',
        action="store_true",
        help="Include a compiler report in the log")

    opts = parser.parse_args()

    return preprocess_options(opts)


if __name__ == '__main__':

    opts = get_options()

    # Large number of deprecation warnings that cannot be resolved yet.
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Display Options.
    log_str = ("Autoencoder-{size} Training\n"
               " Dataset {training_data_file}\n"
               " Precision {precision}\n"
               " Logging to {logdir}\n"
               " Stochastic Rounding {prng}\n"
               "Training Graph\n"
               " Optimizer {optimizer}\n"
               " Batch Size {batch_size}\n"
               " Epochs {epochs}\n"
               " Base Learning Rate 2^{base_learning_rate}\n"
               "  Learning Rate {learning_rate}\n")
    if opts.loss_scaling:
        log_str += " Loss Scaling {loss_scaling}\n"
    if opts.weight_decay:
        log_str += " Weight Decay {weight_decay}\n"
    if not opts.no_validation:
        log_str += ("Validation Graph\n"
                    " Dataset {validation_data_file}\n"
                    " Batch Size {validation_batch_size}\n")
    if not opts.testing_on_checkpoint:
        log_str += "Checkpoint Path {checkpoint_path}\n"

    opts.learning_rate = (2**opts.base_learning_rate) * opts.batch_size
    print(log_str.format(**vars(opts)))

    # load data
    print("Loading training data")
    training_data = AutoencoderData(data_file_name=opts.training_data_file)
    print("Users: {}".format(training_data.size))
    print("Items: {}".format(training_data.input_size))

    print("Loading evaluation data")
    valid_data = AutoencoderData(data_file_name=opts.validation_data_file,
                                 training_data=training_data)
    print("Users: {}".format(valid_data.size))
    print("Items: {}".format(valid_data.input_size))

    if training_data.input_size != valid_data.input_size:
        raise ValueError(
            'Number of items for training data and validation data must be'
            ' equal. Got {} and {}.'.format(
                training_data.input_size, valid_data.input_size))

    opts.input_size = valid_data.input_size

    if opts.build_init:
        build_init(opts, training_data)
    elif opts.testing_on_checkpoint:
        testing_process(opts, valid_data)
    else:
        train_process(opts, training_data, valid_data)
