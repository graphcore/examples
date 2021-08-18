# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging
import pickle
from functools import partial

# External imports
import yaml
import numpy as np
import pandas as pd
import json
# import tensorflow as tf
import tensorflow.compat.v1 as tf

# Local imports
from data_gen import get_datasets
from models import get_model
# Fix for loading Lambda layer checkpoints
from models.layers import *
from utils.optimizers import get_optimizer
# from utils.callbacks import TimingCallback
from utils.device import configure_session
from utils.argparse import ReadYaml

from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu.optimizers import CrossReplicaOptimizer
from tensorflow.python.ops import array_ops
import time


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('--output-dir', help='Override output directory')

    # Override data settings
    add_arg('--data-dir', help='Override the path to input files')
    add_arg('--n-train', type=int, help='Override number of training samples')
    add_arg('--n-valid', type=int, help='Override number of validation samples')
    add_arg('--batch-size', type=int, help='Override the batch size')
    add_arg('--n-epochs', type=int, help='Override number of epochs')
    add_arg('--apply-log', type=int, choices=[0, 1], help='Apply log transform to data')
    add_arg('--staged-files', type=int, choices=[0, 1],
            help='Specify if you are pre-staging subsets of data to local FS')

    # Override IPU settings
    add_arg('--iterations-per-loop', type=int, help="Overide number of iterations (batches) per loop on IPU.")
    add_arg('--num-ipus', type=int, help="Overide number of IPUs for replication.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-estimator', action="store_true", default=False,
                       help="Use IPUEstimator. If not specified, then tf.session with in/out feeds are used")
    group.add_argument('--data-benchmark', action='store_true', default=False,
                       help="Do data-benchmarking")

    # Hyperparameter settings
    add_arg('--conv-size', type=int, help='CNN size parameter')
    add_arg('--fc1-size', type=int, help='Fully-connected size parameter 1')
    add_arg('--fc2-size', type=int, help='Fully-connected size parameter 2')
    add_arg('--hidden-activation', help='Override hidden activation function')
    add_arg('--dropout', type=float, help='Override dropout')
    add_arg('--optimizer', help='Override optimizer type')
    add_arg('--lr', type=float, help='Override learning rate')

    # Other settings
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--print-fom', action='store_true',
            help='Print parsable figure of merit')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()


def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)


def load_config(args):
    """Reads the YAML config file and returns a config dictionary"""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    # Expand paths
    output_dir = config['output_dir'] if args.output_dir is None else args.output_dir
    config['output_dir'] = os.path.expandvars(output_dir)

    # Override data config from command line
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.n_valid is not None:
        config['data']['n_valid'] = args.n_valid
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        config['data']['n_epochs'] = args.n_epochs
    if args.apply_log is not None:
        config['data']['apply_log'] = bool(args.apply_log)
    if args.staged_files is not None:
        config['data']['staged_files'] = bool(args.staged_files)

    # Override ipu config params
    if args.iterations_per_loop is not None:
        config['ipu_config']['iterations_per_loop'] = args.iterations_per_loop
    if args.num_ipus is not None:
        config['ipu_config']['num_ipus'] = args.num_ipus

    # Hyperparameters
    if args.conv_size is not None:
        config['model']['conv_size'] = args.conv_size
    if args.fc1_size is not None:
        config['model']['fc1_size'] = args.fc1_size
    if args.fc2_size is not None:
        config['model']['fc2_size'] = args.fc2_size
    if args.hidden_activation is not None:
        config['model']['hidden_activation'] = args.hidden_activation
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr

    return config


def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)


def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))


def print_training_summary(output_dir, print_fom):
    history = load_history(output_dir)
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])
        # Figure of merit printing for HPO parsing
        if print_fom:
            print('FoM:', history['val_loss'].loc[best])


def get_ipu_options(cosmoflow_config):

    if cosmoflow_config['ipu_config']['num_ipus'] > 1:
        ipu_options = ipu.utils.create_ipu_config(
            use_poplar_text_report=False,
        )
    else:
        ipu_options = ipu.utils.create_ipu_config(
            use_poplar_text_report=False,
        )

    ipu_options = ipu.utils.set_ipu_model_options(
        ipu_options, compile_ipu_code=True)

    ipu_options = ipu.utils.auto_select_ipus(
        ipu_options,
        num_ipus=cosmoflow_config['ipu_config']['num_ipus'])

    return ipu_options


def create_ipu_estimator(cosmoflow_config):

    ipu_options = get_ipu_options(cosmoflow_config)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        # It seems a counter of 1 is not supported.
        iterations_per_loop=cosmoflow_config['ipu_config']['iterations_per_loop'],
        ipu_options=ipu_options,
        num_replicas=cosmoflow_config['ipu_config']['num_ipus']
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        model_dir=cosmoflow_config['output_dir']
    )

    model_fn = partial(get_model_fn, cosmoflow_config=cosmoflow_config)

    return ipu.ipu_estimator.IPUEstimator(
        config=config,
        model_fn=model_fn,
        params={"learning_rate": 0.5},
    )


def get_model_fn(features, labels, mode, params, cosmoflow_config):
    """model definition"""
    model = get_model(**cosmoflow_config['model'])
    outputs = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

    train_config = cosmoflow_config['train']
    loss_name = train_config['loss']
    if loss_name == "mse":
        loss = tf.losses.mean_squared_error(labels=labels, predictions=outputs)
    else:
        raise NotImplementedError("loss: %s" % loss_name)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = outputs
        eval_metric_ops = {
            "mae": tf.metrics.mean_absolute_error(
                labels=labels, predictions=predictions),
        }
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
        if cosmoflow_config['ipu_config']['num_ipus'] > 1:
            optimizer = CrossReplicaOptimizer(optimizer)
        train_op = optimizer.minimize(loss=loss)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
    raise NotImplementedError(mode)


def get_input_fn(data_config):
    datasets = get_datasets(**data_config)
    return datasets["train_dataset"]


class PerformanceHook(tf.train.StepCounterHook):
    def __init__(self, perf_fp, every_n_steps, batch_size):
        self.perf_fp = perf_fp
        self.average_batches_per_sec = []
        self.batch_size = batch_size
        super(PerformanceHook, self).__init__(every_n_steps=every_n_steps)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        batches_per_sec = elapsed_steps / elapsed_time
        self.average_batches_per_sec.append(batches_per_sec)

    def end(self, session):
        header_string = 'Iteration, Batches/Second, Samples/Second'
        with open(self.perf_fp, 'w') as f:
            f.write(header_string + '\n')
            print(header_string)
            for loop_idx, bps in enumerate(self.average_batches_per_sec):
                out_string = '{}, {}, {}'.format(loop_idx, bps, bps * self.batch_size)
                f.write(out_string + '\n')
                print(out_string)


def train_with_ipu_estimator(input_fn, cosmoflow_config):

    ipu_estimator = create_ipu_estimator(cosmoflow_config)

    # Training progress is logged as INFO, so enable that logging level
    tf.logging.set_verbosity(tf.logging.INFO)

    logging.info('Warm up')
    ipu_estimator.train(
        input_fn=input_fn,
        steps=cosmoflow_config['ipu_config']['iterations_per_loop']
    )

    logging.info('Time measured run')
    t0 = time.time()

    data_config = cosmoflow_config['data']

    perf_fp = os.path.join(cosmoflow_config['output_dir'], 'estimator_throughput.txt')
    effective_bs = data_config["batch_size"] * cosmoflow_config['ipu_config']['num_ipus']
    hooks = [PerformanceHook(perf_fp=perf_fp, every_n_steps=1, batch_size=effective_bs)]

    # remember that effective batch-size is batch-size X num_ipus
    num_steps = ((data_config["n_epochs"] * data_config["n_train"]) // effective_bs)

    ipu_estimator.train(
        input_fn=input_fn,
        steps=num_steps,
        hooks=hooks
    )
    t1 = time.time()
    duration_seconds = t1 - t0

    logging.info("Took {:.2f} minutes".format(duration_seconds / 60))
    samples_per_second = data_config["n_epochs"] * data_config["n_train"] / duration_seconds
    print("Took {:.2f} minutes, i.e. {:.0f} samples per second for batch-size {} and no. IPUs = {}".format(
        duration_seconds / 60,
        samples_per_second,
        cosmoflow_config['data']['batch_size'],
        cosmoflow_config['ipu_config']['num_ipus']))

    # Finalize
    logging.info('All done!')

    return


def train_with_session(input_fn, cosmoflow_config):

    with tf.device('cpu'):
        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(input_fn(),  # difference in tf.dataset construction changes throughput
                                                           feed_name="training_infeed",
                                                           replication_factor=cosmoflow_config['ipu_config']['num_ipus'])

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue('outfeed',
                                                          replication_factor=cosmoflow_config['ipu_config']['num_ipus'])

    def cosmoflow_training_loop():

        def body(loss, features, labels):
            with tf.variable_scope("MainGraph"):
                model = get_model(**cosmoflow_config['model'])
                outputs = model(features, training=True)
            train_config = cosmoflow_config['train']
            loss_name = train_config['loss']
            if loss_name == "mse":
                loss = tf.losses.mean_squared_error(labels=labels, predictions=outputs)
            else:
                raise NotImplementedError("loss: %s" % loss_name)

            optimizer = tf.train.GradientDescentOptimizer(cosmoflow_config['optimizer']['lr'])
            if cosmoflow_config['ipu_config']['num_ipus'] > 1:
                optimizer = CrossReplicaOptimizer(optimizer)
            train_op = optimizer.minimize(loss=loss)
            with tf.control_dependencies([train_op]):
                return loss, outfeed_queue.enqueue(loss)

        loss = 0.0
        return ipu.loops.repeat(cosmoflow_config['ipu_config']['iterations_per_loop'],
                                body, [loss], infeed_queue)

    # Compile model
    with ipu.scopes.ipu_scope('/device:IPU:0'):
        res = ipu.ipu_compiler.compile(cosmoflow_training_loop, inputs=[])

    dequeue_outfeed = outfeed_queue.dequeue()

    ipu_options = get_ipu_options(cosmoflow_config)

    ipu.utils.configure_ipu_system(ipu_options)
    ipu.utils.move_variable_initialization_to_cpu()

    data_config = cosmoflow_config['data']
    # remember that effective batch-size is batch-size X num_ipus
    # also note that num_loops is different from num_steps given to IPUEstimator
    num_loops = ((data_config["n_epochs"] * data_config["n_train"]) //
                 (data_config["batch_size"] * cosmoflow_config['ipu_config']['num_ipus'] *
                  cosmoflow_config['ipu_config']['iterations_per_loop']))

    with tf.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(tf.global_variables_initializer())

        # Warm up
        print("Compiling and Warmup...")
        start = time.time()
        sess.run(res)
        duration = time.time() - start
        print("Duration: {:.3f} seconds\n".format(duration))
        print("Executing...")
        losses = []
        average_batches_per_sec = []
        start = time.time()
        for i in range(num_loops):
            t0 = time.time()
            sess.run(res)
            local_losses = sess.run(dequeue_outfeed)
            duration = time.time() - t0
            average_batches_per_sec.append(cosmoflow_config['ipu_config']['iterations_per_loop'] / duration)
            report_string = "{:<7.3} sec/itr.".format(duration)
            print(report_string)
            losses.append(local_losses)

        t1 = time.time()
        duration_seconds = t1 - start

        logging.info("Took {:.2f} minutes".format(duration_seconds / 60))
        print('Iteration, Batches/Second, Samples/Second')
        for loop_idx, bps in enumerate(average_batches_per_sec):
            print('{}, {}, {}'.format(loop_idx, bps,
                                      bps * data_config["batch_size"] * cosmoflow_config['ipu_config']['num_ipus']))

        samples_per_second = np.mean(average_batches_per_sec) * data_config["batch_size"] * cosmoflow_config['ipu_config']['num_ipus']
        print("Took {:.2f} minutes, i.e. {:.0f} samples per second for batch-size {} and no. IPUs = {}".format(
            duration_seconds / 60,
            samples_per_second,
            cosmoflow_config['data']['batch_size'],
            cosmoflow_config['ipu_config']['num_ipus']))

        # Finalize
        logging.info('All done!')

    return


def data_benchmark(input_fn):

    benchmark_op = ipu.dataset_benchmark.dataset_benchmark(input_fn(), 10, 10000, print_stats=True)

    with tf.Session() as sess:
        json_string = sess.run(benchmark_op)
        json_data = json.loads(json_string[0])
        bw_data = json_data['epochs']
        print('Epoch, Bandwidth(GB/s), Samples/Second')
        for epoch, ep_data in enumerate(bw_data):
            print('{}, {}, {}'.format(epoch, ep_data['bandwidth'], ep_data['elements_per_second']))

    return


def main():
    """Main function"""

    # Initialization
    args = parse_args()
    cosmoflow_config = load_config(args)
    os.makedirs(cosmoflow_config['output_dir'], exist_ok=True)
    config_logging(verbose=args.verbose)
    logging.info('Configuration: %s', cosmoflow_config)

    # Configure dataset
    data_config = cosmoflow_config['data']
    input_fn = partial(get_input_fn, data_config)

    # Save configuration to output directory
    save_config(cosmoflow_config)

    if args.data_benchmark:
        data_benchmark(input_fn)
    elif args.use_estimator:
        train_with_ipu_estimator(input_fn, cosmoflow_config)
    else:
        train_with_session(input_fn, cosmoflow_config)


if __name__ == '__main__':
    main()
