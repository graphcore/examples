# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time
from tensorflow.python.ipu.config import IPUConfig
import tqdm
import json
import logging
import argparse
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_infeed_queue, utils, dataset_benchmark

from data_utils import make_dataset


logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser("This program benchmarks the dataset for the dynamic sparse language model")
    parser.add_argument("--source-vocab-length", type=int, default=50256, help="Length of the dataset vocab")
    parser.add_argument("--data-dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--source-sequence-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--gradient-accumulation-count", type=int, default=112, help="The gradient accumulation "
                        "count that would have been used in a pipelining scenario. This will be used for computing "
                        "the amount of batches to prefetch")
    parser.add_argument("--repeat-count", type=int, default=50, help="The batch serialization count that would have "
                        "been used in a run scenario. Will be used for computing the number of batches to prefetch")
    parser.add_argument("--pipeline", action="store_true", help="Whether to simulate the case of a pipeline. Will be used "
                        "to compute the number of batches to prefetch")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--shuffle", action="store_true", help="Enable shuffling the dataset in between each epoch")
    parser.add_argument("--epochs", type=int, default=5, help="number of benchmark epochs")
    parser.add_argument("--on-device-only", action="store_true", help="Only test infeed to device")
    parser.add_argument("--disable-dataset-cache", action='store_true', help="Disable dataset caching")
    parser.add_argument("--disable-dataset-prefetch", action='store_true', help="Disable dataset prefetching")

    return parser.parse_args()


def process_benchmark_results(results, opts):
    benchmark_results = json.loads(results[0])['epochs']
    avg_throughput = np.mean([epoch['elements_per_second'] for epoch in benchmark_results])
    bytes_throughput = np.mean([epoch['bandwidth'] for epoch in benchmark_results])
    token_throughput = avg_throughput * opts.source_sequence_length * opts.batch_size

    logging.info(f"Average throughput {avg_throughput:0.2f} elements/s "
                 f"= {token_throughput:0.2f} tokens/s = {bytes_throughput:0.2f} GB/s\n")


def make_and_run_on_device_benchmark(opts, train=True):
    name = "training" if train else "test"
    logging.info(f"Creating the {name} benchmark for running with a device")
    graph = tf.Graph()

    with graph.as_default():
        ds, num_ds, *_ = make_dataset(opts, use_synthetic_data=False, training=train)
        num_ds = num_ds // opts.batch_size
        infeed = ipu_infeed_queue.IPUInfeedQueue(
            ds, feed_name="to_device_infeed_" + name)

        def empty_loop():
            def empty_body(data_infeed):
                return tf.no_op()
            return ipu.loops.repeat(opts.repeat_count, empty_body, [], infeed)

        with ipu.scopes.ipu_scope("/device:IPU:0"):
            benchmark_op = ipu.ipu_compiler.compile(empty_loop, inputs=[])

    with tf.Session(graph=graph) as sess:
        # run a first un-monitored epoch to force compile
        sess.run(benchmark_op)
        times = []
        for _ in range(opts.epochs):
            progress = tqdm.tqdm(range(num_ds // opts.repeat_count))
            for _ in progress:
                t0 = time.perf_counter()
                sess.run(benchmark_op)
                t1 = time.perf_counter()
                times.append(t1 - t0)
        avg_time = np.mean(times)
        token_throughput = opts.source_sequence_length * opts.batch_size * opts.repeat_count / avg_time
        bytes_throughput = token_throughput * 4 / (2**30)

    logging.info(f"On device throughput: {token_throughput:0.2f} tokens/s = {bytes_throughput:0.2f} GB/s")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Parse options
    opts = parse_args()

    if not opts.on_device_only:
        logger.info("Creating training dataset, infeed queue and benchmark.")
        # Create training dataset and infeed queue
        train_set, num_train, *_ = make_dataset(opts, use_synthetic_data=False, training=True)
        num_train = num_train // opts.batch_size

        infeed_train_queue = ipu_infeed_queue.IPUInfeedQueue(
            train_set, feed_name="train_infeed")
        # Benchmark it
        infeed_perf_train = dataset_benchmark.infeed_benchmark(
            infeed_queue=infeed_train_queue,
            number_of_epochs=opts.epochs,
            elements_per_epochs=num_train,
            print_stats=False)
        ds_perf_train = dataset_benchmark.dataset_benchmark(
            dataset=train_set,
            number_of_epochs=opts.epochs,
            elements_per_epochs=num_train,
            print_stats=False,
            apply_options=True)

        logger.info("Creating test dataset, infeed queue and benchmark.")
        # Create test dataset
        test_set, num_test, *_ = make_dataset(opts, use_synthetic_data=False, training=False)
        num_test = num_test // opts.batch_size

        infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(
            test_set, feed_name="test_infeed")
        # Benchmark it
        infeed_perf_test = dataset_benchmark.infeed_benchmark(
            infeed_queue=infeed_test_queue,
            number_of_epochs=opts.epochs,
            elements_per_epochs=num_test,
            print_stats=False)
        ds_perf_test = dataset_benchmark.dataset_benchmark(
            dataset=test_set,
            number_of_epochs=opts.epochs,
            elements_per_epochs=num_test,
            print_stats=False,
            apply_options=True)

        logging.info("Starting benchmarks...\n")
        with tf.Session() as sess:
            logger.info("Benchmarking training dataset")
            train_results = sess.run(ds_perf_train)
            process_benchmark_results(train_results, opts)
            logger.info("Benchmarking training infeed")
            train_results = sess.run(infeed_perf_train)
            process_benchmark_results(train_results, opts)

            logger.info("Benchmarking test dataset")
            test_results = sess.run(ds_perf_test)
            process_benchmark_results(test_results, opts)
            logger.info("Benchmarking test infeed")
            test_results = sess.run(infeed_perf_test)
            process_benchmark_results(test_results, opts)

    # Set config
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    # Now run on device
    make_and_run_on_device_benchmark(opts, train=True)
    make_and_run_on_device_benchmark(opts, train=False)
