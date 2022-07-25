# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from typing import Iterable, List, Optional

import libpvti as pvti
import tensorflow as tf
import tensorflow.keras.backend as K
from absl import logging
from tensorflow.keras import layers
from tensorflow.python import ipu

import hparams_config
from ipu_automl_io import (
    postprocess_predictions,
    preprocess_normalize_image,
    visualise_detections,
)
from ipu_nms import (
    ipu_postprocessing,
    postprocess_onchip_nms_outputs,
)
from ipu_utils import (
    create_app_json,
    preload_fp32_weights,
    load_weights_into_model,
    set_or_add_env,
)
from ipu_utils.arguments import parse_args
from ipu_utils.benchmark import BenchmarkResult
from ipu_utils.config import (
    ipu_configure,
    ipu_engine_options,
)
from ipu_utils.dataset import (
    get_dataset,
    input_tensor_shape,
)
from tf2 import efficientdet_keras

channel = pvti.createTraceChannel("StepTraceChannel")


@tf.function(experimental_compile=True)
def predict_loop(iterator: Iterable,
                 steps_per_execution: int,
                 outfeed: ipu.ipu_outfeed_queue.IPUOutfeedQueue,
                 model: tf.keras.Model):
    for _ in tf.range(steps_per_execution):
        features = next(iterator)
        predictions = model(features, training=False)
        outfeed.enqueue(predictions)


def create_config(args: argparse.Namespace):
    config = hparams_config.get_efficientdet_config(args.model_name)
    config.is_training_bn = False
    config.nms_configs.score_thresh = 0.4
    config.nms_configs.max_output_size = 100
    config.nms_configs.method = "hard"
    config.override(args.hparams)
    config.strategy = "ipu"
    if args.image_size is not None:
        config.image_size = args.image_size
    return config


def main(args: argparse.Namespace):
    config = create_config(args)

    create_app_json(args, config)

    ipu_configure(args)
    in_shape = (args.micro_batch_size, ) + \
        input_tensor_shape(args, config.image_size)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        # If the model precision != FP32, we need to load the FP32 weights before we set the Keras backend precision
        # so we can cast them (otherwise we can't load all weights from the checkpoint)
        # This initial load must be done outside of the IPU strategy.
        fp32_weights = None
        if args.model_precision != tf.float32 and not args.random_weights:
            logging.info("Loading FP32 weights")
            fp32_weights = preload_fp32_weights(config, in_shape, args.model_dir)

        logging.info("Creating the dataset...")
        dataset, scales, raw_imgs = get_dataset(args, config.image_size)

        if args.model_precision == tf.float16:
            K.set_floatx('float16')

        logging.info("Created")
        iterator = iter(dataset)

        def model_fn(in_shape, training=False):
            inputs = layers.Input(
                in_shape[1:], batch_size=args.micro_batch_size, dtype=args.io_precision)
            cast_input = preprocess_normalize_image(
                inputs, args.model_precision)

            detnet = efficientdet_keras.EfficientDetNet(config=config)
            outputs = detnet(cast_input, training=training)

            # If we need FP16 weights, we'll take the FP32 weights loaded earlier and cast
            # them, then set the weights here. Otherwise we'll just load as usual.
            if not args.random_weights:
                logging.info("Loading weights into model")
                load_weights_into_model(args, detnet, fp32_weights)
                logging.info("Done")

            if args.onchip_nms:
                outputs = ipu_postprocessing(config, outputs, scales)

            return inputs, outputs

        model = tf.keras.Model(*model_fn(in_shape, False))
        model.build(in_shape)
        model.compile(steps_per_execution=args.benchmark_repeats)
        model.summary(line_length=160)

        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(buffer_depth=3)

        all_results = []
        logging.info("Running prediction")
        for i in range(args.num_repeats):
            outputs = []
            with pvti.Tracepoint(channel, f"ipu_step_{i}"):
                st = time.perf_counter()
                strategy.run(predict_loop, args=(
                    iterator, args.benchmark_repeats, outfeed_queue, model))
            pre_dequeue = time.perf_counter()
            if args.benchmark_host_postprocessing:
                # Time with host-processing
                with pvti.Tracepoint(channel, f"post_processing_step_{i}"):
                    for class_outputs, box_outputs in outfeed_queue:
                        outputs.append(postprocess_predictions(config,
                                                               class_outputs,
                                                               box_outputs,
                                                               scales,
                                                               mode="global"))
            else:
                # At a minimum we should include time to dequeue the results
                with pvti.Tracepoint(channel, f"dequeing_outputs_{i}"):
                    for step_output in outfeed_queue:
                        outputs.append(step_output)

            end_time = time.perf_counter()

            step_time_ms = 1000 * (end_time - st)
            dequeue_time = 1000 * (end_time - pre_dequeue)
            logging.debug(f"Time taken: {step_time_ms:.2f} | Dequeue time: {dequeue_time:.2f} "
                          f"| Num samples: {args.micro_batch_size} | Repeats: {args.benchmark_repeats}")

            fps = 1000 * args.micro_batch_size*args.benchmark_repeats / step_time_ms
            latency = step_time_ms/(args.micro_batch_size*args.benchmark_repeats)

            # Don't include the first repeat in timings as it includes compilation
            if i > 0:
                result = BenchmarkResult()
                result.add_step(fps, latency)
                result.set_total_time(step_time_ms)
                all_results.append(result)

            print(
                f"Step {i}  | Time: {step_time_ms:.3f}ms | img/sec: {fps:.2f} | sec/img {latency:.3f}ms")

        BenchmarkResult.print_report(all_results)

        if args.output_predictions:
            # If we haven't benchmarked the post-processing, we won't have the info we need to visualise the detections yet.
            # In that case, capture it here (won't be timed in the latency figures)
            for i, step_output in enumerate(outputs):
                if not args.benchmark_host_postprocessing and not args.onchip_nms:
                    class_outputs, box_outputs = step_output
                    logging.debug("Host-Processing output: ", i)
                    det_outputs = postprocess_predictions(
                        config, class_outputs, box_outputs, scales, mode="global")
                elif args.onchip_nms:
                    det_outputs = postprocess_onchip_nms_outputs(
                        config, step_output)
                    visualise_detections(
                        args, config, raw_imgs, None, det_outputs)


if __name__ == '__main__':
    args = parse_args()
    logging.info(args)

    ipu_engine_options(args)

    if args.synthetic:
        set_or_add_env("TF_POPLAR_FLAGS",
                       " --use_synthetic_data --synthetic_data_initializer=random")

    if args.debug:
        tf.debugging.set_log_device_placement(True)
    logging.set_verbosity(args.log_level)
    main(args)
