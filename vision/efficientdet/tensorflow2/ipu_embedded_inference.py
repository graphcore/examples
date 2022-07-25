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
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Union

import libpvti as pvti
import numpy as np
import tensorflow.compat.v1 as tf
from absl import logging
from tensorflow.python import ipu
from tensorflow.python.ipu import (
    application_compile_op,
    embedded_runtime,
)
from tensorflow.python.keras import backend as K

from utils import set_precision_policy

from ipu_automl_io import (
    preprocess_normalize_image,
)
from ipu_utils.dataset import (
    get_dataset,
    input_tensor_shape,
)
from ipu_inference import create_config
from ipu_nms import ipu_postprocessing
from ipu_utils.config import (
    ipu_configure,
    ipu_engine_options,
)
from ipu_utils import create_app_json, safe_mean
from ipu_utils.arguments import parse_args
from ipu_utils.benchmark import BenchmarkResult
from tf2 import efficientdet_keras

channel = pvti.createTraceChannel("EmbeddedEfficientDet")


def parse_embedded_rt_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-output-dir", type=str, default=None)
    return parser.parse_known_args()


def create_output_path(args):
    exec_filename = "application.poplar_exec"
    if args.exec_output_dir is None:
        tmp_dir = tempfile.mkdtemp()
        poplar_exec_filepath = Path(tmp_dir) / exec_filename
    else:
        poplar_exec_filepath = Path(
            args.exec_output_dir) / exec_filename
        poplar_exec_filepath.parent.mkdir(exist_ok=True, parents=True)
    return poplar_exec_filepath


def clean_temporary_path(tmp_dir: Path):
    shutil.rmtree(tmp_dir)


def main(args: argparse.Namespace, exec_path: Path):
    logging.set_verbosity(args.log_level)
    config = create_config(args)

    create_app_json(args, config)

    ipu_configure(args)
    in_shape = (args.micro_batch_size, ) + \
        input_tensor_shape(args, config.image_size)

    with tf.keras.backend.get_session() as sess:
        K.set_session(sess)

        if args.model_precision == tf.float16:
            set_precision_policy('float16')

        detnet = efficientdet_keras.EfficientDetNet(config=config)
        detnet.build(in_shape)
        detnet.summary(line_length=160)

        dataset, scales, _ = get_dataset(args, config.image_size)

        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(buffer_depth=3)

        def step(inputs):
            cast_input = preprocess_normalize_image(
                inputs, args.model_precision)
            outputs = detnet(cast_input, training=False)
            if args.onchip_nms:
                outputs = ipu_postprocessing(config, outputs, scales)
            return outfeed_queue.enqueue([tf.cast(o, args.model_precision) for o in outputs])

        def loop():
            return ipu.loops.repeat(args.benchmark_repeats,
                                    step,
                                    infeed_queue=infeed_queue)

        init = tf.compat.v1.initialize_all_variables()
        sess.run(init)
        sess.run(infeed_queue.initializer)

        compile_op = application_compile_op.experimental_application_compile_op(
            loop, output_path=str(exec_path), freeze_variables=True)

        sess.run(compile_op)

        inputs = []
        engine_name = args.model_name + "_engine"
        ctx = embedded_runtime.embedded_runtime_start(str(exec_path), inputs,
                                                      engine_name)

        input_placeholder = tf.placeholder(tf.uint8, shape=in_shape)
        call_result = embedded_runtime.embedded_runtime_call(
            [input_placeholder], ctx)

        with tf.Session() as sess:
            logging.info("Warmup run...")
            input_data = np.ones(in_shape, dtype=np.uint8)
            sess.run(call_result, feed_dict={input_placeholder: input_data})
            logging.info("Warmup done.")

            input_data = []
            for i in range(args.benchmark_repeats):
                input_data.append(np.ones(in_shape, dtype=np.uint8) * i)

            all_results: List[BenchmarkResult] = []
            for rep_num in range(args.num_repeats):
                result = BenchmarkResult()

                loop_start = time.perf_counter()
                with pvti.Tracepoint(channel, f"repeat_{rep_num}"):
                    for i in range(args.benchmark_repeats):
                        with pvti.Tracepoint(channel, f"session_run_{i}"):
                            st = time.perf_counter()
                            output = sess.run(call_result,
                                              feed_dict={input_placeholder: input_data[i]})
                            nd = time.perf_counter()

                        step_time_s = nd - st
                        fps = args.micro_batch_size / step_time_s
                        latency_ms = 1000 * step_time_s
                        result.add_step(fps, latency_ms)
                loop_end = time.perf_counter()
                result.set_total_time(1000 * (loop_end - loop_start))
                all_results.append(result)

                mean_throughput, mean_latency = result.get_stats(safe_mean)

                logging.info(f"End of repeat. Avg latency: {mean_latency:.2f}ms, "
                             f"Avg per-step FPS: {mean_throughput:.2f}")

            BenchmarkResult.print_report(all_results)


if __name__ == '__main__':
    ert_args, remaining_args = parse_embedded_rt_args()
    args = parse_args(remaining_args)
    logging.info(args)

    exec_path = create_output_path(ert_args)

    ipu_engine_options(args)

    if args.debug:
        tf.debugging.set_log_device_placement(True)

    main(args, exec_path)
    if ert_args.exec_output_dir is None:
        clean_temporary_path(exec_path.parent)
