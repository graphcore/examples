# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import random
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
from fastspeech2 import build_inference_model
from dataloader import LJSpeechCharLevelDataset
from utils import create_ipu_config
from options import make_global_options
from ckpt_utils import set_weights


def setup_logger():
    logFormatter = logging.Formatter(
        '%(asctime)s.%(msecs)06d: %(levelname)-1.1s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger("FastSpeech2 Inference")
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    # prevent messages from being propagated to the root logger
    logger.propagate = False
    return logger


@tf.function(experimental_compile=True)
def inference_step(iterator, model, outfeed, steps_per_execution):
    for _ in tf.range(steps_per_execution):
        inputs = next(iterator)
        predictions = model(inputs)
        outfeed.enqueue(predictions)
    return predictions


def run_model(opts):
    logger = setup_logger()
    if opts["precision"] == "16":
        policy = tf.keras.mixed_precision.Policy("float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    num_ipus_per_replica = 1
    num_ipus = num_ipus_per_replica * int(opts["replicas"])
    assert num_ipus & (
        num_ipus-1) == 0, f"You‘re trying to apply {num_ipus} IPUs, but we only support to apply the power of 2 IPUs."
    # Set up the IPU system.
    cfg = create_ipu_config(
        available_memory_proportion=opts["available_memory_proportion"],
        num_required_ipus=num_ipus,
        partials_type=opts["partials_type"],
        fp_exceptions=opts["fp_exceptions"],
        enable_stochastic_rounding=False,
        num_io_tiles=0)

    val_datasets = LJSpeechCharLevelDataset(opts, is_train=False)
    if opts["steps_per_epoch"] is not None:
        steps_per_execution = opts["steps_per_epoch"]
    else:
        steps_per_execution = len(
            val_datasets) // (opts["batch_size"] * opts["replicas"])
    (input_id, duration, _, _), _ = next(iter(val_datasets()))
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        infer_datasets = iter(val_datasets.get_inference_data())
        fastspeech2 = tf.keras.Model(*build_inference_model(opts))
        # Set the infeed and outfeed options.
        fastspeech2.compile(steps_per_execution=steps_per_execution)
        fastspeech2.summary()
        # dummy run to get the initial weights of model
        _ = fastspeech2(input_id)
        if opts["init_checkpoint"] is not None:
            # set mode to `gpu` to load weights pretrained on GPU.
            fastspeech2 = set_weights(
                opts["init_checkpoint"], fastspeech2, mode="ipu")
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        total_nb_samples = steps_per_execution * \
            opts["batch_size"] * opts["replicas"]
        logger.info(
            f"**** Steps per execution: {steps_per_execution}, len(datasets)={len(val_datasets)}")
        tputs = []
        latencies = []
        for cur_step in range(opts["epochs"]):
            start = time.time()
            predictions = strategy.run(inference_step, args=[
                                       infer_datasets, fastspeech2, outfeed_queue, steps_per_execution])
            duration = time.time() - start
            tput = total_nb_samples / duration
            latency = 1.0 / tput * 1000.0
            tputs.append(tput)
            latencies.append(latency)
            logger.info(
                f'Step: {cur_step}\t throughput = {tput:0.1f} samples/sec.\t Latency = {latency:0.5f} ms')

            mels_before, mels_after, duration_outputs, f0_outputs, energy_outputs = predictions
        logger.info(
            f"Averaged Metrics:\nthroughput: {int(np.mean(tputs[2:])):d} samples/sec\tlatency avg: {np.mean(latencies[2:]):.2f} ms")


if __name__ == "__main__":
    opts = make_global_options([])
    run_model(opts)
